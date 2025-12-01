# LLaVA-NeXT-Interleave 代码库详解

> 本仓库是一个精简版的 LLaVA‑NeXT‑Interleave 推理与评测脚本集合，基于 Python 3.13 与 `transformers` 4.57+。  
> 主要用途：在给定多图像交错理解（interleave）基准数据集和已训练好的权重的前提下，进行推理并计算评测指标。

---

## 1. 目录结构与每个文件的作用

只列出与推理 / 评测相关的代码与脚本，其它如 `__pycache__/`、`.git/` 等略去。

### 顶层结构

- `README.md`  
  - 英文 README，给出最小使用示例（`scripts/eval_all.sh` 等）和环境要求。  
  - 本文档是更详细的中文补充说明。

- `requirements.txt`  
  - Python 依赖列表：`torch`、`transformers`、`accelerate`、`tqdm`、`shortuuid`、`pillow`、`numpy`、`scikit-learn`、`rouge`，以及可选的 `bitsandbytes`（4/8bit 量化加载）。

- `__init__.py`  
  - 将本目录声明为 Python 包 `llava_next_interleave`，并尝试读取已安装包的 `__version__`。  
  - 对于直接从源码运行脚本来说，只需要把当前目录加入 `PYTHONPATH` 即可（脚本已经帮你做了）。

- `constants.py`  
  - 全局常量与默认配置：
    - GPU 相关：`DEFAULT_CUDA_DEVICES`（在未显式设置 `CUDA_VISIBLE_DEVICES` 时决定默认使用哪些 GPU，单卡顺序运行）。  
    - 模型相关：`IGNORE_INDEX`（损失忽略标签）、`IMAGE_TOKEN_INDEX`、`DEFAULT_IMAGE_TOKEN`、`DEFAULT_IM_START_TOKEN`、`DEFAULT_IM_END_TOKEN` 等。  
  - `scripts/eval_interleave_3d.sh` 会通过一个小的 Python 片段读取这些值。

- `conversation.py`  
  - 会话模板与对话格式工具类：
    - 定义 `SeparatorStyle` 枚举：`SINGLE`、`TWO`、`CHATML`、`QWEN` 等不同对话分隔风格。  
    - 核心类 `Conversation`：维护 system 提示词、角色（user / assistant）、消息列表以及分隔符；提供 `append_message()`、`get_prompt()`、`get_images()` 等方法。  
    - 预置大量常见模型的对话模板，如 `conv_vicuna_v0`、`conv_llama_2`、`conv_qwen` 等。  
  - 底部的 `conv_templates` 字典是关键：`eval/interleave_vqa.py` 中用 `conv_templates["qwen_1_5"]` 来构造多轮问答提示。

- `mm_utils.py`  
  - 多模态 / 图像相关的通用工具：
    - 高分辨率图像处理：`process_highres_image_*`、`process_anyres_image()` 等，用于将任意分辨率图像裁剪 / 分块成统一 patch；  
    - 图像 token 处理：`tokenizer_image_token()` 将 `<image>` 占位符替换为 `IMAGE_TOKEN_INDEX` 插入到文本 token 序列中；  
    - 计算任意分辨率下网格形状：`get_anyres_image_grid_shape()`；  
    - 利用 processor 进行预处理：`process_images()`。  
  - 文本生成相关：
    - `get_model_name_from_path()`：从 checkpoint 路径中提取模型名，写入预测结果。  
    - `KeywordsStoppingCriteria`：自定义 `transformers.StoppingCriteria`，在生成过程中一旦出现指定分隔符（例如对话模板的 `sep` / `sep2`），就停止生成。

- `utils.py`  
  - 杂项工具：
    - Torch 初始化：`disable_torch_init()` 覆盖 `torch.nn.Linear/LayerNorm.reset_parameters`，避免冗余初始化，加快模型创建。`eval/interleave_vqa.py` 在加载模型前会调用。  
    - 日志工具：`build_logger()`、`rank0_print()`、`rank_print()` 等。  
    - 图像 / 视频加载、OpenAI 审核 API 等函数在本简化评估流程中基本不会直接用到，但保留以兼容原版 LLaVA 代码结构。

- `model/`  
  - 模型构建与多模态模块所在的子包。

### model 子包

- `model/__init__.py`  
  - 将 `load_pretrained_model` 作为对外公开接口导出。

- `model/builder.py`  
  - 这是加载模型的统一入口，供 `eval/interleave_vqa.py` 调用。主要职责：
    - 根据 `torch_dtype` 字符串解析出实际的 `torch.dtype`（`_resolve_dtype()`）；  
    - 构造 `BitsAndBytesConfig`（如 `load_in_4bit` 时）等加载参数（`_build_load_kwargs()`）；  
    - 准备 LLaVA‑Qwen 配置，并设置 `delay_load=True`（`_prepare_config()`），以便推迟 vision tower 的加载；  
    - 根据是否提供 `model_base`：
      - 若提供：`model_base` 作为语言模型权重，`model_path` 仅提供配置和多模态 projector 权重；  
      - 否则：直接从 `model_path` 加载完整多模态模型。  
    - 加载并注册额外的图像 token：`DEFAULT_IMAGE_PATCH_TOKEN`、`DEFAULT_IM_START_TOKEN`、`DEFAULT_IM_END_TOKEN`，然后 `model.resize_token_embeddings()`。  
    - 通过 `model.get_vision_tower()` 获取视觉塔，必要时加载并移动到正确 device / dtype，同时得到 `image_processor`。  
    - 决定上下文长度 `context_len`（优先使用 `max_sequence_length` / `max_position_embeddings` 等配置项）。  
  - 返回 `(tokenizer, model, image_processor, context_len)` 四元组。

- `model/llava_arch.py`  
  - 定义多模态架构骨架 `LlavaMetaModel` 和 `LlavaMetaForCausalLM`：  
    - 负责装配 vision tower（`build_vision_tower`）、resampler（`build_vision_resampler`）、projector（`build_vision_projector`）；  
    - 提供 `prepare_inputs_labels_for_multimodal()`、`generate()` 等方法，在文本 token 前后插入图像 embedding；  
    - 该文件是从原版 LLaVA 精简而来，但对使用者来说通常无需修改，仅需保证配置与 checkpoint 一致。

- `model/language_model/llava_qwen.py`  
  - 基于 Qwen 的具体语言模型实现：
    - `LlavaQwenConfig`：在原始 Qwen 配置上扩展多模态相关字段（vision tower 类型、图像 patch merge 策略等）；  
    - `LlavaQwenForCausalLM`：继承 `LlavaMetaForCausalLM` + Qwen 语言模型，实现带多模态输入的 `forward` 与 `generate`。  
    - 通过 `AutoConfig.register("llava_qwen", ...)` 与 `AutoModelForCausalLM.register(...)` 注册到 transformers 框架。

- `model/multimodal_encoder/siglip_encoder.py`  
  - 封装 SigLip 视觉编码器为 `SigLipVisionTower`，负责把图片变成视觉特征（patch 序列）。  
  - 对外通过 `build_vision_tower()` 使用。

- `model/multimodal_encoder/builder.py`  
  - `build_vision_tower(vision_tower_cfg, **kwargs)`：
    - 从配置中解析出 `mm_vision_tower` / `vision_tower` 字段；  
    - 当前只支持包含 `siglip` 的配置，返回 `SigLipVisionTower`；否则抛出异常（因为本仓库是“interleave‑only”的精简版）。

- `model/multimodal_resampler/builder.py`  
  - 定义 `IdentityMap`（直接返回输入）和 `build_vision_resampler(...)`。  
  - 当前若未指定 `mm_resampler_type`，则使用 `IdentityMap`，相当于不做 resample；若指定了其它类型则抛出异常（同样是简化版设计）。

- `model/multimodal_projector/builder.py`  
  - 把视觉特征投影到语言模型隐空间的模块构建器：
    - 支持 `linear`、`pooler`、`mlpNx_gelu`、`mlpNx_resMx_gelu`、`identity` 等多种 projector；  
    - `SimpleResBlock` 提供可选残差块。  
  - 由 `llava_arch.py` 和 `builder.py` 间接调用。

- `model/multimodal_projector/pooler_projector.py`  
  - `PoolerProjector` 的具体实现，支持对视觉特征进行池化后再投射到语言模型空间。

### eval 子包

- `eval/__init__.py`  
  - 仅包含版权信息与模块注释，对评测逻辑无实际影响。

- `eval/interleave_vqa.py`  
  - **核心：推理脚本（生成模型回答）**。  
  - 主要内容：
    - 生成参数构造：`build_generation_kwargs()` 根据温度、top_p、beam 数构造 `model.generate` 的参数，同时在非采样模式下重置相关字段以避免 warnings；  
    - Qwen 预处理：`preprocess_qwen()` 构造 ChatML 风格的 <|im_start|>system/user/assistant 对话，并把 `<image>` 占位符替换为特殊的 `IMAGE_TOKEN_INDEX`，返回 `input_ids` 张量；  
    - `eval_model(args)`：  
      1. 调用 `disable_torch_init()` 加速模型加载；  
      2. 使用 `load_pretrained_model()` 加载 tokenizer / model / image_processor；  
      3. 读取 `--question-file`（如 `data/interleave_data/multi_image_in_domain.json`），构成样本列表；  
      4. 为每个样本（单进程顺序执行）：
         - 从 `image_folder` + `image` 字段中读取所有图片，并用 `process_images()` 预处理成 `image_tensors`；  
         - 从 `conversations` 字段中提取当前轮对话（人类问题、模型参考答案），使用 `conv_templates["qwen_1_5"]` 构造 prompt；  
         - 用 `preprocess_qwen()` 将对话转为 `input_ids`；  
         - 构造 `KeywordsStoppingCriteria`，传给 `model.generate()`，并携带 `images=image_tensors`；  
         - 对生成的 token 进行 `batch_decode()` 得到字符串，去除结尾的分隔符；  
         - 写入 `answers_file`（JSONL，每行一个样本），字段包括：
           - `dataset`：数据集名（来自 `metadata["dataset"]`），  
           - `sample_id`：样本 ID，  
           - `prompt`：实际送入模型的文本（含多图 `<image>` 占位符），  
           - `pred_response`：模型预测文本，  
           - `gt_response`：真实答案文本（多选题时是选项字母等），  
           - `question_type`：如 `"open-ended"` 或 `"multi-choice"`，  
           - `model_id`：从 `model_path` 解析出的模型名，  
           - `shortuuid`：样本级唯一 ID。  
         - 若 `conversations` 中包含多轮（长度 > 2），则会循环追加后续轮的问答，每一轮都复用之前的 `input_ids` / `output_ids` 作为上下文，实现真正的多轮视觉对话评测。
    - 命令行接口：在 `__main__` 中定义若干参数：
      - `--model-path`：checkpoint 路径或 HF repo id（必选）；  
      - `--model-base`：可选基础 LLM（如 LoRA 或 projector 单独存储的情况）；  
      - `--image-folder`：图片根目录（如 `data/interleave_data`）；  
      - `--question-file`：题目 JSON 文件；  
      - `--answers-file`：输出结果 JSONL，默认 `logs/result.jsonl`；  
      - 采样 / 解码参数：`--temperature`、`--top_p`、`--num_beams`；  
      - 模型加载参数：`--torch-dtype`、`--attn-implementation`、`--device-map`。

- `eval/evaluate_interleave.py`  
  - **核心：读取 `result.jsonl`，计算指标并汇总**。  
  - `Eval` 类：
    - 文本预处理：`processPunctuation()`、`process()` 去除标点、大小写与空白差异，使答案对比更鲁棒；  
    - 开放式生成评测：`evaluate_rouge(preds)` 使用 `rouge` 包计算每个样本的 Rouge‑L F1，最后取平均；  
    - 多选题评测：  
      - `evaluate_multichoice()` 先调用 `process_sample()` 标准化大小写和标点，再用 `judge_multi_choice()` 判断预测是否等于 GT；  
      - `judge_multi_choice()` 带有一个小 heuristic：若预测中包含 `A:` 或类似形式，会尝试从中抽取单个字母选项；  
      - `evaluate_multi_choice_image()` 针对某些图像选项任务，逻辑类似但先对 GT / pred 都做文本标准化。  
  - `__main__` 中的流程：
    1. 读取 `--result-dir` 下的 `result.jsonl`；  
    2. 将所有预测按 `pred["dataset"]` 分组；  
    3. 遍历每个数据集，根据首个样本的 `question_type` 判定使用 Rouge 还是多选 Accuracy；  
    4. 打印各数据集指标，并写出：
       - `eval_dataset.json`：按数据集聚合的指标，如 `{ "RAVEN": {"Accuracy": 0.85}, ... }`；  
       - `eval_dataset_details.json`：每个数据集内每个样本的得分列表（含 `id` 与 `score`）；  
    5. 按任务类别进一步聚合：
       - 利用文件顶部的 `spot_the_diff`、`image_edit_instruct`、`visual_story_telling`、`visual_cloze`、`text_rich_vqa`、`multi_image_vqa`、`puzzle`、`nlrv2`、`qbench` 等列表，将同一类别下多个数据集的分数平均；  
       - 打印类别级别指标，并写入 `eval_cat.json`。

### scripts 子目录

- `scripts/eval_all.sh`  
  - 一个简单的“总入口”脚本，顺序调用三次 `eval_interleave_3d.sh`：
    - `multi_image_in_domain`  
    - `multi_image_out_domain`  
    - `multi_view_in_domain`  
  - 用法（见 README）：
    ```bash
    bash scripts/eval_all.sh <ckpt_path_or_repo> <path_to_interleave_data>
    ```

- `scripts/eval_interleave_3d.sh`  
  - 针对单个 split 的评估入口，负责数据路径拼接与单 GPU 顺序推理。  
  - 关键步骤：
    1. 设置 `SCRIPT_DIR`、`PYTHONPATH`；  
    2. 读取命令行参数：
       - `$1` → `CKPT_PATH`：模型 checkpoint 路径；  
       - `$2` → `DATA_PATH`：interleave 数据根目录；  
       - `$3` → `EVAL_TYPE`：例如 `multi_image_in_domain`；  
       - 构造 `JSON_PATH="${DATA_PATH}/${EVAL_TYPE}.json"`、结果目录 `RESULT_NAME="logs/<ckpt_name>/<EVAL_TYPE>"`。  
    3. 若未设置 `CUDA_VISIBLE_DEVICES`，则读取 `constants.DEFAULT_CUDA_DEVICES` 作为默认设备；  
    4. 设置默认温度 `TEMPERATURE=${TEMPERATURE:-0}`；  
    5. 直接运行单进程推理：
       ```bash
       python3 -m eval.interleave_vqa \
         --model-path "${CKPT_PATH}" \
         --question-file "${JSON_PATH}" \
         --answers-file "${RESULT_NAME}/result.jsonl" \
         --image-folder "${DATA_PATH}" \
         --extra-prompt "" \
         --temperature "${TEMPERATURE}"
       ```
    6. 推理结束后调用 `python3 -m eval.evaluate_interleave --result-dir "${RESULT_NAME}"` 计算指标。

---

## 2. 数据与结果文件格式

### 2.1 Interleave 数据目录

默认的数据目录位于：`data/interleave_data/`，内部结构包括：

- 顶层 JSON：
  - `multi_image_in_domain.json`  
  - `multi_image_out_domain.json`  
  - `multi_view_in_domain.json`  
- 图像子目录：
  - `Split1/`、`Split2/` 等，其中包含按数据集名划分的子目录（例如 `RAVEN_val_images`、`AESOP`、`HQ-Edit` 等）。

以 `multi_image_in_domain.json` 为例，每个元素大致如下（简化）：

```json
{
  "sample_id": 1,
  "conversations": [
    {
      "from": "human",
      "value": "Here is a Raven's Progressive Matrice ... <image> <image> ..."
    },
    {
      "from": "gpt",
      "value": "C"
    }
  ],
  "image": [
    "Split1/RAVEN_val_images/1.jpg",
    "Split1/RAVEN_val_images/2.jpg",
    "... more image paths ..."
  ],
  "metadata": {
    "dataset": "RAVEN",
    "split": "val",
    "num_sample": 1400,
    "task_instruction": "...",
    "question_type": "multi-choice"
  }
}
```

- `conversations`：一个包含多轮对话的列表，每两个元素是一轮 `[提问, 回答]`，后续轮也遵循 human / gpt 交替。  
- `image`：相对 `image_folder` 的图片路径列表。  
- `metadata`：
  - `dataset`：数据集名，用于后续评测分组；  
  - `question_type`：`"open-ended"` 或 `"multi-choice"`；  
  - 其它字段仅用于记录信息。

### 2.2 推理结果文件 `result.jsonl`

单进程推理结束后，`scripts/eval_interleave_3d.sh` 会在 `logs/<ckpt_name>/<split>/` 目录下生成 `result.jsonl`。

每行是一个 JSON 对象，大致结构为：

```json
{
  "dataset": "RAVEN",
  "sample_id": 1,
  "prompt": "<完整的文本 prompt，含 <image> 占位符>",
  "pred_response": "C",
  "gt_response": "C",
  "shortuuid": "....",
  "model_id": "llava-qwen-7b-dpo",
  "question_type": "multi-choice"
}
```

若某个样本包含多轮对话，则每一轮都会追加一行（`sample_id` 相同，`prompt` 和 `gt_response` 不同）。

### 2.3 评测结果文件

执行 `python -m eval.evaluate_interleave --result-dir logs/<ckpt_name>/<split>` 后，会在该目录生成：

- `eval_dataset.json`  
  - 以数据集为键的字典，例如：
    ```json
    {
      "RAVEN": {
        "Accuracy": 0.85
      },
      "Spot-the-Diff": {
        "Accuracy": 0.90
      },
      "DocVQA": {
        "Rouge-L f": 0.42
      }
    }
    ```

- `eval_dataset_details.json`  
  - 对每个数据集，记录每个样本的得分：
    ```json
    {
      "RAVEN": [
        {"id": "1", "score": "1"},
        {"id": "17", "score": "0"}
      ],
      "DocVQA": [
        {"id": "42", "score": "0.387"},
        ...
      ]
    }
    ```

- `eval_cat.json`  
  - 按类别聚合后的平均分，例如：
    ```json
    {
      "spot_the_diff": 0.91,
      "image_edit_instruct": 0.73,
      "visual_story_telling": 0.41,
      "visual_cloze": 0.56,
      "text_rich_vqa": 0.38,
      "multi_image_vqa": 0.62,
      "puzzle": 0.84,
      "nlrv2": 0.78,
      "qbench": 0.65
    }
    ```

---

## 3. 端到端数据 / 调用流程

下面以最常见的“跑三个 split 的全量评测”为例，说明数据和配置在各个脚本与模块之间的流动路径。

1. **用户调用 `scripts/eval_all.sh`**  
   ```bash
   bash scripts/eval_all.sh llava-qwen-7b-dpo data/interleave_data
   ```  
   - 依次调用：
     - `bash scripts/eval_interleave_3d.sh llava-qwen-7b-dpo data/interleave_data multi_image_in_domain`  
     - `bash scripts/eval_interleave_3d.sh llava-qwen-7b-dpo data/interleave_data multi_image_out_domain`  
     - `bash scripts/eval_interleave_3d.sh llava-qwen-7b-dpo data/interleave_data multi_view_in_domain`。

2. **`scripts/eval_interleave_3d.sh`：配置环境 + 单进程推理**  
   - 解析三个位置参数：`CKPT_PATH`、`DATA_PATH`、`EVAL_TYPE`；  
   - 构造 `JSON_PATH="${DATA_PATH}/${EVAL_TYPE}.json"`（数据文件），`RESULT_NAME="logs/<ckpt_name>/<EVAL_TYPE>"`（结果目录）；  
   - 若用户未设置 `CUDA_VISIBLE_DEVICES`，则从 `constants.DEFAULT_CUDA_DEVICES` 读取默认值并导出；  
   - 直接调用 `python3 -m eval.interleave_vqa` 在单个 GPU 上顺序跑完整个 split，写出 `result.jsonl`；  
   - 推理完成后，调用 `python3 -m eval.evaluate_interleave` 计算评测指标。

3. **`eval/interleave_vqa.py`：单进程视角下的推理流程**  
   - `eval_model(args)` 中：
     1. 调用 `disable_torch_init()` 减少不必要的权重初始化；  
     2. 通过 `load_pretrained_model()` 加载 tokenizer / model / image_processor；  
     3. 读取问题 JSON 文件，得到样本列表 `lines`；  
     4. 对于每个样本（顺序处理）：
        - 用 `PIL.Image.open()` 加载 `image` 字段中列出的图片，并交给 `process_images()` 处理成统一形状的 `image_tensors`；  
        - 取首轮对话的 human 问题和 gpt 答案，利用 `conv_templates["qwen_1_5"]`、`preprocess_qwen()` 构造输入 `input_ids`；  
        - 构造 `KeywordsStoppingCriteria` 以对话分隔符为停止条件；  
        - 调用 `model.generate(input_ids, images=image_tensors, ...)` 生成回答；  
        - 解码为字符串，去除尾部分隔符，写入 JSONL；  
        - 若有多轮对话，会在循环中不断拼接新的 `input_ids` 和 `output_ids`，形成上下文累积，再次调用 `generate()` 并写入对应轮次结果。

5. **`eval/evaluate_interleave.py`：从结果到指标**  
   - 读取 `result.jsonl`，按 `dataset` 分组；  
   - 对每个数据集：
     - 若 `question_type == "open-ended"`：用 Rouge-L f 评估生成质量；  
     - 若是 `"multi-choice"` 或 `dataset == "nlrv2"`：用多选 Accuracy 评估（对于 image choice 数据集使用 `evaluate_multi_choice_image()`）；  
     - 把该数据集的整体结果与逐样本结果存入字典。  
   - 输出三类 JSON 文件：  
     - `eval_dataset.json`：逐数据集指标；  
     - `eval_dataset_details.json`：逐样本得分；  
     - `eval_cat.json`：按任务类别平均指标。

---

## 4. 配置与可调参数

### 4.1 GPU 相关环境变量

- 在 `scripts/eval_interleave_3d.sh` 中：
  - 若用户 **未设置** `CUDA_VISIBLE_DEVICES`，则自动从 `constants.DEFAULT_CUDA_DEVICES` 读取（例如 `"0"`）并导出；  
  - 若用户设置了 `CUDA_VISIBLE_DEVICES`，则以用户设置为准（单进程、顺序执行）。  

- 推荐用法示例：
  ```bash
  # 在指定 GPU 上运行单进程推理
  CUDA_VISIBLE_DEVICES=1 \
    bash scripts/eval_interleave_3d.sh llava-qwen-7b-dpo data/interleave_data multi_image_in_domain
  ```

### 4.2 模型加载与解码参数

直接使用 Python 入口：

```bash
python -m eval.interleave_vqa \
  --model-path llava-qwen-7b-dpo \
  --image-folder data/interleave_data \
  --question-file data/interleave_data/multi_image_in_domain.json \
  --answers-file logs/demo_result.jsonl \
  --device-map auto \
  --attn-implementation sdpa \
  --temperature 0 \
  --num-beams 1
```

常用参数说明：

- `--model-path` / `--model-base`：  
  - 若只给 `--model-path`：认为该路径下包括完整的多模态权重与配置；  
  - 若同时给 `--model-base`：`model-base` 仅提供语言模型部分权重，`model-path` 提供配置与 projector 权重，适合 LoRA 类场景。

- `--torch-dtype`：`"bfloat16"`（默认）、`"float16"`、`"float32"` 等。  
- `--attn-implementation`：`"sdpa"`（默认）、`"flash_attention_2"`、`"eager"`。  
- `--device-map`：`"auto"` 或 `"cuda"`，也可以用 dict 形式指定到具体设备。  
- `--temperature`、`--top_p`、`--num_beams`：控制生成多样性的标准参数。  
- 若使用 `bitsandbytes` 进行 4bit/8bit 加载，可通过环境变量或修改 `load_pretrained_model()` 参数来启用（例如传入 `load_4bit=True` 等）。

---

## 5. 小结

- **核心调用链**：  
  `scripts/eval_all.sh` → `scripts/eval_interleave_3d.sh` → `eval/interleave_vqa.py`（推理） → `eval/evaluate_interleave.py`（评测）。  
- **数据流向**：  
  `data/interleave_data/*.json` + 图像 → 单 GPU 顺序推理 → `logs/<ckpt>/<split>/result.jsonl` → 评测 → `eval_dataset*.json`、`eval_cat.json`。  
- **关键模块**：  
  - `model/`：负责加载 LLaVA‑Qwen 多模态模型与视觉编码器；  
  - `conversation.py`、`mm_utils.py`：封装对话与图像处理细节；  
  - `eval/interleave_vqa.py`：将数据集样本转换为多轮多图像对话输入模型；  
  - `eval/evaluate_interleave.py`：把模型输出转换为定量指标。

通过阅读本文件并结合 `README.md` 中的命令示例，基本可以完整理解该代码库在 **配置、数据流、模型加载与评测** 各环节的设计与用法。 
