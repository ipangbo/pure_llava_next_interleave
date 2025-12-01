# LLaVA-NeXT-Interleave (extracted)

Minimal, standalone code to run inference and evaluation for LLaVA-NeXT-Interleave models using Python 3.13 and `transformers` 4.57+. Data assets (`data/`) and weights (`llava-qwen-7b-dpo/` or HF downloads) are intentionally left outside this folder.

## Setup
- Run commands from this folder so `python` can pick up the local `llava_next_interleave` package (`export PYTHONPATH=.` if you invoke things from elsewhere).
- Activate the `llava` conda env (requested): `conda activate llava`.
- Install deps: `pip install -r requirements.txt`.
- Place your checkpoint somewhere accessible (HF repo id or local path) and ensure the interleave benchmark images follow the `interleave_data` layout described in `docs/LLaVA-NeXT-Interleave.md` (Split1/2 and JSON metadata).

## Quick evaluation
Run all three interleave splits in one go (set `TEMPERATURE` to override the default):
```bash
bash scripts/eval_all.sh <ckpt_path_or_repo> <path_to_interleave_data>
```

Single split:
```bash
bash scripts/eval_interleave_3d.sh <ckpt_path_or_repo> <path_to_interleave_data> multi_image_in_domain
```
Predictions land under `logs/<ckpt_name>/<split>/result.jsonl`; metrics are written to `eval_*.json` in the same folder.

## GPU selection
- Default CUDA device list lives in `constants.py` (`DEFAULT_CUDA_DEVICES`) so you can pin a specific GPU when `CUDA_VISIBLE_DEVICES` is unset.
- Override with env `LLAVA_CUDA_DEVICES` or `CUDA_VISIBLE_DEVICES`. Evaluation now runs sequentially on a single visible GPU (no multiprocess fan-out).

## Direct Python entrypoint
```bash
python -m eval.interleave_vqa \
  --model-path <ckpt_path_or_repo> \
  --image-folder <path_to_interleave_data> \
  --question-file <path_to_split_json> \
  --answers-file logs/result.jsonl \
  --device-map auto \
  --attn-implementation sdpa
```

Defaults favour `sdpa` attention to drop the legacy FlashAttention dependency; switch to `flash_attention_2` if you have it installed. The loader accepts 4-bit/8-bit via the scripts by setting `BITSANDBYTES` environment if desired.
