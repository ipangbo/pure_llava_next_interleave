import argparse
import json
import os
import re
from typing import Dict

import shortuuid
import torch
import transformers
from PIL import Image
from tqdm import tqdm

from constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from conversation import SeparatorStyle, conv_templates
from mm_utils import KeywordsStoppingCriteria, get_model_name_from_path
from model.builder import load_pretrained_model
from utils import disable_torch_init


def build_generation_kwargs(args, stopping_criteria, image_tensors):
    """Build generation kwargs while avoiding sampling warnings when do_sample=False."""
    do_sample = args.temperature > 0 or (args.top_p is not None and args.top_p < 1.0)
    gen_kwargs = {
        "images": image_tensors,
        "num_beams": args.num_beams,
        "max_new_tokens": 1024,
        "use_cache": True,
        "stopping_criteria": [stopping_criteria],
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        if args.temperature is not None:
            gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
    else:
        gen_kwargs["do_sample"] = False
        # Reset sampling-related knobs to defaults to silence warnings in non-sampling mode.
        gen_kwargs.update(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            typical_p=1.0,
        )
    return gen_kwargs

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["sample_id"]
        question_type = line["metadata"]["question_type"]
        dataset_name = line["metadata"]["dataset"]
        gt = line["conversations"][1]["value"]

        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        cur_prompt = args.extra_prompt + qs

        conv_mode = "qwen_1_5"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = preprocess_qwen([line["conversations"][0], {'from': 'gpt', 'value': None}], tokenizer, has_image=True).to(device)
        img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

        image_tensors = []
        for image_file in image_files:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            image_tensors.append(image_tensor.to(device=device, dtype=model_dtype))

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                **build_generation_kwargs(args, stopping_criteria, image_tensors),
            )

        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "dataset": dataset_name,
                                   "sample_id": idx,
                                   "prompt": cur_prompt,
                                   "pred_response": outputs,
                                   "gt_response": gt,
                                   "shortuuid": ans_id,
                                   "model_id": model_name,
                                   "question_type": question_type,
                                   }) + "\n")
        ans_file.flush()

        if len(line["conversations"]) > 2:

            for i in range(2, len(line["conversations"]), 2):
                input_ids = torch.cat((input_ids, output_ids), dim=1)

                gt = line["conversations"][i + 1]["value"]
                qs = line["conversations"][i]["value"]
                cur_prompt = args.extra_prompt + qs

                conv_mode = "qwen_1_5"

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids_new = preprocess_qwen([line["conversations"][i], {'from': 'gpt', 'value': None}], tokenizer, has_image=True).to(device)
                input_ids = torch.cat((input_ids, input_ids_new), dim=1)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        **build_generation_kwargs(args, stopping_criteria, image_tensors),
                    )
        
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                                        "dataset": dataset_name,
                                        "sample_id": idx,
                                        "prompt": cur_prompt,
                                        "pred_response": outputs,
                                        "gt_response": gt,
                                        "shortuuid": ans_id,
                                        "model_id": model_name,
                                        "question_type": question_type,
                                        }) + "\n")
                ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path or repo id of the interleave checkpoint.")
    parser.add_argument("--model-base", type=str, default=None, help="Optional base LLM if loading LoRA/MM projector separately.")
    parser.add_argument("--image-folder", type=str, required=True, help="Root folder containing benchmark images.")
    parser.add_argument("--extra-prompt", type=str, default="", help="Prefix injected before every question.")
    parser.add_argument("--question-file", type=str, required=True, help="JSON file with interleave benchmark questions.")
    parser.add_argument("--answers-file", type=str, default="logs/result.jsonl", help="Path to write predictions.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", help="torch dtype for loading the model.")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention backend passed to transformers.")
    parser.add_argument("--device-map", type=str, default="auto", help="device_map passed to transformers (e.g., 'auto' or 'cuda').")
    args = parser.parse_args()

    eval_model(args)
