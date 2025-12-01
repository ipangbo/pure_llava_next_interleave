import inspect
import os
from typing import Any, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from model.language_model import LlavaQwenConfig, LlavaQwenForCausalLM
from utils import rank0_print


def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if hasattr(torch, dtype):
            return getattr(torch, dtype)
        if dtype in {"half", "float16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
    return dtype


def _build_load_kwargs(load_8bit: bool, load_4bit: bool, torch_dtype: Union[str, torch.dtype, None]) -> dict[str, Any]:
    dtype = _resolve_dtype(torch_dtype)
    kwargs: dict[str, Any] = {}
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs


def _prepare_config(config_path: str) -> LlavaQwenConfig:
    config = LlavaQwenConfig.from_pretrained(config_path)
    # Avoid loading the vision tower inside init_empty_weights; we load it explicitly later.
    config.delay_load = True
    return config


def load_pretrained_model(
    model_path: str,
    model_base: Optional[str],
    model_name: Optional[str],
    *,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: Union[str, dict, None] = "auto",
    torch_dtype: Union[str, torch.dtype, None] = "bfloat16",
    attn_implementation: str = "sdpa",
    **kwargs: Any,
) -> Tuple[Any, LlavaQwenForCausalLM, Any, int]:
    """Load a pretrained interleave model along with tokenizer and image processor."""

    # Ensure the custom config/model classes are registered
    _ = LlavaQwenConfig

    load_kwargs = _build_load_kwargs(load_8bit, load_4bit, torch_dtype)
    load_kwargs["device_map"] = device_map
    tokenizer = AutoTokenizer.from_pretrained(model_base or model_path, use_fast=False, trust_remote_code=True)

    if model_base:
        config = _prepare_config(model_path)
        model = LlavaQwenForCausalLM.from_pretrained(
            model_base,
            config=config,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        projector_path = os.path.join(model_path, "mm_projector.bin")
        if os.path.exists(projector_path):
            projector_weights = torch.load(projector_path, map_location="cpu")
            load_args = {"strict": False}
            if "assign" in inspect.signature(model.load_state_dict).parameters:
                load_args["assign"] = True
            model.load_state_dict(projector_weights, **load_args)
    else:
        config = _prepare_config(model_path)
        model = LlavaQwenForCausalLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **load_kwargs,
        )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    load_map = None if device_map == "auto" else device_map
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=load_map)

    default_device = next(model.parameters()).device
    if load_map is None:
        vision_tower.to(device=default_device, dtype=model.dtype)
    elif isinstance(load_map, str) and load_map != "cpu":
        vision_tower.to(device=load_map, dtype=model.dtype)

    image_processor = vision_tower.image_processor

    # Debug: confirm vision tower is fully loaded
    print("Vison Tower:\n", vision_tower)
    print("Vison Tower Params:\n", sum(p.numel() for p in vision_tower.parameters()))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    rank0_print(f"Loaded model from {model_path} with context length {context_len}")
    return tokenizer, model, image_processor, context_len
