import json
import math
import os
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image


class LLavaInterleaveDataset:
    """Robust loader for the LLaVA-Interleave benchmark.

    This class takes care of:
    - Reading the question JSON file.
    - Optionally slicing the dataset into chunks (for multi-worker setups).
    - Loading and preprocessing images into tensors on the target device/dtype.
    - Providing a unified sample dict interface for evaluation scripts.
    """

    def __init__(
        self,
        *,
        question_file: str,
        image_folder: str,
        image_processor,
        device: torch.device,
        model_dtype: torch.dtype,
        extra_prompt: str = "",
        num_chunks: int = 1,
        chunk_idx: int = 0,
    ) -> None:
        self.question_file = os.path.expanduser(question_file)
        self.image_folder = os.path.expanduser(image_folder)
        self.image_processor = image_processor
        self.device = device
        self.model_dtype = model_dtype
        self.extra_prompt = extra_prompt

        raw_samples = self._load_questions(self.question_file)
        self.samples = self._slice_chunk(raw_samples, num_chunks, chunk_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw = self.samples[index]
        return self._prepare_sample(raw)

    @staticmethod
    def _load_questions(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Question file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Question file must contain a list of samples. Got type: {type(data)}")
        return data

    @staticmethod
    def _split_list(lst: List[Any], n: int) -> List[List[Any]]:
        if n <= 0:
            raise ValueError("num_chunks must be >= 1")
        chunk_size = math.ceil(len(lst) / n)
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def _slice_chunk(self, samples: List[Dict[str, Any]], num_chunks: int, chunk_idx: int) -> List[Dict[str, Any]]:
        if num_chunks == 1:
            return samples
        chunks = self._split_list(samples, num_chunks)
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            raise IndexError(f"chunk_idx {chunk_idx} out of range for {len(chunks)} chunks")
        return chunks[chunk_idx]

    def _load_images(self, image_files: List[str]) -> List[torch.Tensor]:
        tensors: List[torch.Tensor] = []
        for image_file in image_files:
            abs_path = os.path.join(self.image_folder, image_file)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Image not found: {abs_path}")
            with Image.open(abs_path) as img:
                image = img.convert("RGB")
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            tensors.append(image_tensor.to(device=self.device, dtype=self.model_dtype))
        return tensors

    def _prepare_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        conversations = record["conversations"]
        question = conversations[0]["value"]
        gt_response = conversations[1]["value"]
        metadata = record.get("metadata", {})

        sample = {
            "sample_id": record.get("sample_id"),
            "dataset_name": metadata.get("dataset"),
            "question_type": metadata.get("question_type"),
            "conversations": conversations,
            "question": question,
            "gt_response": gt_response,
            "prompt": f"{self.extra_prompt}{question}",
            "image_paths": [os.path.join(self.image_folder, p) for p in record.get("image", [])],
        }

        image_files = record.get("image", [])
        sample["image_tensors"] = self._load_images(image_files)

        return sample
