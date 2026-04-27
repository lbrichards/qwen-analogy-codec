"""Local Hugging Face model loading helpers."""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_offline_mode(enable: bool = True) -> None:
    """Prefer local model files and avoid accidental downloads."""
    if enable:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)


def select_device_and_dtype(
    device_arg: Optional[str] = None,
    dtype_arg: Optional[str] = None,
) -> tuple[str, torch.dtype]:
    """Select a practical device and dtype for local extraction."""
    device = device_arg
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if dtype_arg is not None:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = dtype_arg.lower()
        if key not in mapping:
            raise ValueError("Unsupported dtype. Choose float32, float16, or bfloat16.")
        dtype = mapping[key]
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    return device, dtype


def load_local_model_and_tokenizer(model_path: str, device: str, dtype: torch.dtype):
    """Load a local causal LM and tokenizer without downloading weights."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path not found or not a directory: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if device in ("cuda", "mps"):
        model.to(device)
    model.eval()
    return model, tokenizer
