#!/usr/bin/env python3
"""Extract hidden-state representations for the target analogy words."""

from __future__ import annotations

import argparse
import os

from qwen_analogy_codec.extract import TARGET_WORDS, extract_representations
from qwen_analogy_codec.paths import DEFAULT_ARTIFACT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LOCAL_HF_MODEL_PATH", "/Users/larry/Development/Qwen2.5-3B-Instruct"),
        help="Local Hugging Face model directory.",
    )
    parser.add_argument("--out", default=str(DEFAULT_ARTIFACT_DIR / "reps.pt"))
    parser.add_argument("--layer-offset", type=int, default=10)
    parser.add_argument("--samples-per-word", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--words", nargs="+", default=TARGET_WORDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_representations(
        model_path=args.model_path,
        output_path=args.out,
        target_words=args.words,
        layer_offset=args.layer_offset,
        n_samples_per_word=args.samples_per_word,
        random_seed=args.seed,
        device_arg=args.device,
        dtype_arg=args.dtype,
        offline=not args.no_offline,
    )
    print(f"Saved representations to {args.out}")


if __name__ == "__main__":
    main()
