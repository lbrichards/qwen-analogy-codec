#!/usr/bin/env python3
"""Train the 2D analogy codec from extracted representations."""

from __future__ import annotations

import argparse

from qwen_analogy_codec.paths import DEFAULT_ARTIFACT_DIR
from qwen_analogy_codec.train import train_from_reps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", default=str(DEFAULT_ARTIFACT_DIR / "reps.pt"))
    parser.add_argument("--codec-base", default=str(DEFAULT_ARTIFACT_DIR / "codec"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_reps(
        reps_path=args.reps,
        codec_base_path=args.codec_base,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
