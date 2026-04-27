#!/usr/bin/env python3
"""Validate the saved analogy codec artifacts."""

from __future__ import annotations

import argparse

from qwen_analogy_codec.paths import DEFAULT_ARTIFACT_DIR
from qwen_analogy_codec.validate import validate_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", default=str(DEFAULT_ARTIFACT_DIR / "reps.pt"))
    parser.add_argument("--codec", default=str(DEFAULT_ARTIFACT_DIR / "codec.pkl"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_gates(
        reps_path=args.reps,
        codec_path=args.codec,
        bootstrap_samples=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
