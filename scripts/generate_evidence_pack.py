#!/usr/bin/env python3
"""Generate CSV and PNG evidence files for the analogy codec."""

from __future__ import annotations

import argparse

from qwen_analogy_codec.evidence import generate_evidence_pack
from qwen_analogy_codec.paths import DEFAULT_ARTIFACT_DIR, DEFAULT_EVIDENCE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", default=str(DEFAULT_ARTIFACT_DIR / "reps.pt"))
    parser.add_argument("--codec", default=str(DEFAULT_ARTIFACT_DIR / "codec.pkl"))
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--train-log", default=str(DEFAULT_ARTIFACT_DIR / "train_log.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_evidence_pack(
        reps_path=args.reps,
        codec_path=args.codec,
        evidence_dir=args.evidence_dir,
        train_log_path=args.train_log,
    )
    print(f"Evidence pack written to {args.evidence_dir}")


if __name__ == "__main__":
    main()
