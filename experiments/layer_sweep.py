#!/usr/bin/env python3
"""Sweep Qwen hidden layers for analogy-codec readability."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from loss_ablation import train_with_weights, evaluate
from qwen_analogy_codec.extract import TARGET_WORDS, TEMPLATES, find_word_span
from qwen_analogy_codec.model_io import (
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)


CLASSIFICATION_ONLY_WEIGHTS = {
    "cls": 1.0,
    "analogy": 0.0,
    "perp": 0.0,
    "orth": 0.0,
    "sep": 0.0,
}


def make_sentences(words: list[str], samples_per_word: int, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    sentences = {}
    for word in words:
        word_sentences = []
        for _ in range(samples_per_word):
            template = rng.choice(TEMPLATES)
            surface = word.lower() if rng.random() < 0.5 else word
            word_sentences.append(template.format(w=surface))
        sentences[word] = word_sentences
    return sentences


@torch.no_grad()
def extract_all_layers(
    model,
    tokenizer,
    *,
    sentences_by_word: dict[str, list[str]],
    device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Extract target-token states for every decoder-layer hidden state."""
    layer_data: dict[int, dict[str, list[torch.Tensor]]] = {}

    for word, sentences in sentences_by_word.items():
        print(f"Extracting all layers for {word!r}...")
        for sentence in sentences:
            start_tok, end_tok = find_word_span(tokenizer, sentence, word)
            if start_tok is None or end_tok is None:
                continue

            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # hidden_states[0] is the embedding output. Indices 1..N correspond
            # to decoder-layer outputs, so those become the reported layer ids.
            for layer_index in range(1, len(hidden_states)):
                hidden = hidden_states[layer_index].squeeze(0)
                layer_data.setdefault(layer_index, {}).setdefault(word, []).append(
                    hidden[end_tok - 1].detach().cpu().float()
                )

    packed = {}
    for layer_index, by_word in layer_data.items():
        packed[layer_index] = {word: torch.stack(by_word[word]) for word in TARGET_WORDS}
    return packed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LOCAL_HF_MODEL_PATH", "/Users/larry/Development/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--samples-per-word", type=int, default=100)
    parser.add_argument("--extract-seed", type=int, default=1337)
    parser.add_argument("--train-seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--train-device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--out-csv", default="results/layer_sweep.csv")
    parser.add_argument("--out-json", default="results/layer_sweep.json")
    parser.add_argument("--out-reps", default=None, help="Optional path for the large all-layer reps cache.")
    parser.add_argument("--figure-png", default="figures/layer_sweep.png")
    parser.add_argument("--figure-pdf", default="figures/layer_sweep.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_mode(not args.no_offline)
    device, dtype = select_device_and_dtype(args.device, args.dtype)
    train_device = args.train_device or ("cuda" if torch.cuda.is_available() else "cpu")

    sentences_by_word = make_sentences(TARGET_WORDS, args.samples_per_word, args.extract_seed)
    print(f"Loading model from {args.model_path} on {device} with dtype={dtype}...")
    model, tokenizer = load_local_model_and_tokenizer(args.model_path, device, dtype)
    layer_reps = extract_all_layers(
        model,
        tokenizer,
        sentences_by_word=sentences_by_word,
        device=device,
    )

    rows = []
    for layer_index in sorted(layer_reps):
        print(f"Training codec for layer {layer_index}...")
        W_e, b_e = train_with_weights(
            layer_reps[layer_index],
            weights=CLASSIFICATION_ONLY_WEIGHTS,
            seed=args.train_seed,
            steps=args.steps,
            lr=args.lr,
            device=train_device,
        )
        metrics = evaluate(W_e, b_e, layer_reps[layer_index], split_seed=args.split_seed)
        layer_offset = max(layer_reps) - layer_index
        row = {
            "layer_index": layer_index,
            "layer_offset_from_final": layer_offset,
            "analogy_cos": metrics["analogy_cos"],
            "heldout_acc": metrics["heldout_acc"],
            "axis_angle_deg": metrics["axis_angle_deg"],
            "steps": args.steps,
            "lr": args.lr,
            "train_seed": args.train_seed,
            "split_seed": args.split_seed,
        }
        rows.append(row)
        print(
            f"  analogy={row['analogy_cos']:.4f}, "
            f"accuracy={row['heldout_acc']:.1%}, "
            f"angle={row['axis_angle_deg']:.1f}"
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_reps = Path(args.out_reps) if args.out_reps else None
    figure_png = Path(args.figure_png)
    figure_pdf = Path(args.figure_pdf)
    for path in [out_csv, out_json, figure_png, figure_pdf]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if out_reps is not None:
        out_reps.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if out_reps is not None:
        torch.save(
            {
                "reps_by_layer": layer_reps,
                "meta": {
                    "model": args.model_path,
                    "samples_per_word": args.samples_per_word,
                    "extract_seed": args.extract_seed,
                    "words": TARGET_WORDS,
                },
            },
            out_reps,
        )

    xs = np.array([row["layer_index"] for row in rows])
    analogy = np.array([row["analogy_cos"] for row in rows])
    accuracy = np.array([row["heldout_acc"] for row in rows])

    plt.figure(figsize=(9, 5))
    plt.plot(xs, analogy, marker="o", linewidth=1.8, label="Analogy cosine")
    plt.plot(xs, accuracy, marker="s", linewidth=1.8, label="Held-out accuracy")
    plt.axvline(26, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="Original layer")
    plt.ylim(-0.35, 1.05)
    plt.xlabel("Decoder layer output index")
    plt.ylabel("Metric")
    plt.title("Classification-only layer sweep")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_png, dpi=180)
    plt.savefig(figure_pdf)
    plt.close()

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    if out_reps is not None:
        print(f"Wrote {out_reps}")
    print(f"Wrote {figure_png}")
    print(f"Wrote {figure_pdf}")


if __name__ == "__main__":
    main()
