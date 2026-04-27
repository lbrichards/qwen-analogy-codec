#!/usr/bin/env python3
"""Project held-out vocabulary through the trained analogy codec."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from qwen_analogy_codec.extract import extract_word_representations
from qwen_analogy_codec.model_io import (
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)


HELDOUT_WORDS = ["prince", "princess", "boy", "girl", "father", "mother", "duke", "duchess"]
TRAIN_WORDS = ["king", "queen", "man", "woman"]


def quadrant(z: torch.Tensor) -> str:
    x, y = z.tolist()
    if x >= 0 and y >= 0:
        return "I (+,+)"
    if x < 0 and y >= 0:
        return "II (-,+)"
    if x < 0 and y < 0:
        return "III (-,-)"
    return "IV (+,-)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LOCAL_HF_MODEL_PATH", "/Users/larry/Development/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--reps", default="artifacts/reps.pt")
    parser.add_argument("--codec", default="artifacts/codec.pkl")
    parser.add_argument("--out-csv", default="results/heldout_vocabulary.csv")
    parser.add_argument("--out-json", default="results/heldout_vocabulary.json")
    parser.add_argument("--out-reps", default="results/heldout_vocabulary_reps.pt")
    parser.add_argument("--figure-png", default="figures/heldout_vocabulary.png")
    parser.add_argument("--figure-pdf", default="figures/heldout_vocabulary.pdf")
    parser.add_argument("--samples-per-word", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--no-offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_mode(not args.no_offline)
    device, dtype = select_device_and_dtype(args.device, args.dtype)

    train_data = torch.load(args.reps, map_location="cpu")
    codec = torch.load(args.codec, map_location="cpu")
    layer_offset = train_data["meta"]["layer_offset"]
    We = codec["We"]
    be = codec["be"]

    def encode(x: torch.Tensor) -> torch.Tensor:
        return x @ We + be

    train_z = {word: encode(train_data["reps"][word]) for word in TRAIN_WORDS}
    train_means = {word: values.mean(0) for word, values in train_z.items()}

    print(f"Loading model from {args.model_path} on {device} with dtype={dtype}...")
    model, tokenizer = load_local_model_and_tokenizer(args.model_path, device, dtype)

    heldout_reps = {}
    heldout_z = {}
    rows = []
    for word in HELDOUT_WORDS:
        print(f"Extracting held-out word {word!r}...")
        reps = extract_word_representations(
            model,
            tokenizer,
            word=word,
            device=device,
            layer_offset=layer_offset,
            n_samples=args.samples_per_word,
        )
        z_values = encode(reps)
        mean_z = z_values.mean(0)
        distances = {f"dist_to_{w}": torch.norm(mean_z - train_means[w]).item() for w in TRAIN_WORDS}
        nearest = min(TRAIN_WORDS, key=lambda w: distances[f"dist_to_{w}"])

        heldout_reps[word] = reps
        heldout_z[word] = z_values
        rows.append(
            {
                "word": word,
                "mean_z1": mean_z[0].item(),
                "mean_z2": mean_z[1].item(),
                "quadrant": quadrant(mean_z),
                "nearest_train_class": nearest,
                **distances,
            }
        )
        print(
            f"  mean=({mean_z[0].item():.3f}, {mean_z[1].item():.3f}), "
            f"quadrant={quadrant(mean_z)}, nearest={nearest}"
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_reps = Path(args.out_reps)
    figure_png = Path(args.figure_png)
    figure_pdf = Path(args.figure_pdf)
    for path in [out_csv, out_json, out_reps, figure_png, figure_pdf]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "reps": heldout_reps,
            "z": heldout_z,
            "meta": {
                "model": args.model_path,
                "layer_offset": layer_offset,
                "samples_per_word": args.samples_per_word,
                "words": HELDOUT_WORDS,
            },
        },
        out_reps,
    )

    colors = {
        "king": "#b2182b",
        "queen": "#ef8a62",
        "man": "#2166ac",
        "woman": "#67a9cf",
        "prince": "#d73027",
        "princess": "#fc8d59",
        "boy": "#4575b4",
        "girl": "#91bfdb",
        "father": "#313695",
        "mother": "#74add1",
        "duke": "#a50026",
        "duchess": "#fdae61",
    }

    plt.figure(figsize=(8, 7))
    for word in TRAIN_WORDS:
        values = train_z[word]
        mean = train_means[word]
        plt.scatter(values[:, 0], values[:, 1], s=20, alpha=0.18, color=colors[word])
        plt.scatter(
            mean[0],
            mean[1],
            s=120,
            marker="o",
            color=colors[word],
            edgecolor="black",
            linewidth=1.2,
            label=f"{word} (train)",
        )
        plt.text(mean[0].item(), mean[1].item(), f" {word}", fontsize=10, weight="bold")

    for word in HELDOUT_WORDS:
        values = heldout_z[word]
        mean = values.mean(0)
        plt.scatter(
            values[:, 0],
            values[:, 1],
            s=18,
            alpha=0.10,
            marker="x",
            color=colors[word],
        )
        plt.scatter(
            mean[0],
            mean[1],
            s=110,
            marker="o",
            facecolors="none",
            edgecolors=colors[word],
            linewidth=2.0,
            label=f"{word} (held-out)",
        )
        plt.text(mean[0].item(), mean[1].item(), f" {word}", fontsize=9)

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.25)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.25)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Held-out vocabulary projected through the trained codec")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best", fontsize=7, frameon=True)
    plt.tight_layout()
    plt.savefig(figure_png, dpi=180)
    plt.savefig(figure_pdf)
    plt.close()

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_reps}")
    print(f"Wrote {figure_png}")
    print(f"Wrote {figure_pdf}")


if __name__ == "__main__":
    main()
