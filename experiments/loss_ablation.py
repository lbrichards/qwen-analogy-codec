#!/usr/bin/env python3
"""Run loss ablations for the two-dimensional analogy codec."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim


CONFIGS = [
    {
        "name": "full_loss",
        "description": "Full loss",
        "weights": {"cls": 1.0, "analogy": 3.0, "perp": 1.0, "orth": 0.1, "sep": 0.5},
    },
    {
        "name": "drop_analogy",
        "description": "Drop L_analogy",
        "weights": {"cls": 1.0, "analogy": 0.0, "perp": 1.0, "orth": 0.1, "sep": 0.5},
    },
    {
        "name": "drop_perp",
        "description": "Drop L_perp",
        "weights": {"cls": 1.0, "analogy": 3.0, "perp": 0.0, "orth": 0.1, "sep": 0.5},
    },
    {
        "name": "drop_analogy_and_perp",
        "description": "Drop L_analogy and L_perp",
        "weights": {"cls": 1.0, "analogy": 0.0, "perp": 0.0, "orth": 0.1, "sep": 0.5},
    },
    {
        "name": "classification_only",
        "description": "Classification only",
        "weights": {"cls": 1.0, "analogy": 0.0, "perp": 0.0, "orth": 0.0, "sep": 0.0},
    },
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_with_weights(
    X_by_class: dict[str, torch.Tensor],
    *,
    weights: dict[str, float],
    seed: int,
    steps: int,
    lr: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    words = list(X_by_class.keys())
    d_model = X_by_class[words[0]].shape[1]

    W_e = nn.Parameter(torch.randn(d_model, 2, device=device) / math.sqrt(d_model))
    b_e = nn.Parameter(torch.zeros(2, device=device))
    U = nn.Parameter(torch.zeros(2, len(words), device=device))
    b_u = nn.Parameter(torch.zeros(len(words), device=device))

    opt = optim.Adam([W_e, b_e, U, b_u], lr=lr)
    labels = {word: index for index, word in enumerate(words)}
    X = torch.cat([X_by_class[word] for word in words], 0).to(device)
    y = torch.cat(
        [torch.full((len(X_by_class[word]),), labels[word]) for word in words]
    ).long().to(device)
    ce = nn.CrossEntropyLoss()

    def encode(x: torch.Tensor) -> torch.Tensor:
        return x @ W_e + b_e

    def cls_logits(z: torch.Tensor) -> torch.Tensor:
        return z @ U + b_u

    for _ in range(steps):
        opt.zero_grad()
        z = encode(X)
        L_cls = ce(cls_logits(z), y)

        mu = {}
        for word in words:
            mu[word] = encode(X_by_class[word].to(device)).mean(0)

        analogy_pred = mu["king"] - mu["man"] + mu["woman"]
        L_analogy = (analogy_pred - mu["queen"]).pow(2).sum()

        gender = mu["woman"] - mu["man"]
        royalty = mu["king"] - mu["man"]
        L_perp = (
            (gender / (gender.norm() + 1e-8)) @ (royalty / (royalty.norm() + 1e-8))
        ).pow(2)

        I = torch.eye(2, device=device)
        L_orth = ((W_e.t() @ W_e) - I).pow(2).mean()

        L_sep = torch.tensor(0.0, device=device)
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i < j:
                    L_sep = L_sep + torch.exp(-(mu[w1] - mu[w2]).norm())

        loss = (
            weights["cls"] * L_cls
            + weights["analogy"] * L_analogy
            + weights["perp"] * L_perp
            + weights["orth"] * L_orth
            + weights["sep"] * L_sep
        )
        loss.backward()
        opt.step()

    return W_e.detach().cpu(), b_e.detach().cpu()


def evaluate(
    W_e: torch.Tensor,
    b_e: torch.Tensor,
    X_by_class: dict[str, torch.Tensor],
    *,
    split_seed: int,
) -> dict[str, float]:
    words = list(X_by_class.keys())

    def encode(x: torch.Tensor) -> torch.Tensor:
        return x @ W_e + b_e

    mu = {word: encode(X_by_class[word]).mean(0) for word in words}

    analogy_pred = mu["king"] - mu["man"] + mu["woman"]
    analogy_cos = torch.nn.functional.cosine_similarity(
        analogy_pred.unsqueeze(0), mu["queen"].unsqueeze(0)
    ).item()

    gender = mu["woman"] - mu["man"]
    royalty = mu["king"] - mu["man"]
    axis_dot = torch.dot(gender / gender.norm(), royalty / royalty.norm()).item()
    axis_angle = math.degrees(math.acos(max(-1.0, min(1.0, axis_dot))))

    torch.manual_seed(split_seed)
    correct = 0
    total = 0
    for word in words:
        n = len(X_by_class[word])
        idx = torch.randperm(n)
        test_data = X_by_class[word][idx[int(0.7 * n) :]]
        for z in encode(test_data):
            dists = {w2: torch.norm(z - mu[w2]) for w2 in words}
            pred = min(dists.items(), key=lambda item: item[1])[0]
            correct += int(pred == word)
            total += 1

    return {
        "analogy_cos": analogy_cos,
        "heldout_acc": correct / total,
        "axis_angle_deg": axis_angle,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", default="artifacts/reps.pt")
    parser.add_argument("--out-csv", default="results/loss_ablation.csv")
    parser.add_argument("--out-json", default="results/loss_ablation.json")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.reps, map_location="cpu")
    X_by_class = data["reps"]

    rows = []
    for config in CONFIGS:
        print(f"Training {config['description']}...")
        W_e, b_e = train_with_weights(
            X_by_class,
            weights=config["weights"],
            seed=args.seed,
            steps=args.steps,
            lr=args.lr,
            device=device,
        )
        metrics = evaluate(W_e, b_e, X_by_class, split_seed=args.split_seed)
        row = {
            "name": config["name"],
            "description": config["description"],
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "split_seed": args.split_seed,
            **config["weights"],
            **metrics,
        }
        rows.append(row)
        print(
            f"  analogy={metrics['analogy_cos']:.4f}, "
            f"accuracy={metrics['heldout_acc']:.1%}, "
            f"angle={metrics['axis_angle_deg']:.1f}"
        )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
