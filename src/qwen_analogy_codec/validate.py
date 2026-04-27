"""Validation gates for the Qwen analogy codec."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


WORDS = ["king", "queen", "man", "woman"]


def encode(x: torch.Tensor, We: torch.Tensor, be: torch.Tensor) -> torch.Tensor:
    return x @ We + be


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_gates(
    *,
    reps_path: str | Path,
    codec_path: str | Path,
    bootstrap_samples: int = 1000,
) -> dict:
    """Run the main analogy, classification, and axis validation gates."""
    data = torch.load(reps_path, map_location="cpu")
    codec = torch.load(codec_path, map_location="cpu")

    X_by_class = data["reps"]
    means_raw = data["means"]
    We = codec["We"]
    be = codec["be"]

    means_z = {word: encode(X_by_class[word], We, be).mean(0) for word in WORDS}

    analogy_raw = means_raw["king"] - means_raw["man"] + means_raw["woman"]
    cos_raw = torch.nn.functional.cosine_similarity(
        analogy_raw.unsqueeze(0), means_raw["queen"].unsqueeze(0)
    ).item()

    analogy_z = means_z["king"] - means_z["man"] + means_z["woman"]
    cos_z = torch.nn.functional.cosine_similarity(
        analogy_z.unsqueeze(0), means_z["queen"].unsqueeze(0)
    ).item()
    l2_z = torch.norm(analogy_z - means_z["queen"]).item()

    cos_samples = []
    for _ in range(bootstrap_samples):
        resampled = {}
        for word in WORDS:
            n = len(X_by_class[word])
            idx = torch.randint(0, n, (n,))
            resampled[word] = X_by_class[word][idx]
        means_boot = {word: encode(resampled[word], We, be).mean(0) for word in WORDS}
        analogy_boot = means_boot["king"] - means_boot["man"] + means_boot["woman"]
        cos_boot = torch.nn.functional.cosine_similarity(
            analogy_boot.unsqueeze(0), means_boot["queen"].unsqueeze(0)
        ).item()
        cos_samples.append(cos_boot)

    y_true = []
    y_pred = []
    torch.manual_seed(1337)
    for i, word in enumerate(WORDS):
        n = len(X_by_class[word])
        idx = torch.randperm(n)
        test_data = X_by_class[word][idx[int(0.7 * n) :]]
        for z in encode(test_data, We, be):
            dists = {w2: torch.norm(z - means_z[w2]) for w2 in WORDS}
            pred = min(dists.items(), key=lambda x: x[1])[0]
            y_true.append(i)
            y_pred.append(WORDS.index(pred))

    cm = confusion_matrix(y_true, y_pred)
    accuracy = (np.array(y_true) == np.array(y_pred)).mean()

    gender = means_z["woman"] - means_z["man"]
    royalty = means_z["king"] - means_z["man"]
    dot = torch.dot(gender / gender.norm(), royalty / royalty.norm()).item()
    angle = math.degrees(math.acos(np.clip(dot, -1, 1)))

    results = {
        "analogy_cos_raw": cos_raw,
        "analogy_cos_z": cos_z,
        "analogy_l2_z": l2_z,
        "analogy_ci95_low": float(np.percentile(cos_samples, 2.5)),
        "analogy_ci95_high": float(np.percentile(cos_samples, 97.5)),
        "heldout_accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "axis_angle": angle,
        "axis_dot": dot,
        "passed": bool(cos_z >= 0.95 and accuracy >= 0.98 and abs(angle - 90) < 15),
    }

    print("Validation summary")
    print(f"  Analogy raw cosine: {cos_raw:.4f}")
    print(f"  Analogy z cosine:   {cos_z:.4f}")
    print(f"  Analogy z L2:       {l2_z:.4f}")
    print(
        f"  Bootstrap 95% CI:   "
        f"[{results['analogy_ci95_low']:.4f}, {results['analogy_ci95_high']:.4f}]"
    )
    print(f"  Held-out accuracy:  {accuracy:.1%}")
    print(f"  Axis angle:         {angle:.1f} deg")
    print("  Confusion matrix:")
    print("       pred: king queen man woman")
    for word, row in zip(WORDS, cm):
        print(f"  {word:6s}     {row.tolist()}")
    print(f"  Passed:            {results['passed']}")
    return results
