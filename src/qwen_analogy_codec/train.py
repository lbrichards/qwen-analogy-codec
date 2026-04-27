"""Train and evaluate a 2D analogy codec."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn, optim

from qwen_analogy_codec.codec_io import save_codec


def train_codec(X_by_class, steps=3000, lr=1e-2, device="cpu", verbose=True):
    """Train a linear encoder from hidden states into a 2D analogy space."""
    words = list(X_by_class.keys())
    d_model = X_by_class[words[0]].shape[1]

    W_e = nn.Parameter(torch.randn(d_model, 2, device=device) / math.sqrt(d_model))
    b_e = nn.Parameter(torch.zeros(2, device=device))
    U = nn.Parameter(torch.zeros(2, len(words), device=device))
    b_u = nn.Parameter(torch.zeros(len(words), device=device))

    opt = optim.Adam([W_e, b_e, U, b_u], lr=lr)
    labels = {word: index for index, word in enumerate(words)}

    def encode(x):
        return x @ W_e + b_e

    def cls_logits(z):
        return z @ U + b_u

    X = torch.cat([X_by_class[word] for word in words], 0).to(device)
    y = torch.cat(
        [torch.full((len(X_by_class[word]),), labels[word]) for word in words]
    ).long().to(device)

    ce = nn.CrossEntropyLoss()
    best_analogy_cos = 0.0

    for step in range(steps):
        opt.zero_grad()
        z = encode(X)
        L_cls = ce(cls_logits(z), y)

        mu = {}
        for word in words:
            mu[word] = encode(X_by_class[word].to(device)).mean(0)

        analogy_pred = mu["king"] - mu["man"] + mu["woman"]
        analogy_target = mu["queen"]
        L_analogy = (analogy_pred - analogy_target).pow(2).sum()

        gender = mu["woman"] - mu["man"]
        royalty = mu["king"] - mu["man"]
        g_norm = gender / (gender.norm() + 1e-8)
        r_norm = royalty / (royalty.norm() + 1e-8)
        L_perp = (g_norm @ r_norm).pow(2)

        I = torch.eye(2, device=device)
        L_orth = ((W_e.t() @ W_e) - I).pow(2).mean()

        L_sep = 0
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i < j:
                    L_sep += torch.exp(-(mu[w1] - mu[w2]).norm())

        loss = L_cls + 3.0 * L_analogy + L_perp + 0.1 * L_orth + 0.5 * L_sep
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            with torch.no_grad():
                analogy_norm = analogy_pred / (analogy_pred.norm() + 1e-8)
                queen_norm = analogy_target / (analogy_target.norm() + 1e-8)
                cos = float(torch.dot(analogy_norm, queen_norm))
                preds = cls_logits(encode(X)).argmax(dim=1)
                acc = (preds == y).float().mean().item()
                best_analogy_cos = max(best_analogy_cos, cos)
                if verbose:
                    print(
                        f"Step {step + 1:4d}: loss={loss.item():.3f}, "
                        f"analogy_cos={cos:.3f}, acc={acc:.3f}"
                    )

    with torch.no_grad():
        preds = cls_logits(encode(X)).argmax(dim=1)
        final_acc = (preds == y).float().mean().item()
        mu_final = {word: encode(X_by_class[word].to(device)).mean(0) for word in words}
        analogy_final = mu_final["king"] - mu_final["man"] + mu_final["woman"]
        analogy_norm = analogy_final / (analogy_final.norm() + 1e-8)
        queen_norm = mu_final["queen"] / (mu_final["queen"].norm() + 1e-8)
        final_cos = float(torch.dot(analogy_norm, queen_norm))

    print("Training complete.")
    print(f"Final analogy cosine: {final_cos:.3f}")
    print(f"Final accuracy: {final_acc:.3f}")
    print(f"Best analogy cosine: {best_analogy_cos:.3f}")

    return W_e.detach().cpu(), b_e.detach().cpu(), U.detach().cpu(), b_u.detach().cpu()


def evaluate_codec(W_e, b_e, X_by_class):
    """Evaluate analogy quality, axis angle, and held-out nearest-mean accuracy."""
    words = list(X_by_class.keys())

    def encode(x):
        return x @ W_e + b_e

    mu = {word: encode(X_by_class[word]).mean(0) for word in words}
    analogy_pred = mu["king"] - mu["man"] + mu["woman"]
    analogy_cos = torch.nn.functional.cosine_similarity(
        analogy_pred.unsqueeze(0), mu["queen"].unsqueeze(0)
    ).item()

    gender = mu["woman"] - mu["man"]
    royalty = mu["king"] - mu["man"]
    axis_dot = (gender / gender.norm() @ royalty / royalty.norm()).item()
    axis_angle = math.degrees(math.acos(max(-1, min(1, axis_dot))))

    all_correct = 0
    all_total = 0
    for word in words:
        data = X_by_class[word]
        n = len(data)
        if n < 10:
            continue
        idx = torch.randperm(n)
        test_data = data[idx[int(0.7 * n) :]]
        for z in encode(test_data):
            dists = {w2: (z - mu[w2]).norm() for w2 in words}
            pred = min(dists.items(), key=lambda x: x[1])[0]
            all_correct += int(pred == word)
            all_total += 1

    held_out_acc = all_correct / all_total if all_total else 0.0
    success = analogy_cos >= 0.95 and held_out_acc >= 0.98 and abs(axis_angle - 90) < 15
    return {
        "analogy_cos": analogy_cos,
        "axis_angle": axis_angle,
        "held_out_acc": held_out_acc,
        "success": success,
    }


def train_from_reps(
    *,
    reps_path: str | Path,
    codec_base_path: str | Path,
    steps: int = 3000,
    lr: float = 0.01,
    device: str | None = None,
) -> dict:
    """Load extracted reps, train the codec, evaluate it, and save the bundle."""
    data = torch.load(reps_path, map_location="cpu")
    X_by_class = data["reps"]
    meta = data["meta"]
    train_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    W_e, b_e, U, b_u = train_codec(X_by_class, steps=steps, lr=lr, device=train_device)
    results = evaluate_codec(W_e, b_e, X_by_class)

    print("Codec evaluation:")
    print(f"  analogy_cos: {results['analogy_cos']:.4f}")
    print(f"  held_out_acc: {results['held_out_acc']:.1%}")
    print(f"  axis_angle: {results['axis_angle']:.1f} deg")

    d_model = X_by_class["man"].shape[1]
    W_d = W_e.t()
    gender_x = torch.tensor([1.0, 0.0]) @ W_d
    royalty_x = torch.tensor([0.0, 1.0]) @ W_d

    save_codec(
        codec_base_path,
        We=W_e,
        be=b_e,
        Wd=W_d,
        bd=torch.zeros(d_model),
        U=U,
        bu=b_u,
        axes={"gender": gender_x, "royalty": royalty_x},
        meta={
            "model": meta.get("model"),
            "layer_offset": meta.get("layer_offset"),
            "analogy_cos": results["analogy_cos"],
            "held_out_acc": results["held_out_acc"],
            "axis_angle": results["axis_angle"],
            "notes": "Qwen analogy codec trained on king/queen/man/woman activations",
        },
    )
    return results
