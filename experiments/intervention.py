#!/usr/bin/env python3
"""Pseudoinverse hidden-state intervention through the trained 2D codec."""

from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from qwen_analogy_codec.extract import find_word_span
from qwen_analogy_codec.model_io import (
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)


TARGET_WORDS = ["king", "queen", "prince", "princess", "man", "woman"]
ALPHAS = [0.0, 0.5, 1.0]
PROMPT = "I saw the king today. The royal person was a"


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Could not locate decoder layers on model.")


@contextmanager
def hidden_shift_hook(model, *, layer_module_index: int, token_index: int, delta_x: torch.Tensor):
    """Add delta_x to one token at one decoder layer output."""
    layers = get_decoder_layers(model)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None

        if hidden.dim() == 3 and hidden.shape[1] > token_index:
            hidden[:, token_index, :] = hidden[:, token_index, :] + delta_x.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )

        if rest is not None:
            return (hidden, *rest)
        return hidden

    handle = layers[layer_module_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def continuation_logprob(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    """Compute log p(continuation | prompt), summing over continuation tokens."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    full = tokenizer(prompt + continuation, return_tensors="pt").to(device)
    full_ids = full["input_ids"][0]
    prompt_len = len(prompt_ids)

    with torch.no_grad():
        logits = model(**full).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    total = torch.tensor(0.0, device=log_probs.device)
    for pos in range(prompt_len, len(full_ids)):
        total = total + log_probs[pos - 1, full_ids[pos]]
    return float(total.detach().cpu())


def greedy_completion(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LOCAL_HF_MODEL_PATH", "/Users/larry/Development/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--reps", default="artifacts/reps.pt")
    parser.add_argument("--codec", default="artifacts/codec.pkl")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--target-token", default="king")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--out-csv", default="results/intervention_logprobs.csv")
    parser.add_argument("--out-json", default="results/intervention.json")
    parser.add_argument("--figure-png", default="figures/intervention_logprob_shifts.png")
    parser.add_argument("--figure-pdf", default="figures/intervention_logprob_shifts.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_offline_mode(not args.no_offline)
    device, dtype = select_device_and_dtype(args.device, args.dtype)

    reps = torch.load(args.reps, map_location="cpu")
    codec = torch.load(args.codec, map_location="cpu")
    We = codec["We"]
    be = codec["be"]
    layer_offset = reps["meta"]["layer_offset"]

    def encode(x: torch.Tensor) -> torch.Tensor:
        return x @ We + be

    train_means = {word: encode(reps["reps"][word]).mean(0) for word in ["king", "queen", "man", "woman"]}
    g_2d = train_means["woman"] - train_means["man"]
    W_pinv = torch.linalg.pinv(We)

    print(f"Loading model from {args.model_path} on {device} with dtype={dtype}...")
    model, tokenizer = load_local_model_and_tokenizer(args.model_path, device, dtype)
    layers = get_decoder_layers(model)
    hidden_state_index = len(layers) - layer_offset
    layer_module_index = hidden_state_index - 1

    start_tok, end_tok = find_word_span(tokenizer, args.prompt, args.target_token)
    if start_tok is None or end_tok is None:
        raise ValueError(f"Could not find target token {args.target_token!r} in prompt.")
    token_index = end_tok - 1

    rows = []
    completions = {}
    baseline_by_word = {}

    for alpha in args.alphas:
        delta_z = alpha * g_2d
        delta_x = delta_z @ W_pinv
        delta_norm = torch.norm(delta_x).item()
        z_target = train_means["king"] + delta_z

        print(f"Running alpha={alpha} with ||delta_x||={delta_norm:.4f}...")
        with hidden_shift_hook(
            model,
            layer_module_index=layer_module_index,
            token_index=token_index,
            delta_x=delta_x,
        ):
            completions[str(alpha)] = greedy_completion(
                model,
                tokenizer,
                args.prompt,
                device,
                args.max_new_tokens,
            )
            for word in TARGET_WORDS:
                continuation = " " + word
                token_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
                logprob = continuation_logprob(model, tokenizer, args.prompt, continuation, device)
                if alpha == 0.0:
                    baseline_by_word[word] = logprob
                rows.append(
                    {
                        "alpha": alpha,
                        "word": word,
                        "continuation": continuation,
                        "token_ids": " ".join(str(t) for t in token_ids),
                        "logprob": logprob,
                        "logprob_shift_from_alpha0": logprob - baseline_by_word.get(word, logprob),
                        "delta_x_norm": delta_norm,
                        "z_target_1": z_target[0].item(),
                        "z_target_2": z_target[1].item(),
                        "layer_hidden_state_index": hidden_state_index,
                        "layer_module_index": layer_module_index,
                        "token_index": token_index,
                    }
                )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    figure_png = Path(args.figure_png)
    figure_pdf = Path(args.figure_pdf)
    for path in [out_csv, out_json, figure_png, figure_pdf]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "prompt": args.prompt,
        "target_token": args.target_token,
        "target_words": TARGET_WORDS,
        "alphas": args.alphas,
        "completions": completions,
        "rows": rows,
        "meta": {
            "model": args.model_path,
            "layer_offset": layer_offset,
            "hidden_state_index": hidden_state_index,
            "layer_module_index": layer_module_index,
            "token_index": token_index,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(8, 5))
    for word in TARGET_WORDS:
        word_rows = [row for row in rows if row["word"] == word]
        xs = [row["alpha"] for row in word_rows]
        ys = [row["logprob_shift_from_alpha0"] for row in word_rows]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=word)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel(r"Intervention strength $\alpha$")
    plt.ylabel("Log-probability shift from alpha=0")
    plt.title("Hidden-state intervention via codec pseudoinverse")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figure_png, dpi=180)
    plt.savefig(figure_pdf)
    plt.close()

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {figure_png}")
    print(f"Wrote {figure_pdf}")
    print("Completions:")
    for alpha, text in completions.items():
        print(f"  alpha={alpha}: {text!r}")


if __name__ == "__main__":
    main()
