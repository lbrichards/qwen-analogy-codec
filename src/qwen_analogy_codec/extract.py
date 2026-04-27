"""Extract target-word hidden states from a local Qwen model."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from qwen_analogy_codec.model_io import (
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)


TARGET_WORDS = ["king", "queen", "man", "woman"]
TEMPLATES = [
    "I saw the {w} today.",
    "The {w} was there.",
    "Look at the {w}.",
    "This is a {w}.",
    "The {w} arrived.",
    "We talked about the {w}.",
    "Everyone knows the {w}.",
    "I found the {w}.",
    "Where is the {w}?",
    "The {w} walked by.",
    "Here comes the {w}.",
    "That's the {w}.",
    "I met the {w}.",
    "The {w} is here.",
    "We need the {w}.",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_word_span(tokenizer, sentence: str, word: str) -> tuple[int | None, int | None]:
    """Find the token span overlapping the target word."""
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]

    start_char = sentence.find(word)
    if start_char < 0:
        for variant in [word.lower(), word.capitalize()]:
            start_char = sentence.find(variant)
            if start_char >= 0:
                word = variant
                break
    if start_char < 0:
        return None, None

    end_char = start_char + len(word)
    start_tok = None
    end_tok = None
    for token_index, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if start_tok is None and not (b <= start_char or a >= end_char):
            start_tok = token_index
        if start_tok is not None and a < end_char:
            end_tok = token_index + 1
    return start_tok, end_tok


@torch.no_grad()
def extract_word_representations(
    model,
    tokenizer,
    *,
    word: str,
    device: str,
    layer_offset: int,
    n_samples: int,
) -> torch.Tensor:
    """Extract hidden vectors for a word across simple sentence contexts."""
    representations = []

    for _ in range(n_samples):
        template = random.choice(TEMPLATES)
        surface = word.lower() if random.random() < 0.5 else word
        sentence = template.format(w=surface)

        start_tok, end_tok = find_word_span(tokenizer, sentence, word)
        if start_tok is None or end_tok is None:
            continue

        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        layer_idx = len(outputs.hidden_states) - 1 - layer_offset
        hidden = outputs.hidden_states[layer_idx].squeeze(0)
        representations.append(hidden[end_tok - 1].cpu().float())

    if not representations:
        raise ValueError(f"No representations extracted for {word!r}")
    return torch.stack(representations)


def extract_representations(
    *,
    model_path: str,
    output_path: str | Path,
    target_words: list[str] | None = None,
    layer_offset: int = 10,
    n_samples_per_word: int = 100,
    random_seed: int = 1337,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
    offline: bool = True,
) -> dict:
    """Load a local model, extract target-word hidden states, and save them."""
    words = target_words or TARGET_WORDS
    seed_everything(random_seed)
    set_offline_mode(offline)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)

    all_reps = {}
    for word in words:
        print(f"Extracting {word!r}...")
        all_reps[word] = extract_word_representations(
            model,
            tokenizer,
            word=word,
            device=device,
            layer_offset=layer_offset,
            n_samples=n_samples_per_word,
        )
        print(f"  shape={tuple(all_reps[word].shape)}")

    means = {word: reps.mean(dim=0) for word, reps in all_reps.items()}
    data = {
        "reps": all_reps,
        "means": means,
        "meta": {
            "model": model_path,
            "layer_offset": layer_offset,
            "n_samples_per_word": n_samples_per_word,
            "words": words,
            "random_seed": random_seed,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    return data
