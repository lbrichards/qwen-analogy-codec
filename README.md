# qwen-analogy-codec

Train and validate a tiny, interpretable 2D codec over frozen Qwen hidden states.

This experiment extracts hidden activations for four words:

```text
king, queen, man, woman
```

It then learns a linear projection from Qwen's high-dimensional hidden space into a 2D
latent space where:

- `king - man + woman` lands near `queen`
- the four word classes are separable with nearest-mean classification
- the gender and royalty directions are close to orthogonal

The saved Sprint-1 artifacts pass the original gates:

| Gate | Metric | Result | Target |
| --- | --- | ---: | ---: |
| Analogy | cosine(`king - man + woman`, `queen`) | ~0.9996 | >= 0.95 |
| Accuracy | held-out nearest-mean classification | 100% | >= 98% |
| Axes | gender/royalty axis angle | ~89.7 deg | ~90 deg |

## Why this matters

This is a small, controlled demonstration that semantic structure can be extracted
from model internals into a compact, human-readable coordinate system. It does not
prove that Qwen has universal global axes for gender or royalty, but it does show
that this analogy structure is present in the hidden states and can be organized by
a simple supervised codec.

## Repository layout

```text
src/qwen_analogy_codec/   reusable package code
scripts/                  command-line entrypoints
artifacts/                saved reps and trained codec from the original run
evidence/                 CSV and PNG validation evidence
docs/                     original POC notes and final report
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The extraction step expects a local Hugging Face Qwen model. By default it looks at:

```text
/Users/larry/Development/Qwen2.5-3B-Instruct
```

You can also set:

```bash
export LOCAL_HF_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct
```

## Reproduce the pipeline

Use the saved artifacts immediately:

```bash
python scripts/validate_gates.py
```

Regenerate the full pipeline:

```bash
python scripts/extract_reps.py --model-path /path/to/Qwen2.5-3B-Instruct
python scripts/train_codec.py
python scripts/validate_gates.py
python scripts/generate_evidence_pack.py
```

## Classification procedure

The confusion matrix is built from real model activations, not synthetic 2D
perturbations. The extraction script samples simple sentence templates for each
target word, extracts the target word hidden state, projects it into 2D, and then
classifies held-out points by nearest class mean in that 2D space.

The original run used 100 examples per word. The validation script uses a 70/30
split per class for the nearest-mean classification report.

## Notes

This is a proof of concept. The classifier check is a separability gate over a
codec trained on the full extracted dataset, not a strict isolated generalization
benchmark.
