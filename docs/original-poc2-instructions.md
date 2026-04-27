# POC-2: Automatic Codec Extractor

**Project Path:** `/Users/larry/Development/myelin_hrm`

## Overview

This experiment trains a 2D interpretable codec on top of a frozen Qwen 2.5-3B-Instruct model. The codec learns a latent space where:
- The analogy `king - man + woman ≈ queen` holds with cosine similarity ≥ 0.95
- A 2D classifier achieves ≥ 98% accuracy on held-out data
- The gender and royalty axes are nearly orthogonal (~90°)

## Prerequisites

- Python 3.12 with virtual environment at `.venv/`
- Qwen2.5-3B-Instruct model at `/Users/larry/Development/Qwen2.5-3B-Instruct`
- Dependencies: torch, transformers, numpy, sklearn, matplotlib, seaborn

## Running the Experiment

```bash
cd /Users/larry/Development/myelin_hrm

# Activate virtual environment
source .venv/bin/activate
# Or use the interpreter directly:
# /Users/larry/Development/myelin_hrm/.venv/bin/python

# Step 1: Extract representations from Qwen model
# Extracts 100 samples per word for {king, queen, man, woman}
# Output: 03_fit_codec/reps.pt
python 02_extract/extract_and_save.py

# Step 2: Train the 2D codec
# Trains linear encoder (2048-dim → 2-dim) with analogy-aware loss
# Output: 03_fit_codec/codec.pkl, 03_fit_codec/codec.json
python 03_fit_codec/train_codec.py

# Step 3: Validate all gates (30 checks)
# Runs comprehensive validation on saved artifacts
# Output: Console report with all metrics
python 03_fit_codec/validate_gates.py
```

## Expected Results

| Gate | Metric | Expected | Threshold |
|------|--------|----------|-----------|
| 1. Analogy | cosine(king-man+woman, queen) | ~0.9996 | ≥ 0.95 |
| 2. Accuracy | Held-out classification | 100% | ≥ 98% |
| 3. Orthogonality | Axis angle | ~89.7° | ≈ 90° |

## File Structure

```
myelin_hrm/
├── latent-bus/
│   └── hf_local_qwen_chatbot.py    # Model loading utilities
├── 02_extract/
│   └── extract_and_save.py         # Step 1: Extract representations
├── 03_fit_codec/
│   ├── train_codec.py              # Step 2: Train codec
│   ├── validate_gates.py           # Step 3: Validate
│   ├── reps.pt                     # Saved representations (3.3 MB)
│   └── codec.pkl                   # Trained codec (44 KB)
└── 04_eval/
    ├── create_dashboard.py         # Generate visualizations
    └── *.csv, *.png                # Evidence pack
```

## Validation Output (Last Run: 2026-04-24)

```
FINAL SUMMARY:
✅ Gate 1 (Analogy): 0.9996 ≥ 0.95
✅ Gate 2 (Accuracy): 100.0% ≥ 98%
✅ Gate 3 (Orthogonality): 89.7° ≈ 90°

All gates PASSED!
```
