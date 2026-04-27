"""Repository path helpers."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "evidence"
