"""Save and load codec bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "qwen-analogy-codec-1.0"


def save_codec(
    path: str | Path,
    *,
    We: torch.Tensor | None,
    be: torch.Tensor | None,
    Wd: torch.Tensor | None,
    bd: torch.Tensor | None,
    U: torch.Tensor | None,
    bu: torch.Tensor | None,
    axes: dict[str, torch.Tensor] | None,
    meta: dict[str, Any] | None,
    write_sidecar: bool = True,
) -> None:
    """Save a codec bundle plus an optional human-readable JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "We": We.detach().cpu() if We is not None else None,
        "be": be.detach().cpu() if be is not None else None,
        "Wd": Wd.detach().cpu() if Wd is not None else None,
        "bd": bd.detach().cpu() if bd is not None else None,
        "U": U.detach().cpu() if U is not None else None,
        "bu": bu.detach().cpu() if bu is not None else None,
        "axes": {k: v.detach().cpu() for k, v in (axes or {}).items()},
        "meta": meta or {},
        "schema_version": SCHEMA_VERSION,
    }
    torch.save(bundle, path.with_suffix(".pkl"))

    if write_sidecar:
        sidecar = {
            **(meta or {}),
            "schema_version": SCHEMA_VERSION,
            "shapes": {
                k: tuple(v.shape) if v is not None else None
                for k, v in {
                    "We": bundle["We"],
                    "be": bundle["be"],
                    "Wd": bundle["Wd"],
                    "bd": bundle["bd"],
                    "U": bundle["U"],
                    "bu": bundle["bu"],
                    "gender_axis": bundle["axes"].get("gender"),
                    "royalty_axis": bundle["axes"].get("royalty"),
                }.items()
            },
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def load_codec(path: str | Path) -> dict[str, Any]:
    """Load a codec bundle from disk."""
    return torch.load(Path(path), map_location="cpu")
