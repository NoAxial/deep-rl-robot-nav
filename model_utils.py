"""Checkpoint and normalization-file resolution helpers."""

from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL = Path("robot_nav_model.zip")
BEST_MODEL = Path("best_model") / "best_model.zip"


def _model_candidates(value: str | Path) -> list[Path]:
    path = Path(value)
    candidates = [path]
    if path.suffix.lower() != ".zip":
        candidates.append(Path(f"{path}.zip"))
    return candidates


def resolve_model_path(requested: str | Path = DEFAULT_MODEL) -> Path:
    """Resolve an explicit checkpoint, falling back to best only for the default."""
    candidates = _model_candidates(requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    requested_path = Path(requested)
    default_names = {DEFAULT_MODEL.name, DEFAULT_MODEL.stem}
    if requested_path.parent == Path(".") and requested_path.name in default_names:
        if BEST_MODEL.is_file():
            return BEST_MODEL

    raise FileNotFoundError(f"Model checkpoint not found: {requested}")


def normalization_path_for(model_path: str | Path) -> Path:
    """Return the normalization statistics expected beside a checkpoint."""
    return Path(model_path).parent / "vec_normalize.pkl"
