"""Small persistence helpers for scan scripts."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True, allow_nan=False)


def replace_with_retry(
    source: str | Path,
    destination: str | Path,
    *,
    attempts: int = 20,
    initial_delay: float = 0.05,
    maximum_delay: float = 0.5,
) -> None:
    """Atomically replace a path, retrying transient Windows/OneDrive locks."""

    if attempts < 1:
        raise ValueError("attempts must be positive")
    delay = initial_delay
    for attempt in range(attempts):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(2.0 * delay, maximum_delay)
