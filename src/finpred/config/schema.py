"""Pydantic config models + YAML loader with `extends:` merge.

STUB (Phase A.1). Implement:
  - dataclasses/BaseModel for each config block (data, windows, model, train, eval, quiz,
    promotion) mirroring configs/default.yaml,
  - load_config(path) -> Config that resolves `extends:` by deep-merging the parent first,
  - device resolution ("auto" -> "cuda" if torch.cuda.is_available() else "cpu").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> Any:
    """Load a YAML config, resolving a possible `extends:` parent via deep merge.

    Returns a validated Config object. STUB.
    """
    raise NotImplementedError("config.schema.load_config — implement in Phase A.1")
