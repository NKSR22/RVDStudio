from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass(slots=True)
class AppPaths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def raw_data_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def resolve_in_project(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        return (self.root / path).resolve()


def sanitize_session_name(session_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", session_name.strip())
    return cleaned.strip("._") or "session"


DEFAULT_CLASSES = ["person", "obstacle"]
DEFAULT_DECISIONS = ["stop_wait", "bypass_left", "bypass_right", "blocked", "clear"]
