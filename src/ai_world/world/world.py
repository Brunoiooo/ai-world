"""World aggregate: metadata + grid + tick counter."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ai_world.world.grid import Grid


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class World:
    name: str
    grid: Grid
    seed: int
    tick: int = 0
    id: int | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    def touch(self) -> None:
        self.updated_at = _now()
