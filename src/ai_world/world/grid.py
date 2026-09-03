"""Tile grid — a thin wrapper over a ``numpy`` ``uint8`` array.

Coordinate layout: ``cells[y, x]``. Row ``y`` grows downward (screen-like).
"""
from __future__ import annotations

import numpy as np

from ai_world.world.tiles import Tile


class Grid:
    __slots__ = ("cells",)

    def __init__(self, cells: np.ndarray):
        if cells.ndim != 2:
            raise ValueError("grid must be 2D")
        self.cells = np.ascontiguousarray(cells, dtype=np.uint8)

    # --- constructors ------------------------------------------------
    @classmethod
    def empty(cls, width: int, height: int, fill: int = Tile.GRASS) -> "Grid":
        return cls(np.full((height, width), int(fill), dtype=np.uint8))

    # --- dimensions ------------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    # --- access ----------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> int:
        return int(self.cells[y, x])

    def set(self, x: int, y: int, tile: int) -> None:
        self.cells[y, x] = int(tile)

    def copy(self) -> "Grid":
        return Grid(self.cells.copy())
