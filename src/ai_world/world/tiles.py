"""Tile types and their visual representation.

At this stage a tile is terrain only (no entities). The enum is deliberately
an ``IntEnum`` so its values map 1:1 onto cells of a ``uint8`` array.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class Tile(IntEnum):
    DEEP_WATER = 0
    WATER = 1
    SAND = 2
    GRASS = 3
    FOREST = 4
    DIRT = 5
    ROCK = 6
    SNOW = 7


TILE_NAMES: dict[int, str] = {
    Tile.DEEP_WATER: "deep water",
    Tile.WATER: "water",
    Tile.SAND: "sand",
    Tile.GRASS: "grass",
    Tile.FOREST: "forest",
    Tile.DIRT: "dirt",
    Tile.ROCK: "rock",
    Tile.SNOW: "snow",
}

TILE_COLORS: dict[int, tuple[int, int, int]] = {
    Tile.DEEP_WATER: (28, 52, 94),
    Tile.WATER: (46, 92, 148),
    Tile.SAND: (204, 188, 130),
    Tile.GRASS: (96, 148, 74),
    Tile.FOREST: (58, 104, 58),
    Tile.DIRT: (122, 96, 66),
    Tile.ROCK: (110, 110, 116),
    Tile.SNOW: (232, 236, 240),
}

# 256x3 palette for fast vectorized tile-index -> RGB conversion.
PALETTE: np.ndarray = np.zeros((256, 3), dtype=np.uint8)
for _tile, _color in TILE_COLORS.items():
    PALETTE[int(_tile)] = _color
