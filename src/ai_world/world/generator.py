"""Deterministic terrain generation from value noise.

The same ``seed`` + the same dimensions => the same map.
"""
from __future__ import annotations

import numpy as np

from ai_world.world.grid import Grid
from ai_world.world.tiles import Tile

# Height thresholds (0..1) -> tile. Must increase monotonically.
_THRESHOLDS: list[tuple[float, int]] = [
    (0.30, Tile.DEEP_WATER),
    (0.40, Tile.WATER),
    (0.44, Tile.SAND),
    (0.62, Tile.GRASS),
    (0.72, Tile.FOREST),
    (0.80, Tile.DIRT),
    (0.90, Tile.ROCK),
    (1.01, Tile.SNOW),
]


def _octave(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """One octave: a random (cells+1)^2 lattice bilinearly interpolated to h x w."""
    grid = rng.random((cells + 1, cells + 1), dtype=np.float32)

    ys = np.linspace(0.0, cells, h, dtype=np.float32)
    xs = np.linspace(0.0, cells, w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, cells)
    x1 = np.minimum(x0 + 1, cells)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    # smoothstep for softer transitions
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)

    top = grid[np.ix_(y0, x0)] * (1.0 - fx) + grid[np.ix_(y0, x1)] * fx
    bot = grid[np.ix_(y1, x0)] * (1.0 - fx) + grid[np.ix_(y1, x1)] * fx
    return top * (1.0 - fy) + bot * fy


def heightmap(width: int, height: int, seed: int) -> np.ndarray:
    """float32 height map in the 0..1 range."""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    field = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    base = max(2, min(width, height) // 64)
    for octave in range(5):
        cells = base * (2 ** octave)
        field += amplitude * _octave(rng, height, width, cells)
        total += amplitude
        amplitude *= 0.5
    field /= total

    # Gentle falloff toward the edges -> the map tends to be an "island".
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    falloff = 1.0 - 0.35 * np.sqrt(yy * yy + xx * xx)
    field *= np.clip(falloff, 0.0, 1.0)

    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-6:
        return np.full((height, width), 0.5, dtype=np.float32)
    return (field - lo) / (hi - lo)


def generate_grid(width: int, height: int, seed: int) -> Grid:
    hm = heightmap(width, height, seed)
    cells = np.empty((height, width), dtype=np.uint8)
    prev = -1.0
    for threshold, tile in _THRESHOLDS:
        mask = (hm > prev) & (hm <= threshold)
        cells[mask] = int(tile)
        prev = threshold
    return Grid(cells)
