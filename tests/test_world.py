import numpy as np

from ai_world.persistence.serialization import deserialize_grid, serialize_grid
from ai_world.world.generator import generate_grid
from ai_world.world.grid import Grid
from ai_world.world.tiles import Tile


def test_grid_dimensions_and_access():
    grid = Grid.empty(20, 12, fill=Tile.SAND)
    assert grid.width == 20 and grid.height == 12
    assert grid.get(0, 0) == Tile.SAND
    grid.set(5, 3, Tile.ROCK)
    assert grid.get(5, 3) == Tile.ROCK
    assert grid.in_bounds(19, 11) and not grid.in_bounds(20, 11)


def test_generator_is_deterministic():
    a = generate_grid(64, 48, seed=1234)
    b = generate_grid(64, 48, seed=1234)
    c = generate_grid(64, 48, seed=9999)
    assert np.array_equal(a.cells, b.cells)
    assert not np.array_equal(a.cells, c.cells)
    assert a.width == 64 and a.height == 48
    assert a.cells.min() >= 0 and a.cells.max() <= int(Tile.SNOW)


def test_grid_roundtrip_through_blob():
    original = generate_grid(50, 30, seed=7)
    blob = serialize_grid(original)
    restored = deserialize_grid(blob, 50, 30)
    assert np.array_equal(original.cells, restored.cells)


def test_non_square_dimensions_are_respected():
    grid = generate_grid(120, 30, seed=3)
    assert grid.cells.shape == (30, 120)
