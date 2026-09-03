import numpy as np
import pytest

from ai_world.persistence.database import connect, init_db
from ai_world.persistence.worlds import WorldRepository


@pytest.fixture()
def repo():
    conn = connect(":memory:")
    init_db(conn)
    yield WorldRepository(conn)
    conn.close()


def test_create_load_roundtrip(repo):
    created = repo.create("Test", 80, 40, seed=42)
    assert created.id is not None

    loaded = repo.load(created.id)
    assert loaded.name == "Test"
    assert loaded.width == 80 and loaded.height == 40
    assert loaded.seed == 42
    assert loaded.tick == 0
    assert np.array_equal(loaded.grid.cells, created.grid.cells)


def test_save_persists_tick_and_grid(repo):
    world = repo.create("World", 32, 32, seed=1)
    world.tick = 5000
    world.grid.set(1, 1, 7)
    repo.save(world)

    reloaded = repo.load(world.id)
    assert reloaded.tick == 5000
    assert reloaded.grid.get(1, 1) == 7


def test_list_orders_by_updated_desc(repo):
    a = repo.create("A", 16, 16, seed=1)
    repo.create("B", 16, 16, seed=2)
    a.tick += 1
    repo.save(a)  # a updated most recently

    names = [w.name for w in repo.list_worlds()]
    assert names[0] == "A"
    assert set(names) == {"A", "B"}
    assert all(not hasattr(s, "grid") for s in repo.list_worlds())


def test_delete_removes_world(repo):
    world = repo.create("To delete", 16, 16, seed=1)
    repo.delete(world.id)
    assert repo.list_worlds() == []
    with pytest.raises(KeyError):
        repo.load(world.id)


def test_dimensions_are_clamped(repo):
    world = repo.create("Too big", 999999, 4, seed=1)
    assert world.width <= 4096
    assert world.height >= 16
