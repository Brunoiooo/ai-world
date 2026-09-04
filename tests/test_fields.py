import numpy as np
import pytest

from ai_world.persistence.database import connect, init_db
from ai_world.persistence.serialization import (
    deserialize_eco_state,
    serialize_eco_state,
)
from ai_world.persistence.worlds import WorldRepository
from ai_world.world.fields import FoodField, SpectrumField, TemperatureField
from ai_world.world.food import CARRION_IX, ENZYME_IX, FOOD_TYPES, N_FOOD_TYPES
from ai_world.world.generator import generate_grid
from ai_world.world.params import EcoParams
from ai_world.world.tiles import Tile

_GRASS_IX = next(i for i, f in enumerate(FOOD_TYPES) if f.terrain == int(Tile.GRASS))
from ai_world.world.weather import WeatherState
from ai_world.world.world import attach_ecosystem


@pytest.fixture()
def params():
    return EcoParams()


@pytest.fixture()
def grid():
    return generate_grid(96, 64, seed=7)


def test_food_grows_on_its_biome_and_stays_bounded(grid, params):
    rng = np.random.default_rng(0)
    field = FoodField.for_grid(grid, params, rng)
    assert field.values.shape == (N_FOOD_TYPES, *grid.cells.shape)
    rock = grid.cells == int(Tile.ROCK)

    for _ in range(2000):
        field.step()

    grass_plane = field.values[_GRASS_IX]
    on_grass = grass_plane[grid.cells == int(Tile.GRASS)].mean()
    assert field.values.min() >= 0.0
    assert field.values.max() <= params.food_capacity + 1e-5
    # the grass forage type thrives on grass, barely holds on rock (diffusion only)
    assert on_grass > 0.3
    assert grass_plane[rock].mean() < on_grass
    # derived channels never grow on their own
    assert field.values[ENZYME_IX].max() == 0.0
    assert field.values[CARRION_IX].max() == 0.0


def test_food_recolonises_a_grazed_out_patch(grid, params):
    field = FoodField.for_grid(grid, params, np.random.default_rng(1))
    ys, xs = np.where(grid.cells == int(Tile.GRASS))
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    field.values[:] = 0.0
    for _ in range(1500):
        field.step()
    assert field.values[_GRASS_IX, y, x] > 0.05  # seed term brought it back


def test_food_absorb_and_deposit(grid, params):
    field = FoodField.for_grid(grid, params, np.random.default_rng(1))
    field.values[_GRASS_IX, 10, 10] = 0.5
    assert field.absorb(_GRASS_IX, 10, 10, 0.2) == pytest.approx(0.2)
    assert field.values[_GRASS_IX, 10, 10] == pytest.approx(0.3)
    assert field.absorb(_GRASS_IX, 10, 10, 5.0) == pytest.approx(0.3)  # clamped
    field.deposit(ENZYME_IX, 10, 10, 0.4)
    assert field.values[ENZYME_IX, 10, 10] == pytest.approx(0.4)


def test_food_diffusion_spreads_a_spike(grid, params):
    field = FoodField.for_grid(grid, params, np.random.default_rng(2))
    field.values[:] = 0.0
    field.values[_GRASS_IX, 32, 48] = 1.0
    neighbour_before = field.values[_GRASS_IX, 32, 49]
    for _ in range(5):
        field.step(regen_multiplier=0.0)
    assert field.values[_GRASS_IX, 32, 49] > neighbour_before
    assert field.values[_GRASS_IX, 32, 48] < 1.0


def test_temperature_gradient(grid, params):
    temp = TemperatureField.for_grid(grid, params, seed=7)
    h = temp.values.shape[0]
    equator = temp.values[h // 2].mean()
    pole = temp.values[:3].mean()
    assert equator > pole


def test_temperature_refresh_stays_normalized(grid, params):
    temp = TemperatureField.for_grid(grid, params, seed=7)
    baseline = temp.values.copy()
    temp.refresh(season_offset=params.season_amplitude, anomaly=0.1)
    assert temp.values.min() >= 0.0 and temp.values.max() <= 1.0
    assert not np.allclose(temp.values, baseline)


def test_weather_fronts_drift_and_are_deterministic(params):
    weather = WeatherState()
    a = weather.front_field(64, 80, seed=3, tick=1000)
    b = weather.front_field(64, 80, seed=3, tick=1000)
    later = weather.front_field(64, 80, seed=3, tick=6000)
    assert a.shape == (64, 80)
    assert np.array_equal(a, b)               # deterministic
    assert not np.allclose(a, later)          # the field drifts over time
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_climate_event_fires_and_expires(params):
    weather = WeatherState()
    rng = np.random.default_rng(0)
    fired = False
    for tick in range(0, 400_000, params.weather_interval):
        weather.advance(tick, params, rng)
        if weather.event:
            fired = True
            break
    assert fired
    for _ in range(weather.event_ticks + 5):
        weather.advance(tick, params, np.random.default_rng(1))
        tick += params.weather_interval
    assert weather.event == ""


def test_spectrum_decays_toward_zero(grid, params):
    spectrum = SpectrumField.for_grid(grid, params)
    spectrum.splat(20, 20, np.ones(spectrum.channels, dtype=np.float32))
    peak = spectrum.values.max()
    for _ in range(200):
        spectrum.step()
    assert spectrum.values.max() < peak * 0.01


def test_eco_state_roundtrip(params):
    from ai_world.world.genome import Innovations

    weather = WeatherState()
    weather.advance(12_345, params)
    rng = np.random.default_rng(99)
    rng.random(10)  # advance it
    innov = Innovations(conn={(0, 9): 0, (1, 10): 1}, next_conn=2, next_node=15)

    blob = serialize_eco_state(params, weather, rng, innov,
                               {"threshold": 4.0, "target": 12, "next_id": 3})
    rp, rw, rr, ri, meta = deserialize_eco_state(blob)
    assert meta["next_id"] == 3

    assert rp == params
    assert rw == weather
    assert rr.random() == rng.random()
    assert ri.conn == innov.conn and ri.next_node == 15


def test_world_create_attaches_ecosystem_regardless_of_size():
    conn = connect(":memory:")
    init_db(conn)
    repo = WorldRepository(conn)

    small = repo.create("eco", 128, 128, seed=3)
    assert small.ecosystem_enabled
    assert small.food.values.shape == (N_FOOD_TYPES, 128, 128)
    assert small.spectrum.values.shape[0] == EcoParams().spectrum_channels

    # no map-size ceiling on the ecosystem any more -- large maps just run slower
    big = repo.create("big", 640, 640, seed=3)
    assert big.ecosystem_enabled
    assert big.food.values.shape == (N_FOOD_TYPES, 640, 640)
    conn.close()


def test_ecosystem_survives_save_load_roundtrip():
    conn = connect(":memory:")
    init_db(conn)
    repo = WorldRepository(conn)

    world = repo.create("eco", 96, 96, seed=11)
    world.food.step()
    world.spectrum.splat(5, 5, np.full(world.spectrum.channels, 0.7, dtype=np.float32))
    world.eco_rng.random(4)
    world.tick = 8000
    repo.save(world)

    reloaded = repo.load(world.id)
    assert reloaded.ecosystem_enabled
    assert np.array_equal(reloaded.food.values, world.food.values)
    assert np.array_equal(reloaded.spectrum.values, world.spectrum.values)
    assert reloaded.eco_rng.random() == world.eco_rng.random()
    conn.close()


def test_ecosystem_seed_is_deterministic():
    grid = generate_grid(64, 64, seed=5)
    from ai_world.world.world import World

    a = World(name="a", grid=grid, seed=5)
    b = World(name="b", grid=grid, seed=5)
    attach_ecosystem(a, EcoParams())
    attach_ecosystem(b, EcoParams())
    assert np.array_equal(a.food.values, b.food.values)
    assert np.array_equal(a.temperature.values, b.temperature.values)
