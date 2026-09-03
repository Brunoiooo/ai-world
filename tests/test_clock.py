from ai_world.simulation.clock import SPEED_LADDER, SimulationClock


def _counter():
    box = {"n": 0}

    def step():
        box["n"] += 1

    return box, step


def test_paused_clock_does_not_step():
    clock = SimulationClock()
    clock.set_paused(True)
    box, step = _counter()
    assert clock.advance(1.0, step) == 0
    assert box["n"] == 0


def test_fixed_speed_matches_tps_over_time():
    clock = SimulationClock()  # defaults to "1x" == 5 TPS
    box, step = _counter()
    total = 0
    for _ in range(10):
        total += clock.advance(0.1, step)  # 1 s total
    assert box["n"] == total
    assert 4 <= total <= 6


def test_faster_slower_clamp_at_ends():
    clock = SimulationClock()
    for _ in range(50):
        clock.slower()
    assert clock.speed_index == 0
    for _ in range(50):
        clock.faster()
    assert clock.speed_index == len(SPEED_LADDER) - 1
    assert clock.is_unlimited


def test_unlimited_speed_runs_many_ticks_in_one_frame():
    clock = SimulationClock()
    while not clock.is_unlimited:
        clock.faster()
    box, step = _counter()
    ticks = clock.advance(1 / 60, step)
    assert ticks >= 64
    assert box["n"] == ticks


def test_no_death_spiral_after_long_frame():
    clock = SimulationClock()
    box, step = _counter()
    clock.advance(100.0, step)  # huge dt
    # the accumulator must not "run away" — the next normal frame stays short
    box["n"] = 0
    clock.advance(1 / 60, step)
    assert box["n"] < 10
