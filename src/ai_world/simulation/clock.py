"""Simulation time control.

Model:
* the simulation is counted in whole *ticks*;
* speed is expressed in *ticks per second* (TPS) along a speed ladder;
* the top rung of the ladder ("MAX") is uncapped — the engine runs as many
  ticks as it can fit into a per-frame time budget;
* pausing stops the passage of time entirely.

Uses a fixed-step accumulator with a guard against the "spiral of death"
(when frames are long, we do not try to catch up indefinitely).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SpeedSetting:
    label: str
    tps: float | None  # None == uncapped ("MAX")


# 1x ~ "human" pace (readable with the naked eye). Then up to the machine's max.
_BASE_TPS = 5.0
_MULTIPLIERS: list[tuple[str, float | None]] = [
    ("0.2x", 0.2),
    ("0.5x", 0.5),
    ("1x", 1.0),
    ("2x", 2.0),
    ("3x", 3.0),
    ("5x", 5.0),
    ("10x", 10.0),
    ("25x", 25.0),
    ("50x", 50.0),
    ("100x", 100.0),
    ("250x", 250.0),
    ("MAX", None),
]
SPEED_LADDER: list[SpeedSetting] = [
    SpeedSetting(label, None if mult is None else _BASE_TPS * mult)
    for label, mult in _MULTIPLIERS
]
_DEFAULT_INDEX = 2  # "1x"

# Hard cap of ticks per frame (for capped modes) — protects the render loop.
_MAX_TICKS_PER_FRAME = 20_000
# Time budget for ticks in MAX mode (the rest of the frame goes to render/input).
_UNLIMITED_BUDGET_S = 0.012


class SimulationClock:
    def __init__(self) -> None:
        self.paused: bool = False
        self._index: int = _DEFAULT_INDEX
        self._accumulator: float = 0.0
        self._measured_tps: float = 0.0  # exponentially smoothed

    # --- controls --------------------------------------------------
    @property
    def speed(self) -> SpeedSetting:
        return SPEED_LADDER[self._index]

    @property
    def speed_index(self) -> int:
        return self._index

    @property
    def is_unlimited(self) -> bool:
        return self.speed.tps is None

    @property
    def measured_tps(self) -> float:
        return self._measured_tps

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._accumulator = 0.0

    def set_paused(self, value: bool) -> None:
        self.paused = value
        if value:
            self._accumulator = 0.0

    def faster(self) -> None:
        self._index = min(self._index + 1, len(SPEED_LADDER) - 1)
        self._accumulator = 0.0

    def slower(self) -> None:
        self._index = max(self._index - 1, 0)
        self._accumulator = 0.0

    def reset_speed(self) -> None:
        self._index = _DEFAULT_INDEX
        self._accumulator = 0.0

    # --- loop ------------------------------------------------------
    def advance(self, dt: float, step: Callable[[], None]) -> int:
        """Runs ``step()`` the appropriate number of times for ``dt`` real seconds.

        Returns the number of ticks executed (for statistics).
        """
        if self.paused or dt <= 0.0:
            self._decay_tps(dt)
            return 0

        setting = self.speed
        if setting.tps is None:
            ticks = self._run_unlimited(step)
        else:
            ticks = self._run_fixed(dt, setting.tps, step)

        instant_tps = ticks / dt if dt > 0 else 0.0
        # EMA — weight 0.1 on the new sample
        self._measured_tps += (instant_tps - self._measured_tps) * 0.1
        return ticks

    def _run_fixed(self, dt: float, tps: float, step: Callable[[], None]) -> int:
        self._accumulator += dt * tps
        ticks = 0
        while self._accumulator >= 1.0 and ticks < _MAX_TICKS_PER_FRAME:
            step()
            self._accumulator -= 1.0
            ticks += 1
        # do not carry a backlog larger than ~3 ticks
        if self._accumulator > 3.0:
            self._accumulator = 3.0
        return ticks

    @staticmethod
    def _run_unlimited(step: Callable[[], None]) -> int:
        deadline = time.perf_counter() + _UNLIMITED_BUDGET_S
        ticks = 0
        # check the clock every 64 ticks to limit perf_counter() overhead
        while ticks < _MAX_TICKS_PER_FRAME:
            for _ in range(64):
                step()
            ticks += 64
            if time.perf_counter() >= deadline:
                break
        return ticks

    def _decay_tps(self, dt: float) -> None:
        if dt > 0:
            self._measured_tps += (0.0 - self._measured_tps) * 0.1
