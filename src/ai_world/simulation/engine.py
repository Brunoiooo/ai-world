"""Engine: a single simulation step of the world.

At this stage a step only increments the tick counter. ``systems`` is the
extension point — a list of callables ``(world) -> None`` run every tick
(e.g. future terrain or entity logic).
"""
from __future__ import annotations

from typing import Callable

from ai_world.world.world import World

System = Callable[[World], None]


class Engine:
    def __init__(self, world: World, systems: list[System] | None = None):
        self.world = world
        self.systems: list[System] = list(systems or [])
        self.steps_this_session: int = 0

    def step(self) -> None:
        for system in self.systems:
            system(self.world)
        self.world.tick += 1
        self.steps_this_session += 1
