"""Ecosystem simulation systems.

Each system is a ``(World) -> None`` callable registered on
:class:`ai_world.simulation.engine.Engine`. They run in list order every tick;
a system that should run less often gates itself on ``world.tick``.

Phase 1 covers the physical substrate only (climate + fields). Organisms,
perception, brains, reproduction and speciation are added by later phases.
"""
from __future__ import annotations

from typing import Callable

from ai_world.world.world import World

System = Callable[[World], None]


def weather_system(world: World) -> None:
    params = world.eco_params
    assert params and world.weather and world.spectrum
    if world.tick % params.weather_interval != 0:
        return
    world.weather.advance(world.tick, params)
    world.refresh_temperature()


def fields_system(world: World) -> None:
    params = world.eco_params
    assert params and world.enzymes and world.spectrum and world.temperature and world.weather
    world.enzymes.step(world.weather.regen_multiplier)
    world.spectrum.step()
    # Reserved channels are authoritative each tick — overwrite whatever decay,
    # diffusion or organism emissions left behind.
    world.spectrum.set_channel(params.enzyme_channel, world.enzymes.values)
    world.spectrum.set_channel(params.temperature_channel, world.temperature.values)


def default_systems() -> list[System]:
    return [weather_system, fields_system]
