"""The food types organisms eat.

Replaces the old single "enzyme" resource. Food is a multi-channel field
(:class:`ai_world.world.fields.FoodField`): a few biome-grown types that spread
on their own terrain, plus two types produced by the organisms themselves --
``enzyme`` (excreted while feeding) and ``carrion`` (left on death).

An organism's genetic ``diet`` vector (:mod:`ai_world.world.genome`) has one
weight per type here: a high weight digests that type well, a negative weight
makes it toxic. ``FOOD_TYPES`` is a pinned registry -- its length feeds the
brain's fixed node layout, the same way ``spectrum_channels`` does.
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_world.world.tiles import Tile


@dataclass(frozen=True)
class FoodType:
    name: str
    source: str                  # "grow" | "excrete" | "carrion"
    terrain: int | None          # Tile this type colonizes, when source == "grow"
    color: tuple[int, int, int]  # tint for the map layer


FOOD_TYPES: tuple[FoodType, ...] = (
    FoodType("grass forage", "grow", int(Tile.GRASS), (124, 200, 84)),
    FoodType("forest mast", "grow", int(Tile.FOREST), (110, 150, 60)),
    FoodType("humus", "grow", int(Tile.DIRT), (150, 96, 58)),
    FoodType("algae", "grow", int(Tile.WATER), (70, 180, 175)),
    FoodType("enzyme", "excrete", None, (222, 208, 96)),
    FoodType("carrion", "carrion", None, (200, 72, 92)),
)

N_FOOD_TYPES = len(FOOD_TYPES)

ENZYME_IX = next(i for i, f in enumerate(FOOD_TYPES) if f.source == "excrete")
CARRION_IX = next(i for i, f in enumerate(FOOD_TYPES) if f.source == "carrion")

FOOD_NAMES: tuple[str, ...] = tuple(f.name for f in FOOD_TYPES)
