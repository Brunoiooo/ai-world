"""Draws the organisms on top of the terrain / field overlays."""
from __future__ import annotations

import math

import numpy as np
import pygame

from ai_world.ui.camera import Camera
from ai_world.world.entity import Entity
from ai_world.world.population import TRAIT_IX
from ai_world.world.world import World

_IX_SIZE = TRAIT_IX["size"]


def signature_rgb(signature: np.ndarray) -> tuple[int, int, int]:
    """Map the first three signature channels to a legible colour."""
    channels = np.resize(np.clip(signature, 0.0, 1.0), 3)
    return tuple(int(70 + 170 * c) for c in channels)  # type: ignore[return-value]


class EntityRenderer:
    def draw(self, surface: pygame.Surface, camera: Camera, world: World) -> None:
        pop = world.population
        if pop is None or not len(pop):
            return
        x0, y0, x1, y1 = camera.visible_tile_bounds()
        visible = np.flatnonzero(
            (pop.x >= x0) & (pop.x < x1) & (pop.y >= y0) & (pop.y < y1)
        )
        zoom = camera.zoom
        sizes = pop.traits[:, _IX_SIZE]

        for i in visible:
            sx, sy = camera.world_to_screen(pop.x[i], pop.y[i])
            colour = signature_rgb(pop.signature[i])
            radius = sizes[i] * zoom * 0.45
            if radius < 2.0:
                surface.set_at((int(sx), int(sy)), colour)
                continue
            pygame.draw.circle(surface, colour, (sx, sy), radius)
            if pop.hp[i] < 1.0:
                pygame.draw.circle(surface, (210, 70, 70), (sx, sy), radius, 1)
            tip = (
                sx + math.cos(pop.heading[i]) * radius * 1.7,
                sy + math.sin(pop.heading[i]) * radius * 1.7,
            )
            pygame.draw.line(surface, (15, 15, 20), (sx, sy), tip, max(1, int(radius / 3)))

    @staticmethod
    def pick(world: World, wx: float, wy: float, radius_tiles: float = 3.0) -> Entity | None:
        if world.population is None:
            return None
        return world.population.nearest(wx, wy, radius_tiles)
