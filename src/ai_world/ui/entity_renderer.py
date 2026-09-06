"""Draws the organisms on top of the terrain / field overlays.

Beyond the body dot this also visualises, for every organism on screen:

* **perception / emitter ports** -- each genetic port is drawn as a wedge in
  the direction it looks (``heading + angle``, spanning ``arc``, out to
  ``reach``), tinted by its spectral signature. An IN port fills in proportion
  to what it currently senses; an OUT port shows a chevron at its tip that
  brightens with what it is currently emitting.
* **actions** -- a soft glyph while the brain *wants* to attack / eat / mate
  this tick, and a brighter burst when the act actually landed (a hit
  connected, food was drawn off the tile, a mating pair formed).
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from ai_world.ui import theme
from ai_world.ui.camera import Camera
from ai_world.world.entity import Entity
from ai_world.world.population import TRAIT_IX
from ai_world.world.world import World

_IX_SIZE = TRAIT_IX["size"]

_ATTACK = theme.DANGER
_EAT = (120, 205, 130)
_MATE = (232, 120, 205)

# don't draw port wedges for more organisms than this in one frame -- the
# largest-on-screen win the budget, the rest still get action glyphs.
_MAX_PORT_ENTITIES = 240
_MIN_PORT_PX = 4.0  # a wedge shorter than this on screen is not worth drawing


def signature_rgb(signature: np.ndarray) -> tuple[int, int, int]:
    """Map the first three signature channels to a legible colour."""
    channels = np.resize(np.clip(signature, 0.0, 1.0), 3)
    return tuple(int(70 + 170 * c) for c in channels)  # type: ignore[return-value]


def _get(arr: np.ndarray, i: int) -> bool:
    return bool(arr[i]) if arr is not None and i < arr.shape[0] else False


class EntityRenderer:
    def __init__(self) -> None:
        self._overlay: pygame.Surface | None = None

    def _port_overlay(self, size: tuple[int, int]) -> pygame.Surface:
        if self._overlay is None or self._overlay.get_size() != size:
            self._overlay = pygame.Surface(size, pygame.SRCALPHA)
        else:
            self._overlay.fill((0, 0, 0, 0))
        return self._overlay

    def draw(
        self,
        surface: pygame.Surface,
        camera: Camera,
        world: World,
        *,
        show_ports: bool = True,
        show_actions: bool = True,
    ) -> None:
        pop = world.population
        if pop is None or not len(pop):
            return
        x0, y0, x1, y1 = camera.visible_tile_bounds()
        visible = np.flatnonzero(
            (pop.x >= x0) & (pop.x < x1) & (pop.y >= y0) & (pop.y < y1)
        )
        zoom = camera.zoom
        sizes = pop.traits[:, _IX_SIZE]

        empty = np.zeros(0, dtype=bool)
        want_atk = getattr(pop, "i_attack", empty)
        want_mate = getattr(pop, "i_mate", empty)
        want_eat = pop.i_eat.any(axis=1) if getattr(pop, "i_eat", empty).size else empty
        did_atk = getattr(pop, "acted_attack", empty)
        did_eat = getattr(pop, "acted_eat", empty)
        did_mate = getattr(pop, "acted_mate", empty)

        brains = getattr(pop, "brains", None)
        if show_ports and brains is not None:
            radii = sizes[visible] * zoom * 0.45
            order = visible[np.argsort(-radii)][:_MAX_PORT_ENTITIES]
            overlay = self._port_overlay(surface.get_size())
            for i in order:
                self._draw_ports(overlay, camera, pop, brains, int(i))
            surface.blit(overlay, (0, 0))

        for i in visible:
            i = int(i)
            sx, sy = camera.world_to_screen(pop.x[i], pop.y[i])
            colour = signature_rgb(pop.signature[i])
            radius = sizes[i] * zoom * 0.45

            if radius < 2.0:
                if show_actions:
                    if _get(did_atk, i) or _get(want_atk, i):
                        colour = _ATTACK
                    elif _get(did_mate, i) or _get(want_mate, i):
                        colour = _MATE
                    elif _get(did_eat, i) or _get(want_eat, i):
                        colour = _EAT
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

            if show_actions:
                self._draw_actions(
                    surface, sx, sy, radius, float(pop.heading[i]),
                    want_atk=_get(want_atk, i), did_atk=_get(did_atk, i),
                    want_eat=_get(want_eat, i), did_eat=_get(did_eat, i),
                    want_mate=_get(want_mate, i), did_mate=_get(did_mate, i),
                )

    # --- ports --------------------------------------------------------
    def _draw_ports(self, overlay, camera: Camera, pop, brains, i: int) -> None:
        genome = pop.genomes[i]
        if not genome.ports:
            return
        in_vals, out_vals = brains.port_readout(i)
        ox, oy = camera.world_to_screen(pop.x[i], pop.y[i])
        heading = float(pop.heading[i])
        ii = io = 0
        for port in genome.ports:
            if port.mode == "in":
                act = in_vals[ii] if ii < len(in_vals) else 0.0
                ii += 1
            else:
                act = out_vals[io] if io < len(out_vals) else 0.0
                io += 1
            self._draw_wedge(overlay, ox, oy, heading, port, camera.zoom, act)

    @staticmethod
    def _draw_wedge(overlay, ox, oy, heading, port, zoom, act) -> None:
        r_px = float(port.reach) * zoom
        if r_px < _MIN_PORT_PX:
            return
        col = signature_rgb(port.signature)
        mag = min(1.0, abs(float(act)))
        base = heading + float(port.angle)
        half = max(0.12, min(math.pi, float(port.arc))) * 0.5
        seg = max(3, int(math.degrees(2 * half) / 12))
        pts = [(ox, oy)]
        for k in range(seg + 1):
            a = base - half + (2 * half) * k / seg
            pts.append((ox + math.cos(a) * r_px, oy + math.sin(a) * r_px))

        if port.mode == "in":
            pygame.draw.polygon(overlay, (*col, int(18 + 150 * mag)), pts)
            pygame.draw.polygon(overlay, (*col, int(70 + 120 * mag)), pts, 1)
        else:
            pygame.draw.polygon(overlay, (*col, int(10 + 40 * mag)), pts)
            pygame.draw.polygon(overlay, (*col, int(60 + 150 * mag)), pts, 1)
            tx, ty = ox + math.cos(base) * r_px, oy + math.sin(base) * r_px
            wing = max(3.0, r_px * 0.06)
            left = (tx + math.cos(base + 2.6) * wing, ty + math.sin(base + 2.6) * wing)
            right = (tx + math.cos(base - 2.6) * wing, ty + math.sin(base - 2.6) * wing)
            pygame.draw.polygon(overlay, (*col, int(90 + 160 * mag)), [(tx, ty), left, right])

    # --- action glyphs ----------------------------------------------
    @staticmethod
    def _draw_actions(
        surface, sx, sy, radius, heading, *,
        want_atk, did_atk, want_eat, did_eat, want_mate, did_mate,
    ) -> None:
        cx, cy = int(sx), int(sy)

        if want_mate or did_mate:
            rr = int(radius * 1.9 + 3)
            pygame.draw.circle(surface, _MATE, (cx, cy), rr, 2 if did_mate else 1)
            if did_mate:
                pygame.draw.circle(surface, _MATE, (cx, cy), rr + 3, 1)

        if want_atk or did_atk:
            reach = radius * (2.6 if did_atk else 2.0)
            w = max(1, int(radius / 2)) if did_atk else 1
            for off in (-0.5, 0.0, 0.5) if did_atk else (0.0,):
                a = heading + off
                pygame.draw.line(
                    surface, _ATTACK, (cx, cy),
                    (cx + math.cos(a) * reach, cy + math.sin(a) * reach), w,
                )

        if want_eat or did_eat:
            ey = int(cy + radius + 4)
            if did_eat:
                pygame.draw.circle(surface, _EAT, (cx, ey), 4)
                pygame.draw.circle(surface, _EAT, (cx, ey), 7, 1)
            else:
                pygame.draw.circle(surface, _EAT, (cx, ey), 3, 1)

    @staticmethod
    def pick(world: World, wx: float, wy: float, radius_tiles: float = 3.0) -> Entity | None:
        if world.population is None:
            return None
        return world.population.nearest(wx, wy, radius_tiles)
