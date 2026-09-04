"""The always-on food layer.

Two looks, picked by zoom:

* Zoomed in (a tile is a few pixels or more) food is a scatter of coloured
  "morsel" specks -- one colour per :data:`ai_world.world.food.FOOD_TYPES`
  entry, the count on a tile tracking how much of that type sits there. Every
  channel, including the organism-made ``enzyme`` and ``carrion``, gets its own
  dot, so food reads on the map the way the organisms do.
* Zoomed out, where specks would just overlap into mush, it falls back to a
  translucent colour wash (weighted blend of the per-type tints, faded by total
  density) so the big picture stays legible.

Speck placement is a deterministic hash of the tile plus the morsel's ordinal,
so the dots hold still between frames instead of shimmering.

Unlike :class:`ai_world.ui.field_overlay.FieldOverlay` (an optional inspector for
one field at a time) this is shown by default, like the organisms.
"""
from __future__ import annotations

import numpy as np
import pygame

from ai_world.ui.camera import Camera
from ai_world.world.food import FOOD_TYPES
from ai_world.world.world import World

_COLORS: tuple[tuple[int, int, int], ...] = tuple(f.color for f in FOOD_TYPES)
_COLOR_ARR = np.array(_COLORS, dtype=np.float32)          # (K, 3)

_SPECK_ZOOM = 6.0            # px/tile at or above which we draw specks, not a wash
_MORSEL_UNIT = 0.15         # food density one speck stands for
_MAX_PER_TILE = 5           # speck cap per tile per channel
_BUDGET = 4500              # total specks per frame, split evenly across channels
_WASH_ALPHA = 125           # peak opacity of the zoomed-out blend


def _hash01(a: np.ndarray, b: np.ndarray, salt: int) -> np.ndarray:
    """Stable pseudo-random floats in [0, 1) from integer coordinate arrays."""
    h = (a * 73856093) ^ (b * 19349663) ^ (salt * 2654435761)
    h = (h ^ (h >> np.int64(13))) & np.int64(0x7FFFFFFF)
    return (h % 9973) / 9973.0


class FoodRenderer:
    def __init__(self) -> None:
        self._dots: dict[tuple[int, int], pygame.Surface] = {}

    def draw(self, surface: pygame.Surface, camera: Camera, world: World) -> None:
        if not world.ecosystem_enabled:
            return
        x0, y0, x1, y1 = camera.visible_tile_bounds()
        if x1 <= x0 or y1 <= y0:
            return
        planes = world.food.values[:, y0:y1, x0:x1]                     # (K, h, w)
        if camera.zoom >= _SPECK_ZOOM:
            self._draw_specks(surface, camera, planes, x0, y0)
        else:
            self._draw_wash(surface, camera, planes, x0, y0)

    # -- zoomed in: one speck per morsel ---------------------------------
    def _dot(self, channel: int, radius: int) -> pygame.Surface:
        key = (channel, radius)
        dot = self._dots.get(key)
        if dot is None:
            dot = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(dot, (*_COLORS[channel], 235), (radius, radius), radius)
            self._dots[key] = dot
        return dot

    def _draw_specks(
        self, surface: pygame.Surface, camera: Camera, planes: np.ndarray, x0: int, y0: int
    ) -> None:
        counts = np.minimum(
            _MAX_PER_TILE, np.floor(planes / _MORSEL_UNIT).astype(np.int64)
        )
        if not counts.any():
            return
        zoom = camera.zoom
        vw, vh = camera.viewport
        ox = vw / 2.0 - camera.cx * zoom
        oy = vh / 2.0 - camera.cy * zoom
        radius = max(1, int(round(zoom * 0.16)))
        per_channel = _BUDGET // len(_COLORS)

        for k in range(len(_COLORS)):
            ys, xs = np.nonzero(counts[k])
            if not len(xs):
                continue
            per = counts[k][ys, xs]
            # thin the tile list before expanding so the work stays bounded
            if per.sum() > per_channel:
                stride = int(per.sum() // per_channel) + 1
                xs, ys, per = xs[::stride], ys[::stride], per[::stride]

            tx = (xs + x0).astype(np.int64)
            ty = (ys + y0).astype(np.int64)
            rep_x = np.repeat(tx, per)
            rep_y = np.repeat(ty, per)
            ordinal = np.arange(rep_x.size) - np.repeat(np.cumsum(per) - per, per)

            jx = _hash01(rep_x * 4 + ordinal, rep_y, k * 2 + 1)
            jy = _hash01(rep_y * 4 + ordinal, rep_x, k * 2 + 2)
            sx = ox + (rep_x + jx) * zoom
            sy = oy + (rep_y + jy) * zoom

            dot = self._dot(k, radius)
            surface.blits(
                [(dot, (int(px) - radius, int(py) - radius)) for px, py in zip(sx, sy)],
                doreturn=0,
            )

    # -- zoomed out: translucent blend ---------------------------------
    def _draw_wash(
        self, surface: pygame.Surface, camera: Camera, planes: np.ndarray, x0: int, y0: int
    ) -> None:
        h, w = planes.shape[1:]
        weight = np.clip(planes, 0.0, 1.0)
        total = weight.sum(axis=0)
        norm = np.maximum(total, 1e-4)

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(
            np.einsum("kyx,kc->yxc", weight, _COLOR_ARR) / norm[..., None], 0, 255
        ).astype(np.uint8)
        rgba[..., 3] = (np.clip(total, 0.0, 1.0) * _WASH_ALPHA).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)

        base = pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA")
        dest = (max(1, round(w * camera.zoom)), max(1, round(h * camera.zoom)))
        scaled = pygame.transform.scale(base, dest)
        surface.blit(scaled, camera.world_to_screen(x0, y0))
