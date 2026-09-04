"""The always-on food layer.

Draws :class:`ai_world.world.fields.FoodField` as a translucent colour wash on
top of the terrain -- one tint per :data:`ai_world.world.food.FOOD_TYPES` entry,
blended by how much of each type sits on a tile, faded by total density. Uses the
same "small base surface, scaled visible slice" trick as
:class:`ai_world.ui.renderer.GridRenderer`.

Unlike :class:`ai_world.ui.field_overlay.FieldOverlay` (an optional inspector for
one field at a time) this is shown by default, like the organisms.
"""
from __future__ import annotations

import numpy as np
import pygame

from ai_world.ui.camera import Camera
from ai_world.world.food import FOOD_TYPES
from ai_world.world.world import World

_MAX_ALPHA = 125  # let the terrain read through where food is dense
_COLORS = np.array([f.color for f in FOOD_TYPES], dtype=np.float32)  # (K, 3)


class FoodRenderer:
    def draw(self, surface: pygame.Surface, camera: Camera, world: World) -> None:
        if not world.ecosystem_enabled:
            return
        x0, y0, x1, y1 = camera.visible_tile_bounds()
        if x1 <= x0 or y1 <= y0:
            return

        planes = world.food.values[:, y0:y1, x0:x1]          # (K, h, w)
        weight = np.clip(planes, 0.0, 1.0)
        total = weight.sum(axis=0)                            # (h, w)
        norm = np.maximum(total, 1e-4)

        rgba = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(
            np.einsum("kyx,kc->yxc", weight, _COLORS) / norm[..., None], 0, 255
        ).astype(np.uint8)
        rgba[..., 3] = (np.clip(total, 0.0, 1.0) * _MAX_ALPHA).astype(np.uint8)
        rgba = np.ascontiguousarray(rgba)

        base = pygame.image.frombuffer(rgba.tobytes(), (x1 - x0, y1 - y0), "RGBA")
        dest_w = max(1, round((x1 - x0) * camera.zoom))
        dest_h = max(1, round((y1 - y0) * camera.zoom))
        scaled = pygame.transform.scale(base, (dest_w, dest_h))
        surface.blit(scaled, camera.world_to_screen(x0, y0))
