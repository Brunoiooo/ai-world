"""Optional colour overlays for the ecosystem's continuous fields.

Draws one field (temperature or a single spectrum channel) as a translucent
layer on top of the terrain, using the same "small base surface, scaled visible
slice" approach as :class:`ai_world.ui.renderer.GridRenderer`. Food has its own
always-on layer (:mod:`ai_world.ui.food_renderer`).
"""
from __future__ import annotations

import numpy as np
import pygame

from ai_world.ui.camera import Camera
from ai_world.world.world import World

_ALPHA = 150


def _ramp(t: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
    t = t[..., None]
    return (np.array(low) * (1.0 - t) + np.array(high) * t).astype(np.uint8)


def _diverging(t: np.ndarray) -> np.ndarray:
    """0 -> blue, 0.5 -> pale, 1 -> red (temperature)."""
    cold = _ramp(np.clip(t * 2.0, 0.0, 1.0), (40, 90, 200), (235, 235, 235))
    warm = _ramp(np.clip((t - 0.5) * 2.0, 0.0, 1.0), (235, 235, 235), (220, 70, 50))
    return np.where(t[..., None] < 0.5, cold, warm)


class FieldOverlay:
    def __init__(self) -> None:
        # None, "temperature", or ("spectrum", channel_index)
        self._mode: object = None

    @property
    def label(self) -> str | None:
        if self._mode is None:
            return None
        if isinstance(self._mode, tuple):
            return f"spectrum ch{self._mode[1]}"
        return str(self._mode)

    def toggle(self, mode: object) -> None:
        self._mode = None if self._mode == mode else mode

    def _plane(self, world: World) -> np.ndarray | None:
        mode = self._mode
        if mode == "temperature":
            return world.temperature.values  # type: ignore[union-attr]
        if isinstance(mode, tuple) and mode[0] == "spectrum":
            spectrum = world.spectrum
            if spectrum is None or mode[1] >= spectrum.channels:
                return None
            return spectrum.values[mode[1]]
        return None

    def _rgb(self, plane: np.ndarray) -> np.ndarray:
        if self._mode == "temperature":
            return _diverging(np.clip(plane, 0.0, 1.0))
        norm = plane / max(float(plane.max()), 1e-4)
        return _ramp(np.clip(norm, 0.0, 1.0), (20, 20, 30), (230, 120, 240))

    def draw(self, surface: pygame.Surface, camera: Camera, world: World) -> None:
        if self._mode is None or not world.ecosystem_enabled:
            return
        plane = self._plane(world)
        if plane is None:
            return

        x0, y0, x1, y1 = camera.visible_tile_bounds()
        if x1 <= x0 or y1 <= y0:
            return

        rgb = np.ascontiguousarray(self._rgb(plane)[y0:y1, x0:x1])
        base = pygame.image.frombuffer(rgb.tobytes(), (x1 - x0, y1 - y0), "RGB")
        dest_w = max(1, round((x1 - x0) * camera.zoom))
        dest_h = max(1, round((y1 - y0) * camera.zoom))
        scaled = pygame.transform.scale(base, (dest_w, dest_h))
        scaled.set_alpha(_ALPHA)
        surface.blit(scaled, camera.world_to_screen(x0, y0))
