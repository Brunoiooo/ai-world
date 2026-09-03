"""2D camera: position in tiles (screen center) + zoom in pixels per tile."""
from __future__ import annotations

import pygame

MIN_ZOOM = 0.05
MAX_ZOOM = 48.0


class Camera:
    def __init__(self, world_w: int, world_h: int, viewport: tuple[int, int]):
        self.world_w = world_w
        self.world_h = world_h
        self.cx = world_w / 2.0
        self.cy = world_h / 2.0
        self.zoom = 1.0
        self.viewport = viewport
        self.fit(viewport)

    # --- settings ---------------------------------------------------
    def set_viewport(self, viewport: tuple[int, int]) -> None:
        self.viewport = viewport

    def fit(self, viewport: tuple[int, int]) -> None:
        vw, vh = viewport
        self.viewport = viewport
        self.zoom = _clamp(min(vw / self.world_w, vh / self.world_h), MIN_ZOOM, MAX_ZOOM)
        self.cx = self.world_w / 2.0
        self.cy = self.world_h / 2.0

    # --- transforms ------------------------------------------------------
    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        vw, vh = self.viewport
        return (vw / 2 + (wx - self.cx) * self.zoom, vh / 2 + (wy - self.cy) * self.zoom)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        vw, vh = self.viewport
        return (self.cx + (sx - vw / 2) / self.zoom, self.cy + (sy - vh / 2) / self.zoom)

    # --- interactions ---------------------------------------------------
    def pan_pixels(self, dx: float, dy: float) -> None:
        self.cx += dx / self.zoom
        self.cy += dy / self.zoom
        self._clamp_center()

    def zoom_at(self, screen_pos: tuple[float, float], factor: float) -> None:
        before = self.screen_to_world(*screen_pos)
        self.zoom = _clamp(self.zoom * factor, MIN_ZOOM, MAX_ZOOM)
        after = self.screen_to_world(*screen_pos)
        self.cx += before[0] - after[0]
        self.cy += before[1] - after[1]
        self._clamp_center()

    def _clamp_center(self) -> None:
        self.cx = _clamp(self.cx, 0.0, float(self.world_w))
        self.cy = _clamp(self.cy, 0.0, float(self.world_h))

    # --- visible slice -----------------------------------------------
    def visible_tile_bounds(self) -> tuple[int, int, int, int]:
        vw, vh = self.viewport
        left, top = self.screen_to_world(0, 0)
        right, bottom = self.screen_to_world(vw, vh)
        x0 = max(0, int(left))
        y0 = max(0, int(top))
        x1 = min(self.world_w, int(right) + 1)
        y1 = min(self.world_h, int(bottom) + 1)
        return x0, y0, x1, y1


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
