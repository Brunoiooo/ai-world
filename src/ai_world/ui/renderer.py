"""Tile grid renderer.

The grid is drawn through a single base surface (1 pixel = 1 tile); each frame
we cut out the visible slice and scale it to the camera zoom. The base surface
is cached and rebuilt only when the terrain changes.
"""
from __future__ import annotations

import numpy as np
import pygame

from ai_world.ui import theme
from ai_world.ui.camera import Camera
from ai_world.world.grid import Grid
from ai_world.world.tiles import PALETTE


class GridRenderer:
    def __init__(self, grid: Grid):
        self._grid = grid
        self._base = self._build_surface(grid)
        self._dirty = False

    def set_grid(self, grid: Grid) -> None:
        self._grid = grid
        self._base = self._build_surface(grid)

    def mark_dirty(self) -> None:
        self._dirty = True

    @staticmethod
    def _build_surface(grid: Grid) -> pygame.Surface:
        rgb = PALETTE[grid.cells]  # (h, w, 3) uint8
        rgb = np.ascontiguousarray(rgb)
        return pygame.image.frombuffer(
            rgb.tobytes(), (grid.width, grid.height), "RGB"
        ).convert()

    def draw(self, surface: pygame.Surface, camera: Camera) -> None:
        if self._dirty:
            self._base = self._build_surface(self._grid)
            self._dirty = False

        surface.fill(theme.BG)
        x0, y0, x1, y1 = camera.visible_tile_bounds()
        if x1 <= x0 or y1 <= y0:
            return

        region = self._base.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        dest_w = max(1, round((x1 - x0) * camera.zoom))
        dest_h = max(1, round((y1 - y0) * camera.zoom))
        scaled = pygame.transform.scale(region, (dest_w, dest_h))
        surface.blit(scaled, camera.world_to_screen(x0, y0))

        if camera.zoom >= 6:
            self._draw_grid_lines(surface, camera, x0, y0, x1, y1)

    @staticmethod
    def _draw_grid_lines(
        surface: pygame.Surface, camera: Camera, x0: int, y0: int, x1: int, y1: int
    ) -> None:
        line = (0, 0, 0, 40)
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        top_y = camera.world_to_screen(x0, y0)[1]
        bot_y = camera.world_to_screen(x0, y1)[1]
        for tx in range(x0, x1 + 1):
            sx = camera.world_to_screen(tx, y0)[0]
            pygame.draw.line(overlay, line, (sx, top_y), (sx, bot_y))
        left_x = camera.world_to_screen(x0, y0)[0]
        right_x = camera.world_to_screen(x1, y0)[0]
        for ty in range(y0, y1 + 1):
            sy = camera.world_to_screen(x0, ty)[1]
            pygame.draw.line(overlay, line, (left_x, sy), (right_x, sy))
        surface.blit(overlay, (0, 0))
