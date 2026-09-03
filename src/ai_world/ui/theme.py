"""UI colors and fonts."""
from __future__ import annotations

import pygame

BG = (18, 20, 26)
PANEL = (28, 31, 40)
PANEL_LIGHT = (38, 42, 54)
BORDER = (58, 63, 78)
TEXT = (223, 227, 235)
TEXT_DIM = (146, 152, 166)
ACCENT = (95, 170, 245)
ACCENT_DIM = (58, 104, 150)
DANGER = (224, 96, 96)
OK = (120, 200, 130)
WARN = (240, 190, 90)


class Fonts:
    """Lazily loaded font set (SysFont with a fallback to the default font)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        font = self._cache.get(key)
        if font is None:
            try:
                font = pygame.font.SysFont("Segoe UI,DejaVu Sans,Arial", size, bold=bold)
            except Exception:
                font = pygame.font.Font(None, size)
            self._cache[key] = font
        return font
