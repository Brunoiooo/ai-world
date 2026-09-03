"""Base screen. A concrete screen overrides the lifecycle methods it needs."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from ai_world.ui.app import App


class Screen:
    def __init__(self, app: "App"):
        self.app = app

    def on_enter(self) -> None:  # noqa: D401 - hook
        ...

    def on_exit(self) -> None:
        ...

    def handle_event(self, event: pygame.event.Event) -> None:
        ...

    def update(self, dt: float) -> None:
        ...

    def draw(self, surface: pygame.Surface) -> None:
        ...
