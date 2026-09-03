"""Main menu: new world / load / quit."""
from __future__ import annotations

import pygame

from ai_world.ui import theme
from ai_world.ui.screens.base import Screen
from ai_world.ui.widgets import Button


class MainMenu(Screen):
    def on_enter(self) -> None:
        self._build_buttons()

    def _build_buttons(self) -> None:
        w, h = self.app.surface.get_size()
        bw, bh, gap = 260, 52, 16
        x = w // 2 - bw // 2
        y = h // 2 - 40

        from ai_world.ui.screens.create_world import CreateWorld
        from ai_world.ui.screens.world_browser import WorldBrowser

        self.buttons = [
            Button((x, y, bw, bh), "New world",
                   lambda: self.app.change_screen(CreateWorld(self.app)),
                   color=theme.ACCENT_DIM),
            Button((x, y + (bh + gap), bw, bh), "Load world",
                   lambda: self.app.change_screen(WorldBrowser(self.app))),
            Button((x, y + 2 * (bh + gap), bw, bh), "Quit",
                   self.app.quit, color=theme.PANEL_LIGHT),
        ]

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            self._build_buttons()
        for button in self.buttons:
            button.handle_event(event)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.quit()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(theme.BG)
        w, h = surface.get_size()

        title = self.app.fonts.get(48, bold=True).render("ai-world", True, theme.TEXT)
        surface.blit(title, title.get_rect(center=(w // 2, h // 2 - 150)))
        sub = self.app.fonts.get(18).render(
            "2D world sandbox — tile grid", True, theme.TEXT_DIM
        )
        surface.blit(sub, sub.get_rect(center=(w // 2, h // 2 - 110)))

        for button in self.buttons:
            button.draw(surface, self.app.fonts)
