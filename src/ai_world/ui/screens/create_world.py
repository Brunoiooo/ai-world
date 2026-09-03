"""World creation screen: name, X/Y dimensions, seed."""
from __future__ import annotations

import random

import pygame

from ai_world import config
from ai_world.ui import theme
from ai_world.ui.screens.base import Screen
from ai_world.ui.widgets import Button, TextInput


class CreateWorld(Screen):
    def on_enter(self) -> None:
        self._error = ""
        self.name = TextInput((0, 0, 320, 40), text="World 1", max_len=40)
        self.width = TextInput((0, 0, 150, 40), text=str(config.DEFAULT_MAP_DIM),
                               numeric=True, max_len=4)
        self.height = TextInput((0, 0, 150, 40), text=str(config.DEFAULT_MAP_DIM),
                                numeric=True, max_len=4)
        self.seed = TextInput((0, 0, 240, 40), text=str(random.randint(0, 2**32 - 1)),
                              numeric=True, max_len=10)
        self._layout()

    def _layout(self) -> None:
        w, h = self.app.surface.get_size()
        cx = w // 2
        top = h // 2 - 150
        self._rows = [
            ("Name", self.name),
            ("Width (X)", self.width),
            ("Height (Y)", self.height),
            ("Seed", self.seed),
        ]
        for i, (_, field) in enumerate(self._rows):
            field.rect.topleft = (cx - 20, top + i * 58)

        from ai_world.ui.screens.main_menu import MainMenu

        by = top + len(self._rows) * 58 + 24
        self.buttons = [
            Button((cx + 240, top + 3 * 58, 140, 40), "Random", self._randomize_seed,
                   color=theme.PANEL_LIGHT),
            Button((cx - 200, by, 180, 48), "Cancel",
                   lambda: self.app.change_screen(MainMenu(self.app)),
                   color=theme.PANEL_LIGHT),
            Button((cx + 20, by, 180, 48), "Create world", self._create,
                   color=theme.ACCENT_DIM),
        ]

    def _randomize_seed(self) -> None:
        self.seed.text = str(random.randint(0, 2**32 - 1))

    def _create(self) -> None:
        name = self.name.text.strip() or "World"
        w = self.width.value_int(config.DEFAULT_MAP_DIM)
        h = self.height.value_int(config.DEFAULT_MAP_DIM)
        if not (config.MIN_MAP_DIM <= w <= config.MAX_MAP_DIM):
            self._error = f"Width must be between {config.MIN_MAP_DIM} and {config.MAX_MAP_DIM}."
            return
        if not (config.MIN_MAP_DIM <= h <= config.MAX_MAP_DIM):
            self._error = f"Height must be between {config.MIN_MAP_DIM} and {config.MAX_MAP_DIM}."
            return
        seed = self.seed.value_int(0)

        world = self.app.repo.create(name, w, h, seed)
        from ai_world.ui.screens.simulation import SimulationScreen

        self.app.change_screen(SimulationScreen(self.app, world))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            self._layout()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            from ai_world.ui.screens.main_menu import MainMenu

            self.app.change_screen(MainMenu(self.app))
            return
        for _, field in self._rows:
            field.handle_event(event)
        for button in self.buttons:
            button.handle_event(event)

    def update(self, dt: float) -> None:
        for _, field in self._rows:
            field.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(theme.BG)
        w, h = surface.get_size()
        cx = w // 2

        title = self.app.fonts.get(34, bold=True).render("New world", True, theme.TEXT)
        surface.blit(title, title.get_rect(center=(cx, h // 2 - 200)))

        font = self.app.fonts.get(18)
        for label, field in self._rows:
            text = font.render(label, True, theme.TEXT_DIM)
            surface.blit(text, (field.rect.x, field.rect.y - 22))
            field.draw(surface, self.app.fonts)

        hint = font.render(
            f"Dimension range: {config.MIN_MAP_DIM}-{config.MAX_MAP_DIM} tiles",
            True, theme.TEXT_DIM,
        )
        surface.blit(hint, (cx - 20, self._rows[0][1].rect.y + 4 * 58 + 4))

        for button in self.buttons:
            button.draw(surface, self.app.fonts)

        if self._error:
            err = font.render(self._error, True, theme.DANGER)
            surface.blit(err, err.get_rect(center=(cx, h // 2 + 190)))
