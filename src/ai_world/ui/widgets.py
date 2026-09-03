"""Minimal widget set: a button and a text input."""
from __future__ import annotations

from typing import Callable

import pygame

from ai_world.ui import theme


class Button:
    def __init__(
        self,
        rect: tuple[int, int, int, int],
        label: str,
        on_click: Callable[[], None],
        *,
        color: tuple[int, int, int] = theme.ACCENT_DIM,
        enabled: bool = True,
    ):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.color = color
        self.enabled = enabled
        self._hover = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        elif (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.enabled
            and self.rect.collidepoint(event.pos)
        ):
            self.on_click()

    def draw(self, surface: pygame.Surface, fonts: theme.Fonts) -> None:
        if not self.enabled:
            bg = theme.PANEL
        elif self._hover:
            bg = tuple(min(255, c + 30) for c in self.color)
        else:
            bg = self.color
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, theme.BORDER, self.rect, width=1, border_radius=6)
        col = theme.TEXT if self.enabled else theme.TEXT_DIM
        text = fonts.get(20).render(self.label, True, col)
        surface.blit(text, text.get_rect(center=self.rect.center))


class TextInput:
    def __init__(
        self,
        rect: tuple[int, int, int, int],
        *,
        text: str = "",
        numeric: bool = False,
        max_len: int = 40,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.numeric = numeric
        self.max_len = max_len
        self.focused = False
        self._cursor_timer = 0.0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                self.focused = False
            elif event.unicode and len(self.text) < self.max_len:
                ch = event.unicode
                if ch.isprintable() and (not self.numeric or ch.isdigit()):
                    self.text += ch

    def update(self, dt: float) -> None:
        self._cursor_timer = (self._cursor_timer + dt) % 1.0

    def value_int(self, default: int = 0) -> int:
        try:
            return int(self.text)
        except ValueError:
            return default

    def draw(self, surface: pygame.Surface, fonts: theme.Fonts) -> None:
        bg = theme.PANEL_LIGHT if self.focused else theme.PANEL
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        border = theme.ACCENT if self.focused else theme.BORDER
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)

        font = fonts.get(20)
        shown = self.text
        if self.focused and self._cursor_timer < 0.5:
            shown += "|"
        text = font.render(shown, True, theme.TEXT)
        surface.blit(text, (self.rect.x + 10, self.rect.centery - text.get_height() // 2))
