"""World browser: select, load, delete."""
from __future__ import annotations

import pygame

from ai_world.ui import theme
from ai_world.ui.screens.base import Screen
from ai_world.ui.widgets import Button

_ROW_H = 60
_LIST_TOP = 110


class WorldBrowser(Screen):
    def on_enter(self) -> None:
        self._refresh()
        self._selected: int | None = None
        self._scroll = 0
        self._confirm_delete = False
        self._layout()

    def _refresh(self) -> None:
        self._worlds = self.app.repo.list_worlds()

    def _layout(self) -> None:
        w, h = self.app.surface.get_size()
        self._list_rect = pygame.Rect(40, _LIST_TOP, w - 80, h - _LIST_TOP - 90)
        by = h - 70

        from ai_world.ui.screens.main_menu import MainMenu

        self.buttons = [
            Button((40, by, 160, 48), "Back",
                   lambda: self.app.change_screen(MainMenu(self.app)),
                   color=theme.PANEL_LIGHT),
            Button((w - 40 - 160, by, 160, 48), "Load", self._load,
                   color=theme.ACCENT_DIM),
            Button((w - 40 - 340, by, 160, 48), "Delete", self._ask_delete,
                   color=theme.PANEL_LIGHT),
        ]
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has_sel = self._selected is not None
        for button in self.buttons:
            if button.label in ("Load", "Delete"):
                button.enabled = has_sel

    # --- actions --------------------------------------------------
    def _load(self) -> None:
        if self._selected is None:
            return
        summary = self._worlds[self._selected]
        world = self.app.repo.load(summary.id)
        from ai_world.ui.screens.simulation import SimulationScreen

        self.app.change_screen(SimulationScreen(self.app, world))

    def _ask_delete(self) -> None:
        if self._selected is not None:
            self._confirm_delete = True

    def _do_delete(self) -> None:
        if self._selected is not None:
            self.app.repo.delete(self._worlds[self._selected].id)
        self._confirm_delete = False
        self._selected = None
        self._refresh()
        self._sync_buttons()

    # --- events --------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            self._layout()

        if self._confirm_delete:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_y):
                    self._do_delete()
                elif event.key in (pygame.K_ESCAPE, pygame.K_n):
                    self._confirm_delete = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_confirm_click(event.pos)
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            from ai_world.ui.screens.main_menu import MainMenu

            self.app.change_screen(MainMenu(self.app))
            return

        if event.type == pygame.MOUSEWHEEL:
            self._scroll = max(0, self._scroll - event.y * 40)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_list_click(event.pos)

        for button in self.buttons:
            button.handle_event(event)

    def _handle_list_click(self, pos: tuple[int, int]) -> None:
        if not self._list_rect.collidepoint(pos):
            return
        rel_y = pos[1] - self._list_rect.y + self._scroll
        index = rel_y // _ROW_H
        if 0 <= index < len(self._worlds):
            self._selected = int(index)
            self._sync_buttons()

    def _confirm_buttons(self) -> tuple[pygame.Rect, pygame.Rect]:
        w, h = self.app.surface.get_size()
        box = pygame.Rect(w // 2 - 220, h // 2 - 90, 440, 180)
        yes = pygame.Rect(box.x + 40, box.bottom - 60, 160, 44)
        no = pygame.Rect(box.right - 200, box.bottom - 60, 160, 44)
        return yes, no

    def _handle_confirm_click(self, pos: tuple[int, int]) -> None:
        yes, no = self._confirm_buttons()
        if yes.collidepoint(pos):
            self._do_delete()
        elif no.collidepoint(pos):
            self._confirm_delete = False

    # --- drawing ------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(theme.BG)
        w, h = surface.get_size()

        title = self.app.fonts.get(32, bold=True).render("Load world", True, theme.TEXT)
        surface.blit(title, (40, 40))

        if not self._worlds:
            msg = self.app.fonts.get(20).render(
                "No saved worlds. Go back and create a new one.", True, theme.TEXT_DIM
            )
            surface.blit(msg, (40, _LIST_TOP + 10))
        else:
            self._draw_list(surface)

        for button in self.buttons:
            button.draw(surface, self.app.fonts)

        if self._confirm_delete:
            self._draw_confirm(surface)

    def _draw_list(self, surface: pygame.Surface) -> None:
        prev_clip = surface.get_clip()
        surface.set_clip(self._list_rect)
        name_font = self.app.fonts.get(22, bold=True)
        meta_font = self.app.fonts.get(16)

        for i, world in enumerate(self._worlds):
            row = pygame.Rect(
                self._list_rect.x,
                self._list_rect.y + i * _ROW_H - self._scroll,
                self._list_rect.w,
                _ROW_H - 6,
            )
            if row.bottom < self._list_rect.top or row.top > self._list_rect.bottom:
                continue
            bg = theme.PANEL_LIGHT if i == self._selected else theme.PANEL
            pygame.draw.rect(surface, bg, row, border_radius=6)
            if i == self._selected:
                pygame.draw.rect(surface, theme.ACCENT, row, width=1, border_radius=6)

            surface.blit(name_font.render(world.name, True, theme.TEXT),
                         (row.x + 14, row.y + 8))
            meta = (f"{world.width}x{world.height}  ·  tick {world.tick:,}  ·  "
                    f"updated {world.updated_at:%Y-%m-%d %H:%M}")
            surface.blit(meta_font.render(meta, True, theme.TEXT_DIM),
                         (row.x + 14, row.y + 34))

        surface.set_clip(prev_clip)

    def _draw_confirm(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        surface.blit(shade, (0, 0))

        box = pygame.Rect(w // 2 - 220, h // 2 - 90, 440, 180)
        pygame.draw.rect(surface, theme.PANEL, box, border_radius=10)
        pygame.draw.rect(surface, theme.BORDER, box, width=1, border_radius=10)

        name = self._worlds[self._selected].name if self._selected is not None else ""
        q = self.app.fonts.get(20).render(f'Delete world "{name}"?', True, theme.TEXT)
        surface.blit(q, q.get_rect(center=(box.centerx, box.y + 46)))
        info = self.app.fonts.get(15).render(
            "This cannot be undone.", True, theme.TEXT_DIM
        )
        surface.blit(info, info.get_rect(center=(box.centerx, box.y + 74)))

        yes, no = self._confirm_buttons()
        pygame.draw.rect(surface, theme.DANGER, yes, border_radius=6)
        pygame.draw.rect(surface, theme.PANEL_LIGHT, no, border_radius=6)
        yt = self.app.fonts.get(18).render("Delete (Y)", True, theme.TEXT)
        nt = self.app.fonts.get(18).render("Cancel (N)", True, theme.TEXT)
        surface.blit(yt, yt.get_rect(center=yes.center))
        surface.blit(nt, nt.get_rect(center=no.center))
