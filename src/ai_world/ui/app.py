"""Application core: window, main loop, screen switching, shared resources."""
from __future__ import annotations

import pygame

from ai_world import config
from ai_world.persistence import WorldRepository, connect, init_db
from ai_world.ui import theme
from ai_world.ui.screens.base import Screen


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("ai-world")
        self.surface = pygame.display.set_mode(
            config.DEFAULT_WINDOW_SIZE, pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.fonts = theme.Fonts()
        self.running = True

        self.conn = connect(config.database_path())
        init_db(self.conn)
        self.repo = WorldRepository(self.conn)

        self._screen: Screen | None = None
        # import here to avoid an import cycle
        from ai_world.ui.screens.main_menu import MainMenu

        self.change_screen(MainMenu(self))

    # --- screens ---------------------------------------------------
    @property
    def screen(self) -> Screen:
        assert self._screen is not None
        return self._screen

    def change_screen(self, screen: Screen) -> None:
        if self._screen is not None:
            self._screen.on_exit()
        self._screen = screen
        screen.on_enter()

    def quit(self) -> None:
        self.running = False

    # --- loop -----------------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(config.TARGET_FPS) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.surface = pygame.display.set_mode(
                            (event.w, event.h), pygame.RESIZABLE
                        )
                    self.screen.handle_event(event)
                self.screen.update(dt)
                self.screen.draw(self.surface)
                pygame.display.flip()
        finally:
            self.conn.close()
            pygame.quit()
