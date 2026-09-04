"""Simulation screen: world view + time control + pause menu."""
from __future__ import annotations

import pygame

from ai_world.simulation import Engine, SimulationClock, default_systems
from ai_world.simulation.ecosystem import max_hp
from ai_world.ui import theme
from ai_world.ui.camera import Camera
from ai_world.ui.entity_renderer import EntityRenderer
from ai_world.ui.field_overlay import FieldOverlay
from ai_world.ui.food_renderer import FoodRenderer
from ai_world.ui.renderer import GridRenderer
from ai_world.ui.screens.base import Screen
from ai_world.ui.species_panel import SpeciesPanel
from ai_world.ui.widgets import Button
from ai_world.world.fields import TemperatureField
from ai_world.world.food import FOOD_NAMES
from ai_world.world.tiles import TILE_NAMES

_TOP_BAR = 60
_PAN_SPEED = 900.0  # px/s while holding a key


class SimulationScreen(Screen):
    def __init__(self, app, world):
        super().__init__(app)
        self.world = world
        systems = default_systems() if world.ecosystem_enabled else []
        self.engine = Engine(world, systems)
        self.clock = SimulationClock()

    def on_enter(self) -> None:
        self.camera = Camera(self.world.width, self.world.height, self.app.surface.get_size())
        self.renderer = GridRenderer(self.world.grid)
        self.overlay = FieldOverlay()
        self.food_renderer = FoodRenderer()
        self.entities = EntityRenderer()
        self._selected_id: int | None = None
        self._show_entities = True
        self._show_food = True
        self._show_species = False
        self.species_panel = SpeciesPanel()
        self._dragging = False
        self._menu_open = False
        self._status = ""
        self._status_timer = 0.0
        self._ticks_last_frame = 0
        if self.world.population is not None:
            self.world.population.rebuild_index()
        self._build_menu()

    def _build_menu(self) -> None:
        w, h = self.app.surface.get_size()
        bw, bh, gap = 320, 50, 14
        x = w // 2 - bw // 2
        y = h // 2 - 130
        self._menu_buttons = [
            Button((x, y, bw, bh), "Resume (Esc)", self._close_menu, color=theme.ACCENT_DIM),
            Button((x, y + (bh + gap), bw, bh), "Save", self._save),
            Button((x, y + 2 * (bh + gap), bw, bh), "Save and exit", self._save_and_exit),
            Button((x, y + 3 * (bh + gap), bw, bh), "Exit without saving",
                   self._exit_no_save, color=theme.PANEL_LIGHT),
        ]

    # --- menu actions --------------------------------------------
    def _open_menu(self) -> None:
        self._menu_open = True
        self._menu_resume_paused = self.clock.paused
        self.clock.set_paused(True)

    def _close_menu(self) -> None:
        self._menu_open = False
        self.clock.set_paused(self._menu_resume_paused)

    def _save(self) -> None:
        self.app.repo.save(self.world)
        self._flash(f"Saved (tick {self.world.tick:,})")

    def _save_and_exit(self) -> None:
        self.app.repo.save(self.world)
        self._go_menu()

    def _exit_no_save(self) -> None:
        self._go_menu()

    def _go_menu(self) -> None:
        from ai_world.ui.screens.main_menu import MainMenu

        self.app.change_screen(MainMenu(self.app))

    def _flash(self, text: str) -> None:
        self._status = text
        self._status_timer = 2.5

    # --- events --------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.VIDEORESIZE:
            self.camera.set_viewport((event.w, event.h))
            self._build_menu()
            return

        if self._menu_open:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._close_menu()
            for button in self._menu_buttons:
                button.handle_event(event)
            return

        if self.species_panel.handle_event(event, self.app.surface):
            return

        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key, event.mod)
        elif event.type == pygame.MOUSEWHEEL:
            self.camera.zoom_at(pygame.mouse.get_pos(), 1.12 ** event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (2, 3):
            self._dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self.camera.pan_pixels(-event.rel[0], -event.rel[1])
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._select_at(event.pos)

    def _select_at(self, screen_pos: tuple[int, int]) -> None:
        if self.world.population is None or screen_pos[1] <= _TOP_BAR:
            return
        wx, wy = self.camera.screen_to_world(*screen_pos)
        picked = self.entities.pick(self.world, wx, wy, radius_tiles=6.0 / self.camera.zoom + 1.0)
        self._selected_id = picked.id if picked else None

    def _handle_key(self, key: int, mod: int) -> None:
        if key == pygame.K_ESCAPE:
            self._open_menu()
        elif key == pygame.K_SPACE:
            self.clock.toggle_pause()
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.clock.faster()
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.clock.slower()
        elif key == pygame.K_0:
            self.clock.reset_speed()
        elif key == pygame.K_f:
            self.camera.fit(self.app.surface.get_size())
        elif key == pygame.K_F5 or (key == pygame.K_s and mod & pygame.KMOD_CTRL):
            self._save()
        elif key == pygame.K_h:
            self._show_entities = not self._show_entities
        elif key == pygame.K_TAB:
            self._show_species = not self._show_species
        elif key == pygame.K_g:
            self.species_panel.toggle()
        elif key == pygame.K_c:
            self._show_food = not self._show_food
        elif key == pygame.K_t:
            self.overlay.toggle("temperature")
        elif pygame.K_1 <= key <= pygame.K_6:
            self.overlay.toggle(("spectrum", key - pygame.K_1))

    # --- update --------------------------------------------------
    def update(self, dt: float) -> None:
        if self._status_timer > 0:
            self._status_timer -= dt

        if not self._menu_open:
            self._pan_from_keys(dt)

        self._ticks_last_frame = self.clock.advance(dt, self.engine.step)

    def _pan_from_keys(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
        dy = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
        if dx or dy:
            self.camera.pan_pixels(dx * _PAN_SPEED * dt, dy * _PAN_SPEED * dt)

    # --- drawing ------------------------------------------------
    def draw(self, surface: pygame.Surface) -> None:
        self.renderer.draw(surface, self.camera)
        if self._show_food:
            self.food_renderer.draw(surface, self.camera, self.world)
        self.overlay.draw(surface, self.camera, self.world)
        if self._show_entities:
            self.entities.draw(surface, self.camera, self.world)
        self._draw_top_bar(surface)
        self._draw_hint_bar(surface)
        self._draw_inspector(surface)
        if self._show_species:
            self._draw_species_panel(surface)
        if self.species_panel.open:
            self.species_panel.draw(surface, self.world, self.app.fonts)
        if self._menu_open:
            self._draw_menu(surface)

    def _draw_species_panel(self, surface: pygame.Surface) -> None:
        registry = self.world.species
        if registry is None:
            return
        from ai_world.ui.entity_renderer import signature_rgb

        species = registry.sorted_species()[:14]
        total = max(1, sum(s.count for s in species))
        w = surface.get_width()
        pw, rowh = 250, 22
        panel = pygame.Rect(w - pw - 12, _TOP_BAR + 12, pw, rowh * (len(species) + 1) + 16)
        box = pygame.Surface(panel.size, pygame.SRCALPHA)
        box.fill((*theme.PANEL, 235))
        surface.blit(box, panel.topleft)
        pygame.draw.rect(surface, theme.BORDER, panel, 1)

        font = self.app.fonts.get(14)
        surface.blit(
            self.app.fonts.get(14, bold=True).render(
                f"{len(registry.species)} species  ·  thr {registry.threshold:.1f}",
                True, theme.TEXT),
            (panel.x + 10, panel.y + 8),
        )
        for i, sp in enumerate(species):
            y = panel.y + 12 + (i + 1) * rowh
            colour = signature_rgb(sp.representative.body_signature)
            pygame.draw.rect(surface, colour, (panel.x + 10, y + 3, 12, 12))
            surface.blit(font.render(f"#{sp.id}", True, theme.TEXT_DIM), (panel.x + 30, y))
            bar_w = int((pw - 110) * sp.count / total)
            pygame.draw.rect(surface, theme.ACCENT_DIM, (panel.x + 70, y + 3, bar_w, 12))
            surface.blit(font.render(str(sp.count), True, theme.TEXT_DIM),
                         (panel.x + pw - 36, y))

    def _selected(self):
        if self._selected_id is None or self.world.population is None:
            return None
        i = self.world.population.index_of(self._selected_id)
        return None if i is None else self.world.population.snapshot(i)

    def _draw_inspector(self, surface: pygame.Surface) -> None:
        entity = self._selected()
        if entity is None:
            return
        g = entity.genome
        ph = g.physiology
        in_ports = sum(1 for p in g.ports if p.mode == "in")
        out_ports = len(g.ports) - in_ports
        params = self.world.eco_params
        ceiling = max_hp(g, params, entity.age) if params else 1.0
        lines = [
            f"organism #{entity.id}   gen {entity.generation}   species {entity.species_id}",
            f"energy {entity.energy:.2f}   hp {entity.hp:.2f} / {ceiling:.2f} max   age {entity.age:,}",
            f"speed {entity.speed:.2f} / max {ph.max_speed:.2f}   size {ph.size:.2f}",
            f"metabolism {ph.metabolic_efficiency:.2f}   regen {ph.hp_regen_rate:.3f}"
            f"   senescence {ph.senescence_rate:.2f}",
            f"comfort {ph.comfort_center:.2f} ± {ph.comfort_width:.2f}   mut {ph.mutation_rate:.2f}",
            f"attack {ph.attack_power:.2f}   armor {ph.armor:.2f}",
            f"litter {ph.litter_size:.2f}   cooldown x{ph.repro_cooldown_mult:.2f}",
            "diet " + "  ".join(
                f"{n[:4]} {v:+.1f}" for n, v in zip(FOOD_NAMES, g.diet)
            ),
            f"brain: {g.node_count} nodes · {g.enabled_conn_count} conns · "
            f"ports {in_ports}in/{out_ports}out",
        ]
        font = self.app.fonts.get(14)
        pad, lh = 10, 18
        pw = max(font.size(s)[0] for s in lines) + pad * 2
        panel = pygame.Rect(12, _TOP_BAR + 12, pw, lh * len(lines) + pad * 2)
        box = pygame.Surface(panel.size, pygame.SRCALPHA)
        box.fill((*theme.PANEL, 235))
        surface.blit(box, panel.topleft)
        pygame.draw.rect(surface, theme.BORDER, panel, 1)
        for i, text in enumerate(lines):
            surface.blit(font.render(text, True, theme.TEXT), (panel.x + pad, panel.y + pad + i * lh))

        sx, sy = self.camera.world_to_screen(entity.x, entity.y)
        pygame.draw.circle(surface, theme.ACCENT, (sx, sy), 10, 2)

    def _draw_top_bar(self, surface: pygame.Surface) -> None:
        w = surface.get_width()
        bar = pygame.Rect(0, 0, w, _TOP_BAR)
        pygame.draw.rect(surface, theme.PANEL, bar)
        pygame.draw.line(surface, theme.BORDER, (0, _TOP_BAR), (w, _TOP_BAR))

        big = self.app.fonts.get(22, bold=True)
        small = self.app.fonts.get(15)

        surface.blit(big.render(self.world.name, True, theme.TEXT), (16, 8))
        surface.blit(
            small.render(
                f"{self.world.width}x{self.world.height}  ·  seed {self.world.seed}",
                True, theme.TEXT_DIM,
            ),
            (16, 34),
        )

        speed = self.clock.speed
        state = "PAUSED" if self.clock.paused else f"{speed.label}"
        state_col = theme.WARN if self.clock.paused else theme.OK
        tps = "-" if self.clock.paused else f"{self.clock.measured_tps:,.0f} TPS"

        stats = [
            (f"tick {self.world.tick:,}", theme.TEXT, big),
            (state, state_col, big),
            (tps, theme.TEXT_DIM, small),
            (f"{self.app.clock.get_fps():.0f} FPS", theme.TEXT_DIM, small),
        ]
        pop = self.world.population
        if pop is not None:
            count = len(pop)
            mean_energy = float(pop.energy.mean()) if count else 0.0
            label = "extinct" if not count else f"pop {count:,}"
            stats.insert(2, (label, theme.WARN if not count else theme.TEXT, big))
            n_species = len(self.world.species.species) if self.world.species else 0
            stats.insert(3, (
                f"E {mean_energy:.2f}   {n_species} sp   +{pop.births:,}/-{pop.deaths:,}",
                theme.TEXT_DIM, small,
            ))
        x = w - 16
        for text, col, font in reversed(stats):
            rendered = font.render(text, True, col)
            x -= rendered.get_width()
            surface.blit(rendered, (x, 18))
            x -= 22

        # tile under the cursor
        mx, my = pygame.mouse.get_pos()
        if my > _TOP_BAR:
            wx, wy = self.camera.screen_to_world(mx, my)
            tx, ty = int(wx), int(wy)
            if 0 <= tx < self.world.width and 0 <= ty < self.world.height:
                tile = self.world.grid.get(tx, ty)
                label = f"({tx}, {ty}) {TILE_NAMES.get(tile, '?')}"
                if self.world.ecosystem_enabled:
                    temp = self.world.temperature.at(tx, ty)
                    column = self.world.food.values[:, ty, tx]
                    top = int(column.argmax())
                    label += (
                        f"   {TemperatureField.to_celsius(temp):.0f}°C"
                        f"   food {float(column.sum()):.2f}"
                        f" ({FOOD_NAMES[top]} {float(column[top]):.2f})"
                    )
                surface.blit(small.render(label, True, theme.TEXT_DIM), (w // 2 - 120, 34))

    def _draw_hint_bar(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        text = ("Space: pause   +/-: speed   0: reset   WASD: pan   wheel: zoom   "
                "F: fit   F5: save   Esc: menu")
        if self.world.ecosystem_enabled:
            overlay = self.overlay.label or "off"
            food = "on" if self._show_food else "off"
            text += (f"   |   C: food ({food})   T/1-6: overlay ({overlay})   "
                     f"H: organisms   Tab: species   G: species lab   click: inspect")
        rendered = self.app.fonts.get(14).render(text, True, theme.TEXT_DIM)
        bar = pygame.Rect(0, h - 26, w, 26)
        pygame.draw.rect(surface, theme.PANEL, bar)
        pygame.draw.line(surface, theme.BORDER, (0, h - 26), (w, h - 26))
        surface.blit(rendered, (12, h - 22))

        if self._status_timer > 0:
            s = self.app.fonts.get(16, bold=True).render(self._status, True, theme.OK)
            surface.blit(s, (w // 2 - s.get_width() // 2, _TOP_BAR + 10))

    def _draw_menu(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        surface.blit(shade, (0, 0))

        title = self.app.fonts.get(30, bold=True).render("Paused", True, theme.TEXT)
        surface.blit(title, title.get_rect(center=(w // 2, h // 2 - 180)))

        for button in self._menu_buttons:
            button.draw(surface, self.app.fonts)
