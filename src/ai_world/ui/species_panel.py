"""Full-screen "Species Lab" overlay: population history + trait evolution.

Toggled with ``G`` from :class:`ai_world.ui.screens.simulation.SimulationScreen`.
Reads only from ``world.species`` (see :class:`ai_world.world.species.SpeciesRegistry`):

* ``census``        -- ``(tick, {species_id: count})`` per census, for the stacked
  population-over-time chart and the per-species count sparklines;
* ``trait_history`` -- ``(tick, {species_id: mean-trait vector})`` index-aligned
  with ``census``, for the trait-drift chart and the trend arrows.

It is a plain component (like :class:`ai_world.ui.field_overlay.FieldOverlay`), not
a :class:`~ai_world.ui.screens.base.Screen` -- the simulation keeps running and
the charts update live behind the shaded backdrop.
"""
from __future__ import annotations

import numpy as np
import pygame

from ai_world.ui import theme
from ai_world.ui.entity_renderer import signature_rgb
from ai_world.world.genome import PHYS_BOUNDS, PHYS_FIELDS
from ai_world.world.species import TRAIT_CHANNELS, SpeciesRegistry
from ai_world.world.world import World

_MARGIN = 24
_MAX_SAMPLES = 900  # downsample the history to at most this many x-steps


class SpeciesPanel:
    def __init__(self) -> None:
        self.open = False
        self._selected: int | None = None
        self._trait_ix = 0
        self._rows: list[tuple[pygame.Rect, int]] = []

    def toggle(self) -> None:
        self.open = not self.open

    def close(self) -> None:
        self.open = False

    # --- input --------------------------------------------------------
    def handle_event(self, event: pygame.event.Event, surface: pygame.Surface) -> bool:
        """Return ``True`` when the event was consumed by the panel."""
        if not self.open:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_g):
                self.close()
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                self._trait_ix = (self._trait_ix - 1) % len(TRAIT_CHANNELS)
            elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                self._trait_ix = (self._trait_ix + 1) % len(TRAIT_CHANNELS)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, sid in self._rows:
                if rect.collidepoint(event.pos):
                    self._selected = sid
                    break
            return True
        return True  # swallow everything else while the lab is up

    # --- drawing -----------------------------------------------------
    def draw(self, surface: pygame.Surface, world: World, fonts) -> None:
        registry = world.species
        w, h = surface.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((*theme.BG, 232))
        surface.blit(shade, (0, 0))

        title = fonts.get(26, bold=True).render("Species Lab", True, theme.TEXT)
        surface.blit(title, (_MARGIN, _MARGIN - 6))

        if registry is None or not registry.census:
            msg = fonts.get(18).render(
                "no census yet — let an ecosystem world run for a while", True, theme.TEXT_DIM
            )
            surface.blit(msg, msg.get_rect(center=(w // 2, h // 2)))
            self._draw_footer(surface, fonts, h)
            return

        top = _MARGIN + 34
        bottom = h - _MARGIN - 22
        split = _MARGIN + int((w - 2 * _MARGIN) * 0.62)

        self._draw_population(surface, fonts, registry,
                              pygame.Rect(_MARGIN, top, split - _MARGIN - 16, bottom - top))

        right = pygame.Rect(split, top, w - _MARGIN - split, bottom - top)
        list_h = min(right.height // 2, 30 + 22 * max(1, len(registry.species)))
        self._draw_species_list(surface, fonts, registry,
                                pygame.Rect(right.x, right.y, right.width, list_h))
        self._draw_detail(surface, fonts, registry,
                          pygame.Rect(right.x, right.y + list_h + 12,
                                      right.width, right.bottom - right.y - list_h - 12))
        self._draw_footer(surface, fonts, h)

    def _draw_footer(self, surface: pygame.Surface, fonts, h: int) -> None:
        hint = fonts.get(14).render(
            "click a species · [ ] change trait · Esc / G close", True, theme.TEXT_DIM
        )
        surface.blit(hint, (_MARGIN, h - _MARGIN - 6))

    # --- population history ----------------------------------------
    def _draw_population(
        self, surface: pygame.Surface, fonts, registry: SpeciesRegistry, rect: pygame.Rect
    ) -> None:
        _panel(surface, rect)
        plot = rect.inflate(-24, -44)
        plot.move_ip(0, 10)

        ticks, sids, series = _stack_series(registry.census)
        if len(ticks) < 2:
            return
        totals = series.sum(axis=0)
        y_max = _nice_max(float(totals.max()))

        _grid(surface, fonts, plot, ticks[0], ticks[-1], y_max)
        xs = np.linspace(plot.left, plot.right, len(ticks))
        base = np.full(len(ticks), plot.bottom, dtype=float)
        order = sorted(range(series.shape[0]), key=lambda i: -series[i].sum())
        for i in order:
            top = base - series[i] / y_max * plot.height
            lower = list(zip(xs, base))
            upper = list(zip(xs, top))
            pygame.draw.polygon(surface, _color_for(sids[i], registry), lower + upper[::-1])
            pygame.draw.lines(surface, theme.BG, False, upper, 1)  # band separator
            base = top

        title = fonts.get(15, bold=True).render(
            f"population over time   ·   peak {int(totals.max()):,}   ·   "
            f"tick {ticks[0]:,}–{ticks[-1]:,}",
            True, theme.TEXT,
        )
        surface.blit(title, (rect.x + 12, rect.y + 8))

    # --- species list ---------------------------------------------
    def _draw_species_list(
        self, surface: pygame.Surface, fonts, registry: SpeciesRegistry, rect: pygame.Rect
    ) -> None:
        _panel(surface, rect)
        surface.blit(
            fonts.get(15, bold=True).render(
                f"{len(registry.species)} living species   ·   thr {registry.threshold:.1f}",
                True, theme.TEXT),
            (rect.x + 12, rect.y + 8),
        )
        self._rows = []
        font = fonts.get(14)
        rowh = 22
        counts = _latest_counts(registry.census)
        selected = self._resolve_selected(registry)
        for i, sp in enumerate(registry.sorted_species()):
            y = rect.y + 32 + i * rowh
            if y + rowh > rect.bottom:
                break
            row = pygame.Rect(rect.x + 6, y, rect.width - 12, rowh - 2)
            self._rows.append((row, sp.id))
            if sp.id == selected:
                pygame.draw.rect(surface, theme.PANEL_LIGHT, row, border_radius=4)
            pygame.draw.rect(surface, _color_for(sp.id, registry), (row.x + 4, y + 5, 12, 12))
            surface.blit(font.render(f"#{sp.id}", True, theme.TEXT_DIM), (row.x + 24, y + 3))
            spark = pygame.Rect(row.x + 66, y + 3, row.width - 66 - 44, rowh - 8)
            _sparkline(surface, spark, _species_series(registry.census, sp.id))
            surface.blit(font.render(f"{counts.get(sp.id, sp.count):,}", True, theme.TEXT),
                         (row.right - 40, y + 3))

    # --- selected-species detail --------------------------------
    def _draw_detail(
        self, surface: pygame.Surface, fonts, registry: SpeciesRegistry, rect: pygame.Rect
    ) -> None:
        _panel(surface, rect)
        sid = self._resolve_selected(registry)
        if sid is None:
            return
        sp = registry.species[sid]
        first, last = _trait_bounds(registry.trait_history, sid)

        font = fonts.get(14)
        x, y = rect.x + 12, rect.y + 8
        surface.blit(fonts.get(15, bold=True).render(f"species #{sid}", True, theme.TEXT), (x, y))
        y += 22
        lineage = f"born tick {sp.first_tick:,}"
        if sp.parent_id is not None:
            lineage += f"   ·   split from #{sp.parent_id}"
        surface.blit(font.render(lineage, True, theme.TEXT_DIM), (x, y))
        y += 22

        bar_w = rect.width - 24 - 150
        for k, name in enumerate(PHYS_FIELDS):
            lo, hi = PHYS_BOUNDS[name]
            cur = float(last[k]) if last is not None else _repr_value(sp, name)
            frac = (cur - lo) / (hi - lo) if hi > lo else 0.0
            row_y = y + k * 17
            surface.blit(font.render(name, True, theme.TEXT_DIM), (x, row_y))
            track = pygame.Rect(x + 132, row_y + 4, bar_w, 9)
            pygame.draw.rect(surface, theme.PANEL_LIGHT, track)
            pygame.draw.rect(surface, theme.ACCENT_DIM,
                             (track.x, track.y, int(track.width * np.clip(frac, 0, 1)), track.height))
            trend = _trend(first[k] if first is not None else None,
                           last[k] if last is not None else None)
            surface.blit(font.render(f"{cur:.2f} {trend}", True, theme.TEXT),
                         (track.right + 8, row_y))
        y += len(PHYS_FIELDS) * 17 + 6

        if last is not None:
            nodes, conns, ports = last[10], last[11], last[12]
            surface.blit(
                font.render(
                    f"brain  {nodes:.0f} nodes · {conns:.0f} conns · {ports:.1f} ports",
                    True, theme.TEXT_DIM),
                (x, y),
            )
        y += 24

        chart = pygame.Rect(x, y, rect.width - 24, rect.bottom - y - 12)
        if chart.height > 40:
            self._draw_trait_chart(surface, fonts, registry, sid, chart)

    def _draw_trait_chart(
        self, surface: pygame.Surface, fonts, registry: SpeciesRegistry, sid: int,
        rect: pygame.Rect,
    ) -> None:
        channel = self._trait_ix % len(TRAIT_CHANNELS)
        name = TRAIT_CHANNELS[channel]
        xs, ys = _trait_track(registry.trait_history, sid, channel)
        surface.blit(
            fonts.get(14, bold=True).render(f"{name} over time   [ / ]", True, theme.TEXT),
            (rect.x, rect.y),
        )
        plot = pygame.Rect(rect.x, rect.y + 20, rect.width, rect.height - 20)
        pygame.draw.rect(surface, theme.PANEL_LIGHT, plot, 1)
        if len(xs) < 2:
            return
        lo, hi = float(min(ys)), float(max(ys))
        span = hi - lo or 1.0
        px = np.interp(xs, (xs[0], xs[-1]), (plot.left + 2, plot.right - 2))
        py = plot.bottom - 2 - (np.asarray(ys) - lo) / span * (plot.height - 4)
        pygame.draw.lines(surface, theme.ACCENT, False, list(zip(px, py)), 2)
        f = fonts.get(12)
        surface.blit(f.render(f"{hi:.2f}", True, theme.TEXT_DIM), (plot.left + 3, plot.top + 1))
        surface.blit(f.render(f"{lo:.2f}", True, theme.TEXT_DIM),
                     (plot.left + 3, plot.bottom - 14))

    # --- helpers ---------------------------------------------------
    def _resolve_selected(self, registry: SpeciesRegistry) -> int | None:
        if self._selected in registry.species:
            return self._selected
        ordered = registry.sorted_species()
        return ordered[0].id if ordered else None


# ---------------------------------------------------------------------------
# module-level chart / series helpers
# ---------------------------------------------------------------------------
def _downsample(n: int) -> np.ndarray:
    if n <= _MAX_SAMPLES:
        return np.arange(n)
    return np.linspace(0, n - 1, _MAX_SAMPLES).round().astype(int)


def _stack_series(census: list) -> tuple[np.ndarray, list[int], np.ndarray]:
    """(ticks, species_ids, matrix[species, sample]) for the stacked area chart."""
    keep = _downsample(len(census))
    rows = [census[i] for i in keep]
    ticks = np.array([t for t, _ in rows])
    ids = sorted({sid for _, c in rows for sid in c})
    mat = np.zeros((len(ids), len(rows)), dtype=float)
    ix = {sid: k for k, sid in enumerate(ids)}
    for j, (_, counts) in enumerate(rows):
        for sid, n in counts.items():
            mat[ix[sid], j] = n
    return ticks, ids, mat


def _species_series(census: list, sid: int) -> np.ndarray:
    keep = _downsample(len(census))
    return np.array([census[i][1].get(sid, 0) for i in keep], dtype=float)


def _latest_counts(census: list) -> dict[int, int]:
    return dict(census[-1][1]) if census else {}


def _trait_track(history: list, sid: int, channel: int) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for tick, means in history:
        vec = means.get(sid)
        if vec is not None:
            xs.append(tick)
            ys.append(float(vec[channel]))
    return xs, ys


def _trait_bounds(history: list, sid: int):
    """Earliest and latest recorded mean-trait vector for ``sid`` (or None)."""
    first = last = None
    for _, means in history:
        vec = means.get(sid)
        if vec is None:
            continue
        if first is None:
            first = vec
        last = vec
    return first, last


def _repr_value(sp, name: str) -> float:
    return float(getattr(sp.representative.physiology, name))


def _trend(first, last) -> str:
    if first is None or last is None:
        return ""
    delta = float(last) - float(first)
    scale = abs(float(first)) or 1.0
    if delta > 0.02 * scale:
        return "▲"
    if delta < -0.02 * scale:
        return "▼"
    return "–"


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(v))
    for step in (1, 2, 2.5, 5, 10):
        if step * mag >= v:
            return step * mag
    return 10 * mag


def _color_for(sid: int, registry: SpeciesRegistry) -> tuple[int, int, int]:
    sp = registry.species.get(sid)
    if sp is not None:
        return signature_rgb(sp.representative.body_signature)
    rng = np.random.default_rng(sid)  # stable pseudo-colour for extinct species
    return tuple(int(70 + 150 * c) for c in rng.random(3))


def _panel(surface: pygame.Surface, rect: pygame.Rect) -> None:
    box = pygame.Surface(rect.size, pygame.SRCALPHA)
    box.fill((*theme.PANEL, 240))
    surface.blit(box, rect.topleft)
    pygame.draw.rect(surface, theme.BORDER, rect, 1)


def _grid(surface, fonts, plot: pygame.Rect, t0: int, t1: int, y_max: float) -> None:
    font = fonts.get(12)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(plot.bottom - frac * plot.height)
        pygame.draw.line(surface, theme.PANEL_LIGHT, (plot.left, y), (plot.right, y))
        label = font.render(f"{y_max * frac:,.0f}", True, theme.TEXT_DIM)
        surface.blit(label, (plot.left + 2, y + (2 if frac == 1.0 else -14)))
    surface.blit(font.render(f"{t0:,}", True, theme.TEXT_DIM), (plot.left, plot.bottom + 4))
    label = font.render(f"{t1:,}", True, theme.TEXT_DIM)
    surface.blit(label, (plot.right - label.get_width(), plot.bottom + 4))


def _sparkline(surface: pygame.Surface, rect: pygame.Rect, values: np.ndarray) -> None:
    if values.size < 2:
        return
    hi = float(values.max()) or 1.0
    px = np.linspace(rect.left, rect.right, values.size)
    py = rect.bottom - values / hi * rect.height
    pygame.draw.lines(surface, theme.TEXT_DIM, False, list(zip(px, py)), 1)
