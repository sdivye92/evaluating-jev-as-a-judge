"""Inline-SVG chart builders (white surface, thin marks, rounded data ends, hover tooltips).

Colour identifies the model (Jev blue, Haiku orange, Sonnet aqua). The thinking arm of a model keeps its
hue and is drawn with a 45-degree hatched fill (hollow markers on dot charts), so mode is a second channel
and never colour alone.
"""

import html
import itertools
import math

FONT = "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e8e7e3", "#bdbcb6"
MINUS = "−"
# judge display name -> (hue, hatched)
STYLE = {
    "Jev 1.13": ("#2a78d6", False),
    "Haiku 5.5": ("#eb6834", False),
    "Haiku 5.5 + thinking": ("#eb6834", True),
    "Sonnet 5.5": ("#1baf7a", False),
    "Sonnet 5.5 + thinking": ("#1baf7a", True),
}
_uid = itertools.count(1)


def esc(s):
    return html.escape(str(s), quote=True)


def num(v, d=1):
    return f"{v:.{d}f}".replace("-", MINUS)


def _tint(hex_color, amount=0.38):
    c = hex_color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda v: round(255 - (255 - v) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


class Svg:
    def __init__(self, w, h, aria):
        self.w, self.h, self.parts, self.defs, self.aria = w, h, [], {}, aria
        self.uid = next(_uid)

    def add(self, s):
        self.parts.append(s)

    def fill(self, name):
        """Solid hue, or a registered hatch pattern for a thinking arm."""
        color, hatched = STYLE[name]
        if not hatched:
            return color
        pid = f"hatch{self.uid}-{color.lstrip('#')}"
        self.defs[pid] = (f'<pattern id="{pid}" width="6" height="6" patternUnits="userSpaceOnUse" '
                          f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{_tint(color)}"/>'
                          f'<rect width="3" height="6" fill="{color}"/></pattern>')
        return f"url(#{pid})"

    def text(self, x, y, s, size=12, fill=INK2, anchor="start", weight=400, rotate=None):
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                 f'font-weight="{weight}"{tr}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>')

    def legend(self, entries, x0, y, maxw):
        """entries: ('swatch', name) | ('line', label, colour, width). Wraps; returns the number of rows."""
        x, row = x0, 0
        for e in entries:
            label = e[1]
            width = 15 + 6.9 * len(label) + 20
            if x + width > maxw and x > x0:
                x, row = x0, row + 1
            yy = y + 20 * row
            if e[0] == "swatch":
                self.add(f'<rect x="{x:.1f}" y="{yy - 9:.1f}" width="11" height="11" rx="2" fill="{self.fill(e[1])}"/>')
            else:
                self.add(f'<line x1="{x:.1f}" y1="{yy - 4:.1f}" x2="{x + 14:.1f}" y2="{yy - 4:.1f}" stroke="{e[2]}" stroke-width="{e[3]}"/>')
            self.text(x + 18, yy, label, 12.5, INK2)
            x += width
        return row + 1

    def render(self):
        body = "\n".join(p for p in self.parts if p.strip())
        defs = f'<defs>{"".join(self.defs.values())}</defs>\n' if self.defs else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" role="img" '
                f'aria-label="{esc(self.aria)}" style="width:100%;height:auto;max-width:{self.w}px;'
                f'font-family:{FONT};background:#ffffff">\n{defs}<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>\n'
                f'{body}\n</svg>')


def _legend_rows(entries, maxw, x0=56):
    x, rows = x0, 1
    for e in entries:
        width = 15 + 6.9 * len(e[1]) + 20
        if x + width > maxw and x > x0:
            x, rows = x0, rows + 1
        x += width
    return rows


def _bar_up(x, y_top, w, y_base, fill, r=4):
    h = y_base - y_top
    if h <= r:
        return f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{w:.1f}" height="{max(h, 0.5):.1f}" fill="{fill}"/>'
    return (f'<path d="M{x:.1f},{y_base:.1f} L{x:.1f},{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} '
            f'L{x + w - r:.1f},{y_top:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} L{x + w:.1f},{y_base:.1f} Z" fill="{fill}"/>')


def grouped_bars(cats, series, vals, cis, ymax, step, fmt, aria, ylab, cat_sub=None, label_series=0,
                 ref_lines=(), group_marks=None, mark_label=None, w=760, h=340, ymin=0.0, tick_fmt=None):
    """cats x series grouped columns with optional 95% CI whiskers; labels only the story series.

    ``series`` is a list of judge names (see STYLE). ``vals[c][s]`` / ``cis[c][s]`` may be None for a missing bar.
    """
    entries = [("swatch", n) for n in series]
    entries += [("line", label, "#8a8984", 1) for _, label in ref_lines]
    if group_marks:
        entries.append(("line", mark_label or "baseline", INK, 2.5))
    rows = _legend_rows(entries, w - 14)
    ml, mr, mt, mb = 56, 14, 26 + 20 * rows, 52
    s = Svg(w, h + 20 * (rows - 1), aria)
    h = s.h
    pw, ph = w - ml - mr, h - mt - mb
    y = lambda v: mt + (ymax - v) / (ymax - ymin) * ph
    base = y(ymin)
    s.text(16, mt + ph / 2, ylab, 12, INK2, "middle", rotate=-90)
    tick = ymin
    while tick <= ymax + 1e-9:
        s.line(ml, y(tick), w - mr, y(tick), GRID if tick > ymin else AXIS)
        s.text(ml - 8, y(tick) + 4, (tick_fmt or fmt)(tick), 12, INK2, "end")
        tick += step
    for v, _ in ref_lines:
        s.line(ml, y(v), w - mr, y(v), "#8a8984", 1)
    ns, gap = len(series), 2
    gw = pw / len(cats)
    bw = min(24, (gw * 0.82 - gap * (ns - 1)) / ns)
    for ci_, cat in enumerate(cats):
        gx = ml + gw * ci_ + gw / 2
        total = ns * bw + (ns - 1) * gap
        x0 = gx - total / 2
        for si, name in enumerate(series):
            v = vals[ci_][si]
            if v is None:
                continue
            x = x0 + si * (bw + gap)
            ci = cis[ci_][si] if cis else None
            tip = f"{name} · {cat}: {fmt(v)}" + (f" (95% CI {fmt(ci[0])} to {fmt(ci[1])})" if ci else "")
            s.add(f'<g><title>{esc(tip)}</title>{_bar_up(x, y(v), bw, base, s.fill(name))}</g>')
            if ci:
                cx = x + bw / 2
                s.add(f'<line x1="{cx:.1f}" y1="{y(ci[0]):.1f}" x2="{cx:.1f}" y2="{y(ci[1]):.1f}" stroke="{INK}" stroke-width="1.25" opacity="0.75"/>')
                for yy in (y(ci[0]), y(ci[1])):
                    s.add(f'<line x1="{cx - 3:.1f}" y1="{yy:.1f}" x2="{cx + 3:.1f}" y2="{yy:.1f}" stroke="{INK}" stroke-width="1.25" opacity="0.75"/>')
            if si == label_series:
                top = y(ci[1] if ci else v)
                s.text(x + bw / 2, top - 6, fmt(v), 11.5, INK, "middle", 600)
        if group_marks and group_marks[ci_] is not None:
            yy = y(group_marks[ci_])
            s.add(f'<g><title>{esc((mark_label or "baseline") + ": " + fmt(group_marks[ci_]))}</title>'
                  f'<line x1="{x0 - 5:.1f}" y1="{yy:.1f}" x2="{x0 + total + 5:.1f}" y2="{yy:.1f}" stroke="{INK}" stroke-width="2.5"/></g>')
        s.text(gx, h - mb + 20, cat, 13, INK, "middle", 600)
        if cat_sub:
            s.text(gx, h - mb + 36, cat_sub[ci_], 11.5, INK2, "middle")
    s.legend(entries, ml, 22, w - 14)
    return s.render()


def forest(groups, xmin, xmax, step, xlabel, left_note, right_note, aria, w=800, lc=200, rc=150):
    """groups: [(title, [(label, judge_name, value, lo, hi)])]; shared x axis in points.

    Dot colour is the judge's hue; a hollow dot marks a thinking arm.
    """
    n_rows = sum(len(g[1]) for g in groups)
    row_h, grp_h = 25, 30
    mt, mb = 12, 70
    h = mt + len(groups) * grp_h + n_rows * row_h + mb
    s = Svg(w, h, aria)
    pl, pr = lc, w - rc
    x = lambda v: pl + (v - xmin) / (xmax - xmin) * (pr - pl)
    plot_top, plot_bot = mt, h - mb
    t = xmin
    while t <= xmax + 1e-9:
        s.line(x(t), plot_top, x(t), plot_bot, GRID)
        s.text(x(t), plot_bot + 18, num(t, 0) if t else "0", 12, INK2, "middle")
        t += step
    s.line(x(0), plot_top, x(0), plot_bot, INK, 1.25)
    y = mt
    for title, rows in groups:
        s.text(12, y + 20, title, 13, INK, "start", 600)
        y += grp_h
        for label, name, v, lo, hi in rows:
            color, hollow = STYLE[name]
            cy = y + row_h / 2
            s.text(lc - 12, cy + 4, label, 12.5, INK2, "end")
            tip = f"{title} {label}: {num(v)} points (95% CI {num(lo)} to {num(hi)})"
            dot = (f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="5.5" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>' if hollow
                   else f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            s.add(f'<g><title>{esc(tip)}</title>'
                  f'<line x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
                  f'{dot}</g>')
            s.text(w - rc + 14, cy + 4, f"{num(v)} [{num(lo)}, {num(hi)}]", 12, INK)
            y += row_h
    s.text(x(xmin), plot_bot + 40, left_note, 12, INK2, "start")
    s.text(x(xmax), plot_bot + 40, right_note, 12, INK2, "end")
    s.text((pl + pr) / 2, plot_bot + 58, xlabel, 12, INK2, "middle")
    return s.render()


def reliability(series_pts, aria, w=760, h=380, min_n=10):
    ml, mr, mt, mb = 56, 190, 20, 52
    xmin, xmax, ymin, ymax = 0.5, 1.0, 0.3, 1.0
    s = Svg(w, h, aria)
    pw, ph = w - ml - mr, h - mt - mb
    x = lambda v: ml + (v - xmin) / (xmax - xmin) * pw
    y = lambda v: mt + (ymax - v) / (ymax - ymin) * ph
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.line(ml, y(t), ml + pw, y(t), GRID if t > ymin else AXIS)
        s.text(ml - 8, y(t) + 4, f"{int(round(t * 100))}%", 12, INK2, "end")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.text(x(t), mt + ph + 18, f"{int(round(t * 100))}%", 12, INK2, "middle")
    s.text(ml + pw / 2, h - 10, "Stated confidence  max(p, 1 − p)", 12, INK2, "middle")
    s.text(16, mt + ph / 2, "Actual accuracy", 12, INK2, "middle", rotate=-90)
    s.add(f'<line x1="{x(0.5):.1f}" y1="{y(0.5):.1f}" x2="{x(1.0):.1f}" y2="{y(1.0):.1f}" stroke="#8a8984" stroke-width="1.25"/>')
    max_n = max(n for pts in series_pts.values() for _, _, n in pts)
    ly = mt + 24
    for name, pts in series_pts.items():
        color = STYLE[name][0]
        pts = [p for p in pts if p[2] >= min_n]
        if len(pts) > 1:
            d = " ".join(f"{x(c):.1f},{y(a):.1f}" for c, a, _ in pts)
            s.add(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for c, a, n in pts:
            r = 4 + 4.5 * math.sqrt(n / max_n)
            tip = f"{name}: confidence {c:.2f}, accuracy {a:.1%}, n = {n}"
            s.add(f'<g><title>{esc(tip)}</title><circle cx="{x(c):.1f}" cy="{y(a):.1f}" r="{r:.1f}" fill="{color}" stroke="#ffffff" stroke-width="2"/></g>')
        s.add(f'<circle cx="{w - mr + 22}" cy="{ly - 4}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        s.text(w - mr + 34, ly, name, 12.5, INK2)
        ly += 22
    s.add(f'<line x1="{w - mr + 13}" y1="{ly - 4}" x2="{w - mr + 31}" y2="{ly - 4}" stroke="#8a8984" stroke-width="1.25"/>')
    s.text(w - mr + 34, ly, "perfect calibration", 12.5, INK2)
    ly += 22
    s.text(w - mr + 14, ly + 14, "Marker area ∝ number of", 11.5, INK2)
    s.text(w - mr + 14, ly + 29, f"items (bins with n < {min_n}", 11.5, INK2)
    s.text(w - mr + 14, ly + 44, "are omitted).", 11.5, INK2)
    return s.render()


def hbars(cats, series, p50, p95, xmax, step, aria, w=800, label_series=0, unit="s", dp=2):
    """Per-item latency: bar = median, whisker = 95th percentile; only the story series is labeled."""
    entries = [("swatch", n) for n in series]
    rows = _legend_rows(entries, w - 24, 150)
    ml, mr, mt, mb = 150, 24, 26 + 20 * rows, 40
    ns, bh, gap, cat_gap = len(series), 16, 3, 24
    ph = len(cats) * (ns * bh + (ns - 1) * gap + cat_gap)
    h = mt + ph + mb
    s = Svg(w, h, aria)
    pw = w - ml - mr
    x = lambda v: ml + v / xmax * pw
    t = 0.0
    while t <= xmax + 1e-9:
        s.line(x(t), mt, x(t), mt + ph - cat_gap + 6, GRID if t else AXIS)
        s.text(x(t), mt + ph - cat_gap + 24, f"{t:g} {unit}", 12, INK2, "middle")
        t += step
    s.legend(entries, ml, 22, w - 24)
    y = mt
    for ci_, cat in enumerate(cats):
        s.text(ml - 14, y + (ns * bh + (ns - 1) * gap) / 2 + 4, cat, 13, INK, "end", 600)
        for si, name in enumerate(series):
            v, hi = p50[ci_][si], p95[ci_][si]
            cy = y + bh / 2
            r = 4
            wv = max(x(v) - ml, 1)
            path = (f'<path d="M{ml},{y} L{ml + wv - r:.1f},{y} Q{ml + wv:.1f},{y} {ml + wv:.1f},{y + r} '
                    f'L{ml + wv:.1f},{y + bh - r} Q{ml + wv:.1f},{y + bh} {ml + wv - r:.1f},{y + bh} L{ml},{y + bh} Z" fill="{s.fill(name)}"/>')
            tip = f"{name} · {cat}: median {v:.{dp}f} {unit}, 95th percentile {hi:.{dp}f} {unit}"
            s.add(f'<g><title>{esc(tip)}</title>{path}'
                  f'<line x1="{x(v):.1f}" y1="{cy}" x2="{x(hi):.1f}" y2="{cy}" stroke="{INK}" stroke-width="1.25" opacity="0.75"/>'
                  f'<line x1="{x(hi):.1f}" y1="{cy - 4}" x2="{x(hi):.1f}" y2="{cy + 4}" stroke="{INK}" stroke-width="1.25" opacity="0.75"/></g>')
            if si == label_series:
                s.text(x(hi) + 8, cy + 4, f"{v:.{dp}f} {unit} (p95 {hi:.{dp}f} {unit})", 11.5, INK, "start", 600)
            y += bh + gap
        y += cat_gap - gap
    s.text(ml + pw, h - 6, "Bar: median · whisker: 95th percentile", 11.5, INK2, "end")
    return s.render()
