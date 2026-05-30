"""
Idealised-bus fan-out diagrams.

A small drawing API for the ZX rewrite sketched in the project notes: a single Z
spider with ``k`` outputs, where

* exactly one output is a plain *regular* edge straight off the Z spider, and
* each of the remaining ``k - 1`` outputs is routed through its own X spider, and
  every X spider fans out into ``m_i`` *idealised* edges.

The ``simplify`` rewrite replaces each fan of ``m_i`` parallel idealised edges by a
single **bold idealised bus** carrying the same multiplicity.

Typical use
-----------
>>> from idealised_bus import Fanout
>>> diag = Fanout([3, 5, 2])          # m1=3, m2=5, m3=2  ->  Z spider has 4 outputs
>>> diag.draw()                        # the left-hand (expanded) diagram
>>> diag.simplify().draw()             # the right-hand (bus) diagram

Specify the Z spider's output count directly with :meth:`Fanout.from_outputs`:

>>> Fanout.from_outputs(15)            # 15 outputs = 1 regular edge + 14 X spiders
>>> Fanout.from_outputs(4, [3, 5, 2])  # same as Fanout([3, 5, 2])

In a notebook, ``interactive_fanout([3, 5, 2])`` renders the diagram together with
a *Simplify* button that toggles between the two pictures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle

# --- visual style ----------------------------------------------------------
# Spider fills follow the standard ZX convention (green Z, red X) as used in
# Rodatz, Poor & Kissinger, "Fault Tolerance by Construction" (arXiv:2506.17181):
# pastel fills with a black outline.
PURPLE = "#C724E0"          # idealised edges / bus
BLACK = "#111111"           # regular edges and spider outlines
Z_FILL = "#d8f6d8"          # green  Z spider
X_FILL = "#e4a2a2"          # red    X spider

IDEAL_LW = 1.7              # a single idealised edge
BUS_LW = 7.0               # a bolded idealised bus
REG_LW = 2.0               # a regular edge

NODE_R = 0.15              # spider radius in data coordinates
MAX_SPOKES = 4             # idealised edges drawn before we switch to an ellipsis

# --- layout knobs ----------------------------------------------------------
# X spiders sit on an arc above Z that widens with their count; each spider's fan
# and regular leg splay outward along its own radial direction, so the picture
# stays legible from n = 2 up to large n.
_FAN_TILT = 16.0           # fan offset from the radial direction (degrees)
_REG_TILT = 26.0           # regular-leg offset from the radial direction (degrees)
_DOWN_LEN = 1.25           # length of the Z spider's regular output leg


def _unit(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([np.cos(a), np.sin(a)])


def _layout(nx: int) -> dict:
    """Arc angles and sizes for ``nx`` X spiders (mirrors the webpage)."""
    if nx <= 1:
        return dict(angles=np.array([90.0]), R=2.4, spread=0.0,
                    fan_len=1.25, reg_len=1.15, node_r=0.15)
    span = min(150.0, 56.0 + 22.0 * (nx - 1))
    R = max(2.4, 1.0 + 0.62 * nx)
    angles = np.linspace(90 + span / 2, 90 - span / 2, nx)
    return dict(angles=angles, R=R, spread=min(42.0, span / (nx - 1) * 0.72),
                fan_len=1.05 if nx > 8 else 1.25,
                reg_len=0.85 if nx > 8 else 1.15,
                node_r=0.13 if nx > 8 else 0.15)


@dataclass
class Fanout:
    """A Z-spider fan-out diagram with ``len(multiplicities)`` X spiders.

    Parameters
    ----------
    multiplicities:
        ``[m1, m2, ...]`` -- the number of idealised edges on each X spider. The Z
        spider then has ``len(multiplicities) + 1`` outputs (the ``+1`` being the
        plain regular output).
    labels:
        Optional symbolic labels for the fans (defaults to ``m_1, m_2, ...``).
    simplified:
        Whether fans are currently drawn as idealised buses.
    """

    multiplicities: list[int]
    labels: list[str] | None = None
    simplified: bool = False

    def __post_init__(self) -> None:
        self.multiplicities = [int(m) for m in self.multiplicities]
        if any(m < 1 for m in self.multiplicities):
            raise ValueError("every multiplicity m_i must be a natural number (>= 1)")
        if self.labels is None:
            self.labels = [f"$m_{{{i + 1}}}$" for i in range(len(self.multiplicities))]
        elif len(self.labels) != len(self.multiplicities):
            raise ValueError("labels and multiplicities must have the same length")

    # -- structure ----------------------------------------------------------
    @property
    def n_x_spiders(self) -> int:
        return len(self.multiplicities)

    @property
    def z_outputs(self) -> int:
        """Total outputs on the Z spider (one regular + one per X spider)."""
        return self.n_x_spiders + 1

    def simplify(self) -> "Fanout":
        """Collapse every fan of idealised edges into a bolded bus (in place)."""
        self.simplified = True
        return self

    def expand(self) -> "Fanout":
        """Inverse of :meth:`simplify` -- draw individual idealised edges again."""
        self.simplified = False
        return self

    @classmethod
    def from_outputs(cls, n: int, multiplicities: list[int] | None = None,
                     default: int = 1, labels: list[str] | None = None) -> "Fanout":
        """Build from the Z spider's output count ``n``.

        The Z spider gets ``n`` outputs: one plain regular edge plus ``n - 1`` X
        spiders (e.g. ``n = 15`` -> 14 X spiders). ``multiplicities`` (length
        ``n - 1``) sets each X spider's idealised-edge count; if omitted, every X
        spider gets ``default``.
        """
        if n < 1:
            raise ValueError("n (Z spider outputs) must be >= 1")
        if multiplicities is None:
            mults = [default] * (n - 1)
        else:
            mults = list(multiplicities)
            if len(mults) != n - 1:
                raise ValueError(
                    f"n={n} needs {n - 1} multiplicities, got {len(mults)}")
        return cls(mults, labels=labels)

    # -- drawing ------------------------------------------------------------
    def draw(self, ax: Axes | None = None, simplified: bool | None = None,
             title: str | None = None) -> Axes:
        """Render the diagram. ``simplified`` overrides ``self.simplified``."""
        simplified = self.simplified if simplified is None else simplified
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        lay = _layout(self.n_x_spiders)
        R, spread = lay["R"], lay["spread"]
        fan_len, reg_len, node_r = lay["fan_len"], lay["reg_len"], lay["node_r"]
        lw = 0.7 if self.n_x_spiders > 8 else 1.0   # thin strokes a touch when crowded

        z = np.array([0.0, 0.0])

        # Z spider's own regular output (straight down).
        z_leg = z + np.array([0.0, -_DOWN_LEN])
        ax.plot([z[0], z_leg[0]], [z[1], z_leg[1]], color=BLACK, lw=REG_LW * lw, zorder=1)

        for ang, m, label in zip(lay["angles"], self.multiplicities, self.labels):
            xp = z + R * _unit(ang)
            fan_ang = ang + _FAN_TILT          # fan splays outward, one side of radial
            reg_ang = ang - _REG_TILT          # regular leg splays out the other side
            fan_unit = _unit(fan_ang)

            # Z -> X spider connecting edge (idealised, purple).
            ax.plot([z[0], xp[0]], [z[1], xp[1]], color=PURPLE, lw=(IDEAL_LW + 0.4) * lw, zorder=1)

            # The X spider's plain regular output leg.
            reg_end = xp + reg_len * _unit(reg_ang)
            ax.plot([xp[0], reg_end[0]], [xp[1], reg_end[1]], color=BLACK, lw=REG_LW * lw, zorder=1)

            if simplified:
                self._draw_bus(ax, xp, fan_unit, fan_len, label, lw)
            else:
                self._draw_fan(ax, xp, m, fan_ang, spread, fan_len, label, lw)

            _spider(ax, xp, "X", node_r)

        _spider(ax, z, "Z", node_r)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.18)
        if title:
            ax.set_title(title)
        return ax

    def _draw_fan(self, ax: Axes, xp: np.ndarray, m: int, fan_ang: float,
                  spread: float, fan_len: float, label: str, lw: float) -> None:
        """Draw ``m`` parallel idealised edges as a braced fan of spokes."""
        spokes = min(m, MAX_SPOKES)
        for k in range(spokes):
            frac = 0.0 if spokes == 1 else (k / (spokes - 1) - 0.5)
            tip = xp + fan_len * _unit(fan_ang + spread * frac)
            ax.plot([xp[0], tip[0]], [xp[1], tip[1]], color=PURPLE, lw=IDEAL_LW * lw, zorder=1)

        # Ellipsis in the middle of the fan when we truncated the spokes.
        if m > spokes:
            mid = xp + 0.62 * fan_len * _unit(fan_ang)
            ax.text(*mid, r"$\cdots$", ha="center", va="center",
                    rotation=fan_ang - 90, fontsize=13, color=PURPLE, zorder=2)

        half = spread * 0.5 if spokes > 1 else 9.0
        a = xp + fan_len * _unit(fan_ang + half)
        b = xp + fan_len * _unit(fan_ang - half)
        _brace(ax, a, b, outward=_unit(fan_ang), label=label)

    def _draw_bus(self, ax: Axes, xp: np.ndarray, fan_unit: np.ndarray,
                  fan_len: float, label: str, lw: float) -> None:
        """Draw a single bolded idealised bus."""
        tip = xp + fan_len * fan_unit
        ax.plot([xp[0], tip[0]], [xp[1], tip[1]], color=PURPLE, lw=BUS_LW * lw,
                solid_capstyle="round", zorder=1)
        lbl = xp + (fan_len + 0.22) * fan_unit
        ax.text(*lbl, label, ha="center", va="center", fontsize=14, color=BLACK, zorder=2)


# --- low-level drawing helpers ---------------------------------------------
def _spider(ax: Axes, pos: np.ndarray, kind: str, r: float = NODE_R) -> None:
    """Z spider -> green node; X spider -> red node (standard ZX convention)."""
    face = Z_FILL if kind == "Z" else X_FILL
    ax.add_patch(Circle(pos, r, facecolor=face, edgecolor=BLACK,
                        linewidth=2.0, zorder=3))


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3 - 2 * u)


def _brace(ax: Axes, a: np.ndarray, b: np.ndarray, outward: np.ndarray,
           label: str, height: float = 0.24) -> None:
    """A curly brace spanning ``a``..``b`` and bulging along ``outward``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    # Upper-brace height profile in local coords (x along span, y along outward),
    # built from four smoothstep quarters meeting at a central tip.
    x = np.linspace(0.0, 1.0, 120)
    q = np.clip((x % 0.5) / 0.25, 0, 2)            # quarter coordinate, 0..2
    rise = np.where(q <= 1, 0.5 * _smoothstep(q), 0.5 + 0.5 * _smoothstep(q - 1))
    y = np.where(x <= 0.5, rise, rise[::-1])        # mirror the two halves

    seg = b - a
    pts = a + np.outer(x, seg) + np.outer(y * height, outward)
    ax.plot(pts[:, 0], pts[:, 1], color=BLACK, lw=1.3, zorder=2)

    peak = (a + b) / 2 + (height + 0.13) * outward
    ax.text(*peak, label, ha="center", va="center", fontsize=14,
            color=BLACK, zorder=2)


# --- convenience entry points ----------------------------------------------
def draw_rewrite(multiplicities: list[int], labels: list[str] | None = None,
                 figsize: tuple[float, float] = (12, 5)) -> plt.Figure:
    """Draw the full rewrite: expanded fans ``≡`` idealised buses, side by side."""
    diag = Fanout(multiplicities, labels=labels)
    fig, (left, mid, right) = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.12, 1]})
    diag.draw(left, simplified=False)
    diag.draw(right, simplified=True)
    mid.text(0.5, 0.5, r"$\equiv$", ha="center", va="center", fontsize=40)
    mid.axis("off")
    fig.suptitle(r"$m_1, \ldots, m_{%d} \in \mathbb{N}$" % len(multiplicities),
                 y=0.06, fontsize=14)
    fig.tight_layout()
    return fig


def interactive_fanout(multiplicities: list[int] = (3, 5, 2),
                       labels: list[str] | None = None):
    """Notebook widget: the diagram plus a *Simplify* button that toggles the bus."""
    import ipywidgets as widgets
    from IPython.display import display

    diag = Fanout(list(multiplicities), labels=labels)
    button = widgets.Button(description="Simplify", button_style="primary",
                            icon="compress")
    out = widgets.Output()

    def render() -> None:
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            diag.draw(ax)
            plt.show()
        button.description = "Expand" if diag.simplified else "Simplify"
        button.icon = "expand" if diag.simplified else "compress"

    def on_click(_):
        diag.simplified = not diag.simplified
        render()

    button.on_click(on_click)
    render()
    display(widgets.VBox([button, out]))
    return diag


if __name__ == "__main__":
    fig = draw_rewrite([3, 5, 2])
    fig.savefig("idealised_bus.png", dpi=130, bbox_inches="tight")
    print("wrote idealised_bus.png")
