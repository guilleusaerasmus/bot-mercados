"""Gráficos en PNG para Telegram.

Se usa la API orientada a objetos de matplotlib (Figure, sin pyplot): no guarda estado
global y funciona desde los hilos en los que el bot ejecuta las consultas.
"""

import io

import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure

from .analisis import sma

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
INK, MUTED, GRID = "#111827", "#6b7280", "#e5e7eb"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _dates(ax) -> None:
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _naive(s: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(s.index)
    return pd.Series(s.values, index=idx.tz_localize(None) if idx.tz is not None else idx)


def price_chart(title: str, df: pd.DataFrame, currency: str, period_label: str,
                long_closes: pd.Series | None = None) -> bytes:
    """Precio con medias de 50 y 200 sesiones y volumen debajo.

    long_closes: historial más largo para que las medias cubran todo el periodo dibujado.
    """
    closes = _naive(df["Close"].dropna())
    source = _naive(long_closes.dropna()) if long_closes is not None else closes
    fig = Figure(figsize=(9, 5.4))
    grid = fig.add_gridspec(4, 1, hspace=0.08)
    ax = fig.add_subplot(grid[:3, 0])
    _style(ax)
    ax.plot(closes.index, closes.values, color=COLORS[0], linewidth=1.8, label="Precio")
    for window, color in ((50, "#f59e0b"), (200, "#6b7280")):
        avg = sma(source, window)
        avg = avg[avg.index >= closes.index[0]].dropna()
        if len(avg):
            ax.plot(avg.index, avg.values, color=color, linewidth=1.1, label=f"Media {window}")
    last = closes.iloc[-1]
    ax.scatter([closes.index[-1]], [last], color=COLORS[0], zorder=3, s=18)
    ax.annotate(f"{last:,.2f}", (closes.index[-1], last), textcoords="offset points", xytext=(6, 0),
                va="center", fontsize=9, color=INK)
    change = last / closes.iloc[0] - 1
    ax.set_title(f"{title}  ·  {period_label}  ·  {change:+.1%}", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_ylabel(currency, color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.tick_params(labelbottom=False)

    vol_ax = fig.add_subplot(grid[3, 0], sharex=ax)
    _style(vol_ax)
    if "Volume" in df and df["Volume"].fillna(0).sum() > 0:
        volume = _naive(df["Volume"].fillna(0))
        vol_ax.bar(volume.index, volume.values, color="#93c5fd", width=1.0)
        vol_ax.set_ylabel("Volumen", color=MUTED, fontsize=8)
        vol_ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v:,.0f}")
    else:
        vol_ax.set_visible(False)
    _dates(vol_ax)
    fig.text(0.99, 0.005, "Datos: Yahoo Finance · No es asesoramiento", ha="right", fontsize=7, color=MUTED)
    return _png(fig)


def dca_chart(title: str, path: pd.DataFrame, subtitle: str) -> bytes:
    """Lo aportado frente a lo que vale: el gráfico del simulador, pensado para compartirse."""
    path = path.copy()
    path.index = _naive(path["valor"]).index
    fig = Figure(figsize=(9, 5))
    ax = fig.add_subplot()
    _style(ax)
    value, invested = path["valor"], path["aportado"]
    gains, losses = (value >= invested).values, (value < invested).values
    ax.fill_between(value.index, invested.values, value.values, where=gains, color="#bbf7d0", interpolate=True,
                    label="Ganancia" if gains.any() else None)
    ax.fill_between(value.index, invested.values, value.values, where=losses, color="#fecaca", interpolate=True,
                    label="Pérdida" if losses.any() else None)
    ax.plot(value.index, value.values, color=COLORS[0], linewidth=2, label="Lo que vale")
    ax.plot(invested.index, invested.values, color=MUTED, linewidth=1.4, linestyle="--", label="Lo que has puesto")
    for series, color in ((value, COLORS[0]), (invested, MUTED)):
        ax.annotate(f"{series.iloc[-1]:,.0f} €".replace(",", "."), (series.index[-1], series.iloc[-1]),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=10, color=color, fontweight="bold")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f} €".replace(",", "."))
    # Título en la figura y subtítulo en el eje: así matplotlib los coloca sin que se pisen.
    fig.suptitle(title, x=0.125, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _dates(ax)
    fig.text(0.99, 0.005, "Bot de Mercados (Telegram) · Rentabilidades pasadas no garantizan futuras",
             ha="right", fontsize=7, color=MUTED)
    return _png(fig)


def allocation_chart(weights: dict[str, float], total_label: str) -> bytes:
    """Donut con el peso de cada posición en la cartera."""
    items = sorted(weights.items(), key=lambda x: -x[1])
    if len(items) > 8:  # el resto se agrupa para que se lea bien
        items = items[:7] + [("Otros", sum(v for _, v in items[7:]))]
    fig = Figure(figsize=(6.4, 5))
    ax = fig.add_subplot()
    palette = COLORS + ["#64748b", "#ca8a04"]
    ax.pie([v for _, v in items], labels=[f"{k}\n" + f"{v:.1f} %".replace(".", ",") for k, v in items],
           colors=palette[:len(items)], startangle=90, counterclock=False,
           wedgeprops={"width": 0.38, "edgecolor": "white"}, textprops={"fontsize": 9, "color": INK})
    ax.text(0, 0, total_label, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_title("Reparto de tu cartera", fontsize=12, color=INK, fontweight="bold")
    return _png(fig)


def compare_chart(series: dict[str, pd.Series], period_label: str) -> bytes:
    """Varias series rebasadas a 100 para comparar su evolución."""
    fig = Figure(figsize=(9, 5))
    ax = fig.add_subplot()
    _style(ax)
    ranking = []
    for i, (label, s) in enumerate(series.items()):
        s = _naive(s.dropna())
        base100 = s / s.iloc[0] * 100
        color = COLORS[i % len(COLORS)]
        ax.plot(base100.index, base100.values, color=color, linewidth=1.8, label=f"{label} ({base100.iloc[-1] - 100:+.1f} %)")
        ranking.append((base100.iloc[-1], label))
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle="--")
    best = max(ranking)[1]
    ax.set_title(f"Comparativa en base 100  ·  {period_label}  ·  mejor: {best}", loc="left", fontsize=12,
                 color=INK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _dates(ax)
    fig.text(0.99, 0.005, "Datos: Yahoo Finance (precios ajustados) · No es asesoramiento", ha="right", fontsize=7, color=MUTED)
    return _png(fig)
