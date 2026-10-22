"""Cálculos financieros puros (sin red): indicadores técnicos, riesgo, carteras, simulaciones e impuestos.

Todo recibe series de pandas y devuelve diccionarios con números redondeados, listos para
mandar al modelo en JSON. Al no depender de internet, se puede probar con datos inventados.
"""

import math
from datetime import date

import pandas as pd

TRADING_DAYS = 252


def num(value, digits=2):
    """Convierte números de pandas/numpy a float JSON; NaN, infinito o ausente -> None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, digits)


def pct(fraction, digits=2):
    """Fracción (0.134) -> porcentaje (13.4)."""
    f = num(fraction, 8)
    return None if f is None else round(f * 100, digits)


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# --- Indicadores técnicos -------------------------------------------------------------

def sma(closes: pd.Series, window: int) -> pd.Series:
    return closes.rolling(window).mean()


def rsi(closes: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Wilder: media exponencial de subidas frente a bajadas, de 0 a 100."""
    delta = closes.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gains / losses)


def macd(closes: pd.Series) -> tuple[pd.Series, pd.Series]:
    line = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    return line, line.ewm(span=9, adjust=False).mean()


def technical_summary(closes: pd.Series, volume: pd.Series | None = None) -> dict:
    """Señales técnicas básicas a partir de ~1 año de cierres diarios."""
    closes = closes.dropna()
    if len(closes) < 30:
        return {"error": "Historial insuficiente para el análisis técnico."}
    last = closes.iloc[-1]
    out: dict = {"precio": num(last, 4)}

    for window in (50, 200):
        if len(closes) >= window:
            avg = sma(closes, window).iloc[-1]
            out[f"media_{window}"] = num(avg, 4)
            out[f"vs_media_{window}_pct"] = pct(last / avg - 1)
    if "media_50" in out and "media_200" in out:
        out["tendencia"] = "alcista" if out["media_50"] > out["media_200"] else "bajista"
        diff = (sma(closes, 50) - sma(closes, 200)).dropna().iloc[-60:]
        sign = diff > 0
        changes = sign[sign != sign.shift(1)].iloc[1:]
        if len(changes):
            out["cruce_reciente"] = {
                "tipo": "cruce dorado (alcista)" if changes.iloc[-1] else "cruce de la muerte (bajista)",
                "fecha": _day(changes.index[-1]),
            }

    rsi_value = num(rsi(closes).iloc[-1], 1)
    out["rsi_14"] = rsi_value
    if rsi_value is not None:
        out["rsi_lectura"] = "sobrecompra (>70)" if rsi_value > 70 else "sobreventa (<30)" if rsi_value < 30 else "neutral"

    line, signal = macd(closes)
    out["macd"] = "por encima de su señal (impulso alcista)" if line.iloc[-1] > signal.iloc[-1] else "por debajo de su señal (impulso bajista)"

    year = closes.iloc[-TRADING_DAYS:]
    out["maximo_52s"] = num(year.max(), 4)
    out["minimo_52s"] = num(year.min(), 4)
    out["distancia_maximo_52s_pct"] = pct(last / year.max() - 1)
    out["minimo_1m"] = num(closes.iloc[-21:].min(), 4)
    out["maximo_1m"] = num(closes.iloc[-21:].max(), 4)
    out["volatilidad_3m_pct"] = pct(closes.iloc[-63:].pct_change().std() * math.sqrt(TRADING_DAYS))

    if volume is not None:
        volume = volume.dropna()
        if len(volume) >= 60 and volume.iloc[-60:].mean() > 0:
            out["volumen_vs_media_3m"] = num(volume.iloc[-1] / volume.iloc[-60:].mean(), 2)
    return out


# --- Riesgo y rentabilidad --------------------------------------------------------------

def max_drawdown(values: pd.Series) -> dict:
    """Mayor caída desde un máximo previo, con las fechas del máximo y del mínimo."""
    values = values.dropna()
    drawdown = values / values.cummax() - 1
    trough = drawdown.idxmin()
    peak = values.loc[:trough].idxmax()
    return {"caida_pct": pct(drawdown.min()), "desde": _day(peak), "hasta": _day(trough)}


def performance(values: pd.Series) -> dict:
    """Métricas de una serie de precios o de valor de cartera."""
    values = values.dropna()
    if len(values) < 2:
        return {"error": "Sin datos suficientes."}
    returns = values.pct_change().dropna()
    days = (values.index[-1] - values.index[0]).days
    total = values.iloc[-1] / values.iloc[0] - 1
    vol = returns.std() * math.sqrt(TRADING_DAYS)
    out = {
        "desde": _day(values.index[0]),
        "hasta": _day(values.index[-1]),
        "rentabilidad_pct": pct(total),
        "volatilidad_anual_pct": pct(vol),
        "maxima_caida": max_drawdown(values),
    }
    if days >= 365:
        out["rentabilidad_anualizada_pct"] = pct((1 + total) ** (365.25 / days) - 1)
    if vol > 0:
        # Sharpe simplificado (sin restar el tipo sin riesgo): rentabilidad media anual / volatilidad.
        out["ratio_sharpe"] = num(returns.mean() * TRADING_DAYS / vol, 2)
    return out


def align(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Une series de distintos mercados por fecha (sin hora ni zona horaria)."""
    frames = {}
    for name, s in series.items():
        s = s.dropna()
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        frames[name] = pd.Series(s.values, index=idx.normalize()).groupby(level=0).last()
    # Festivos distintos en cada bolsa: se arrastra el último precio conocido.
    return pd.DataFrame(frames).sort_index().ffill().dropna()


def portfolio(prices: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Cartera con pesos constantes (rebalanceo diario) a partir de precios ya alineados y en la misma moneda."""
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    returns = prices[w.index].pct_change().dropna()
    port_returns = (returns * w).sum(axis=1)
    values = (1 + port_returns).cumprod()
    values = pd.concat([pd.Series([1.0], index=[prices.index[0]]), values])

    assets = {}
    for t in w.index:
        p = performance(prices[t])
        assets[t] = {"peso_pct": pct(w[t]), "rentabilidad_pct": p.get("rentabilidad_pct"),
                     "volatilidad_anual_pct": p.get("volatilidad_anual_pct")}

    corr = returns.corr()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    pairs.sort(key=lambda x: -x[2])

    weighted_vol = sum(w[t] * returns[t].std() for t in w.index) * math.sqrt(TRADING_DAYS)
    port_vol = port_returns.std() * math.sqrt(TRADING_DAYS)
    out = {
        "cartera": performance(values),
        "activos": assets,
        # Cuánto riesgo "desaparece" al combinar activos que no se mueven a la vez.
        "reduccion_riesgo_por_diversificacion_pct": pct(1 - port_vol / weighted_vol) if weighted_vol > 0 else None,
    }
    if pairs:
        out["correlaciones_mas_altas"] = [{"par": f"{a}-{b}", "correlacion": num(c)} for a, b, c in pairs[:3]]
        out["correlacion_mas_baja"] = {"par": f"{pairs[-1][0]}-{pairs[-1][1]}", "correlacion": num(pairs[-1][2])}
    heaviest = w.idxmax()
    if w[heaviest] > 0.4:
        out["aviso_concentracion"] = f"{heaviest} pesa un {pct(w[heaviest])} % de la cartera."
    return out


# --- Simulación de aportaciones periódicas -------------------------------------------------

def xirr(cashflows: list[tuple[date, float]]) -> float | None:
    """TIR anual de flujos con fechas (aportaciones negativas, valor final positivo), por bisección."""
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    flows = [((d - t0).days / 365.25, amount) for d, amount in cashflows]

    def npv(rate):
        return sum(amount / (1 + rate) ** t for t, amount in flows)

    low, high = -0.99, 10.0
    if npv(low) * npv(high) > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def dca_path(prices: pd.Series, initial: float, monthly: float) -> tuple[pd.DataFrame, list[tuple[date, float]]]:
    """Evolución diaria de lo aportado y de lo que vale, y los flujos de cada aportación.

    Invierte `initial` el primer día y `monthly` el primer día bursátil de cada mes siguiente.
    """
    prices = prices.dropna()
    days = pd.Series(prices.index, index=prices.index)
    buy_days = days.groupby([prices.index.year, prices.index.month]).first().tolist()

    contributions = pd.Series(0.0, index=prices.index)
    units_bought = pd.Series(0.0, index=prices.index)
    flows: list[tuple[date, float]] = []
    for i, day in enumerate(buy_days):
        amount = (initial if initial > 0 else monthly) if i == 0 else monthly
        if amount <= 0:
            continue
        contributions[day] += amount
        units_bought[day] += amount / prices.loc[day]
        flows.append((pd.Timestamp(day).date(), -amount))
    path = pd.DataFrame({"aportado": contributions.cumsum(), "valor": units_bought.cumsum() * prices})
    return path, flows


def simulate_dca(prices: pd.Series, initial: float, monthly: float) -> dict:
    """Resultado de invertir periódicamente (ver dca_path).

    Con precios ajustados por dividendos, el resultado incluye los dividendos reinvertidos.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return {"error": "Sin datos suficientes."}
    path, flows = dca_path(prices, initial, monthly)
    invested = float(path["aportado"].iloc[-1])
    final_value = float(path["valor"].iloc[-1])
    flows.append((pd.Timestamp(prices.index[-1]).date(), final_value))
    irr = xirr(flows)
    lump_sum = invested * prices.iloc[-1] / prices.iloc[0]
    return {
        "desde": _day(prices.index[0]),
        "hasta": _day(prices.index[-1]),
        "aportaciones": len(flows) - 1,
        "total_aportado": num(invested),
        "valor_final": num(final_value),
        "ganancia": num(final_value - invested),
        "rentabilidad_total_pct": pct(final_value / invested - 1) if invested else None,
        "tir_anual_pct": pct(irr) if irr is not None else None,
        "si_todo_al_principio": num(lump_sum),
        "peor_caida_precio": max_drawdown(prices),
    }


# --- Cartera personal: ventas por FIFO ----------------------------------------------------

def fifo_sell(lots: list[dict], quantity: float, sell_price: float) -> tuple[list[tuple[int, float]], float]:
    """Vende `quantity` acciones consumiendo primero los lotes más antiguos (FIFO).

    Hacienda exige FIFO para valores homogéneos: las primeras acciones compradas son las
    primeras vendidas. `lots` va ordenado por fecha de compra, con id, shares y price.
    Devuelve [(id_lote, acciones_restantes)] para actualizar y la ganancia (en la moneda del valor).
    """
    remaining = quantity
    updates, gain = [], 0.0
    for lot in lots:
        if remaining <= 1e-12:
            break
        used = min(lot["shares"], remaining)
        gain += used * (sell_price - lot["price"])
        updates.append((lot["id"], lot["shares"] - used))
        remaining -= used
    if remaining > 1e-9:
        raise ValueError("No hay acciones suficientes para vender.")
    return updates, gain


# --- Impuestos (IRPF, España, territorio común) -----------------------------------------

# Escala de la base del ahorro vigente desde 2025 (Ley 7/2024), aplicable en 2026.
SAVINGS_BRACKETS = [(6_000, 0.19), (50_000, 0.21), (200_000, 0.23), (300_000, 0.27), (math.inf, 0.30)]


def savings_tax(base: float) -> tuple[float, list[dict]]:
    tax, lower, detail = 0.0, 0.0, []
    for upper, rate in SAVINGS_BRACKETS:
        if base <= lower:
            break
        part = min(base, upper) - lower
        tax += part * rate
        label = f"{lower:,.0f} - " + ("en adelante" if upper == math.inf else f"{upper:,.0f}")
        detail.append({"tramo": label.replace(",", "."),
                       "base": round(part, 2), "tipo_pct": rate * 100, "cuota": round(part * rate, 2)})
        lower = upper
    return round(tax, 2), detail


def spanish_savings_tax(gains: float, income: float, previous_losses: float = 0.0, withholdings: float = 0.0) -> dict:
    """Estima el IRPF de la base del ahorro.

    gains: saldo neto de ganancias y pérdidas patrimoniales del año (venta de acciones, fondos...).
    income: rendimientos del capital mobiliario (dividendos, intereses).
    previous_losses: pérdidas patrimoniales de los 4 años anteriores pendientes de compensar.
    withholdings: retenciones ya practicadas (19 % de dividendos e intereses en España).
    Un saldo negativo de un bloque compensa el positivo del otro hasta un 25 % de este.
    """
    g, r = float(gains), float(income)
    p = max(float(previous_losses), 0.0)
    cap_r = 0.25 * r if r > 0 else 0.0
    notes = []

    if g < 0 and r > 0:
        used = min(-g, cap_r)
        r, g, cap_r = r - used, g + used, cap_r - used
        if used:
            notes.append(f"Se compensan {used:,.2f} € de pérdidas con dividendos/intereses (límite del 25 %).")
    if r < 0 and g > 0:
        used = min(-r, 0.25 * g)
        g, r = g - used, r + used
        if used:
            notes.append(f"Se compensan {used:,.2f} € de rendimientos negativos con ganancias (límite del 25 %).")
    if p > 0 and g > 0:
        used = min(p, g)
        g, p = g - used, p - used
        notes.append(f"Se compensan {used:,.2f} € de pérdidas de años anteriores con ganancias.")
    if p > 0 and r > 0 and cap_r > 0:
        used = min(p, cap_r)
        r, p = r - used, p - used
        notes.append(f"Se compensan {used:,.2f} € de pérdidas de años anteriores con dividendos/intereses (25 %).")

    base = max(g, 0) + max(r, 0)
    tax, detail = savings_tax(base)
    pending = p + max(-g, 0)
    out = {
        "base_ahorro": round(base, 2),
        "cuota": tax,
        "tipo_efectivo_pct": round(tax / base * 100, 2) if base else 0.0,
        "detalle_tramos": detail,
        "retenciones": round(withholdings, 2),
        "resultado": round(tax - withholdings, 2),
        "resultado_texto": "a pagar" if tax - withholdings > 0 else "a devolver",
        "notas": notes,
    }
    if pending > 0:
        out["perdidas_pendientes_para_4_anos"] = round(pending, 2)
    if r < 0:
        out["rendimientos_negativos_pendientes"] = round(-r, 2)
    return out
