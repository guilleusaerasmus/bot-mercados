"""Herramientas que el modelo puede llamar (tool calling).

Cada herramienta tiene dos partes:
  1. Un esquema JSON que se envía al modelo para que sepa qué hace y qué argumentos lleva.
  2. Una función de Python que se ejecuta de verdad cuando el modelo la pide.
El modelo nunca ejecuta código: solo pide "llama a X con estos argumentos" y nosotros devolvemos el resultado.

Los esquemas y las respuestas son compactos a propósito: cada token cuenta contra el límite
gratuito de tokens por minuto del proveedor.
"""

import inspect
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from . import analisis, graficos, mercado
from .analisis import num, pct
from .db import Database
from .documentos import DocumentError, SecClient
from .mercado import DataError
from .planes import TZ, Plans
from .rag import KnowledgeBase

log = logging.getLogger(__name__)

MAX_TICKERS = 6
MAX_POSITIONS = 10
PERIODS = {"1mo": "1 mes", "3mo": "3 meses", "6mo": "6 meses", "ytd": "en lo que va de año", "1y": "1 año",
           "2y": "2 años", "5y": "5 años", "10y": "10 años", "max": "todo el historial"}
ALERT_KINDS = ("encima", "debajo", "movimiento")
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_RECOMMENDATIONS = {"strong_buy": "compra fuerte", "buy": "compra", "hold": "mantener",
                    "underperform": "infraponderar", "sell": "venta", "strong_sell": "venta fuerte"}


class ToolError(Exception):
    """Error esperado (entrada inválida, límite del plan...) que se le explica al modelo."""


@dataclass
class ToolContext:
    """Lo que una herramienta necesita saber de quién pregunta (y dónde dejar los gráficos)."""
    user_id: int
    chat_id: int
    db: Database | None = None
    plans: Plans | None = None
    images: list[tuple[bytes, str]] = field(default_factory=list)
    kb: KnowledgeBase | None = None  # documentos para el RAG
    sec: SecClient | None = None  # informes anuales de la SEC (None si falta SEC_USER_AGENT)


def _prune(value: Any) -> Any:
    """Quita los campos vacíos: ahorra tokens y evita que el modelo comente datos inexistentes."""
    if isinstance(value, dict):
        cleaned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, {}, [], "")}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def valid_ticker(text: str) -> bool:
    return bool(_TICKER_RE.match(text))


def _clean_ticker(ticker) -> str:
    t = str(ticker).strip().upper()
    if not _TICKER_RE.match(t):
        raise ToolError(f"Ticker no válido: '{t}'.")
    return t


def _clean_tickers(tickers, limit: int = MAX_TICKERS) -> list[str]:
    if isinstance(tickers, str):
        tickers = [tickers]
    if not isinstance(tickers, list) or not tickers:
        raise ToolError("Indica al menos un ticker.")
    if len(tickers) > limit:
        raise ToolError(f"Máximo {limit} tickers por consulta.")
    return list(dict.fromkeys(_clean_ticker(t) for t in tickers))


def _period(periodo: str) -> str:
    if periodo not in PERIODS:
        raise ToolError(f"Periodo no válido. Usa uno de: {', '.join(PERIODS)}.")
    return periodo


def _closes(ticker: str, period: str = "1y") -> tuple[pd.Series, pd.DataFrame, dict]:
    df, meta = mercado.history(ticker, period)
    closes = df["Close"].dropna() if not df.empty else pd.Series(dtype=float)
    if len(closes) == 0:
        raise ToolError(f"Sin datos para {ticker} (¿ticker incorrecto? Usa buscar_ticker).")
    return closes, df, meta


def _dividends_last_12m(ticker: str) -> tuple[float | None, int]:
    """Dividendos por acción cobrados en los últimos 365 días (contados desde hoy) y número de pagos."""
    divs = mercado.dividends(ticker)
    if divs is None or len(divs) == 0:
        return None, 0
    cutoff = pd.Timestamp.now(tz=pd.DatetimeIndex(divs.index).tz) - pd.Timedelta(days=365)
    recent = divs[divs.index > cutoff]
    return (float(recent.sum()) if len(recent) else None), len(recent)


def _change_since(closes: pd.Series, start) -> float | None:
    before = closes[closes.index < start]
    return pct(closes.iloc[-1] / before.iloc[-1] - 1) if len(before) else None


# --- Herramientas -----------------------------------------------------------------------

def buscar_ticker(consulta: str) -> dict:
    consulta = str(consulta).strip()
    if not consulta or len(consulta) > 60:
        raise ToolError("La búsqueda debe tener entre 1 y 60 caracteres.")
    results = [
        {"ticker": q.get("symbol"), "nombre": q.get("longname") or q.get("shortname"),
         "mercado": q.get("exchDisp") or q.get("exchange"), "tipo": q.get("quoteType")}
        for q in mercado.search(consulta) if q.get("symbol")
    ]
    if not results:
        raise ToolError(f"No se ha encontrado ningún valor para '{consulta}'.")
    return {"resultados": results[:6]}


def cotizacion(tickers) -> dict:
    out = []
    for t in _clean_tickers(tickers):
        try:
            closes, _, meta = _closes(t, "1y")
        except ToolError as exc:
            out.append({"ticker": t, "error": str(exc)})
            continue
        last = closes.iloc[-1]
        idx = pd.DatetimeIndex(closes.index)
        out.append({
            "ticker": t,
            "nombre": meta.get("longName") or meta.get("shortName"),
            "precio": num(last, 4),
            "moneda": meta.get("currency"),
            "fecha": idx[-1].strftime("%Y-%m-%d"),
            "dia_pct": pct(last / closes.iloc[-2] - 1) if len(closes) > 1 else None,
            "1m_pct": _change_since(closes, idx[-1] - timedelta(days=30)),
            "ano_actual_pct": _change_since(closes, idx[-1].replace(month=1, day=1, hour=0, minute=0)),
            "1a_pct": pct(last / closes.iloc[0] - 1),
            "vs_maximo_52s_pct": pct(last / closes.max() - 1),
        })
    return _prune({"cotizaciones": out})


def ficha_empresa(ticker: str) -> dict:
    t = _clean_ticker(ticker)
    closes, df, meta = _closes(t, "1y")
    i = mercado.info(t)
    cal = mercado.calendar(t)
    price = closes.iloc[-1]

    net_debt = None
    if i.get("totalDebt") is not None and i.get("totalCash") is not None:
        net_debt = i["totalDebt"] - i["totalCash"]
    ebitda = i.get("ebitda")
    target = i.get("targetMeanPrice")
    eps, fwd_eps = i.get("trailingEps"), i.get("forwardEps")
    earnings = cal.get("Earnings Date") or []
    summary = (i.get("longBusinessSummary") or "")[:500]  # suficiente para incluir las marcas o segmentos
    ttm, payments = _dividends_last_12m(t)

    return _prune({
        "empresa": {"ticker": t, "nombre": i.get("longName") or meta.get("longName"), "tipo": i.get("quoteType"),
                    "sector": i.get("sector"), "industria": i.get("industry"), "pais": i.get("country"),
                    "empleados": i.get("fullTimeEmployees"), "resumen": summary},
        "precio": num(price, 4),
        "moneda": meta.get("currency"),
        "capitalizacion": num(i.get("marketCap"), 0),
        "valoracion": {"per": num(i.get("trailingPE")), "per_estimado": num(i.get("forwardPE")),
                       "peg": num(i.get("trailingPegRatio") or i.get("pegRatio")),
                       "precio_ventas": num(i.get("priceToSalesTrailing12Months")),
                       "precio_valor_contable": num(i.get("priceToBook")), "ev_ebitda": num(i.get("enterpriseToEbitda"))},
        "rentabilidad": {"margen_bruto_pct": pct(i.get("grossMargins")) if i.get("grossMargins") else None,
                         "margen_operativo_pct": pct(i.get("operatingMargins")),
                         "margen_neto_pct": pct(i.get("profitMargins")),
                         "roe_pct": pct(i.get("returnOnEquity")), "roa_pct": pct(i.get("returnOnAssets"))},
        "salud": {"deuda_neta": num(net_debt, 0),
                  "deuda_neta_ebitda": num(net_debt / ebitda) if net_debt is not None and ebitda else None,
                  "liquidez_corriente": num(i.get("currentRatio")),
                  "flujo_caja_libre": num(i.get("freeCashflow"), 0)},
        "crecimiento": {"ingresos_pct": pct(i.get("revenueGrowth")), "beneficios_pct": pct(i.get("earningsGrowth")),
                        "bpa": num(eps), "bpa_estimado": num(fwd_eps),
                        "crecimiento_bpa_esperado_pct": pct(fwd_eps / eps - 1) if eps and fwd_eps and eps > 0 else None},
        # Rentabilidad calculada con lo realmente pagado en 12 meses: el campo dividendYield de
        # Yahoo es incoherente entre empresas (p. ej. 1,1 % para Inditex cuando pagó un 3 %).
        "dividendo": {"rentabilidad_12m_pct": pct(ttm / price) if ttm else None,
                      "pagado_12m_por_accion": num(ttm, 4) if ttm else None, "pagos_12m": payments or None,
                      "payout_pct": pct(i.get("payoutRatio")),
                      "proximo_ex_dividendo": str(cal["Ex-Dividend Date"]) if cal.get("Ex-Dividend Date") else None},
        "analistas": {"recomendacion": _RECOMMENDATIONS.get(i.get("recommendationKey") or "", i.get("recommendationKey")),
                      "nota_1_compra_5_venta": num(i.get("recommendationMean")),
                      "numero": i.get("numberOfAnalystOpinions"), "precio_objetivo": num(target),
                      "objetivo_alto": num(i.get("targetHighPrice")), "objetivo_bajo": num(i.get("targetLowPrice")),
                      "potencial_pct": pct(target / price - 1) if target else None},
        "tecnico": analisis.technical_summary(closes, df.get("Volume")),
        "proximos_resultados": {"fecha": str(earnings[0]) if earnings else None,
                                "bpa_esperado": num(cal.get("Earnings Average"), 3)},
    })


def rendimiento(tickers, periodo: str = "1y", en_euros: bool = False) -> dict:
    periodo = _period(periodo)
    out = []
    for t in _clean_tickers(tickers):
        try:
            closes = mercado.closes_in_eur(t, periodo) if en_euros else _closes(t, periodo)[0]
        except (ToolError, DataError) as exc:
            out.append({"ticker": t, "error": str(exc)})
            continue
        out.append({"ticker": t, **analisis.performance(closes)})
    ranked = sorted((r for r in out if r.get("rentabilidad_pct") is not None), key=lambda r: -r["rentabilidad_pct"])
    result = {"periodo": PERIODS[periodo], "moneda": "EUR" if en_euros else "local de cada valor", "valores": out}
    if len(ranked) > 1:
        result["ranking"] = [r["ticker"] for r in ranked]
    return _prune(result)


_LEGAL_SUFFIX = re.compile(r"[,\s]+(S\.?A\.?U?\.?|S\.?L\.?|Inc\.?|Corp(oration)?\.?|PLC|N\.?V\.?|AG|SE|Ltd\.?|Co\.?)\s*$", re.I)
# Páginas de cotización que Google News mezcla con las noticias.
_QUOTE_PAGE = re.compile(r"^Precio de acciones|cotización e historial", re.I)


def noticias(consulta: str, *, incluir_enlaces: bool = False) -> dict:
    consulta = str(consulta).strip()
    if not consulta or len(consulta) > 60:
        raise ToolError("Indica un ticker o un nombre de empresa (máx. 60 caracteres).")
    query = consulta
    if _TICKER_RE.match(consulta) and consulta.upper() == consulta:
        _, meta = mercado.history(consulta, "5d")
        name = meta.get("longName") or meta.get("shortName")
        if name:
            query = _LEGAL_SUFFIX.sub("", name).strip() or name
    items = [n for n in mercado.news(f'"{query}"', limit=10) if not _QUOTE_PAGE.search(n["titular"])][:6]
    if not items:
        items = [n for n in mercado.news(query, limit=10) if not _QUOTE_PAGE.search(n["titular"])][:6]
    if not items:
        raise ToolError(f"No hay noticias recientes sobre '{query}'.")
    # Los enlaces solo hacen falta en /noticias: al modelo no le aportan nada y gastan tokens.
    return {"busqueda": query,
            "noticias": [{k: v for k, v in n.items() if incluir_enlaces or k != "enlace"} for n in items]}


MARKET = {
    "Índices": [("^IBEX", "IBEX 35"), ("^STOXX50E", "Euro Stoxx 50"), ("^GDAXI", "DAX"), ("^GSPC", "S&P 500"),
                ("^IXIC", "Nasdaq"), ("^N225", "Nikkei 225")],
    "Divisas": [("EURUSD=X", "EUR/USD"), ("EURGBP=X", "EUR/GBP")],
    "Materias primas": [("GC=F", "Oro (USD/onza)"), ("BZ=F", "Brent (USD/barril)")],
    "Cripto": [("BTC-EUR", "Bitcoin (EUR)")],
    "Riesgo": [("^VIX", "VIX (miedo)"), ("^TNX", "Bono EE. UU. 10 años (%)")],
}
IN_POINTS = {"^VIX", "^TNX"}


def resumen_mercado() -> dict:
    tickers = [t for group in MARKET.values() for t, _ in group]
    prices = mercado.last_prices(tickers)
    out = {}
    for group, items in MARKET.items():
        rows = []
        for t, label in items:
            p = prices.get(t)
            if not p:
                continue
            if t in IN_POINTS:  # para un tipo de interés o el VIX, la variación se expresa en puntos
                rows.append({"nombre": label, "valor": num(p["precio"], 2),
                             "dia_puntos": num(p["precio"] - p["anterior"]) if p.get("anterior") else None,
                             "semana_puntos": num(p["precio"] - p["inicio"]) if p.get("inicio") else None})
                continue
            rows.append({"nombre": label, "valor": num(p["precio"], 2),
                         "dia_pct": pct(p["precio"] / p["anterior"] - 1) if p.get("anterior") else None,
                         "semana_pct": pct(p["precio"] / p["inicio"] - 1) if p.get("inicio") else None})
        out[group] = rows
    if not any(out.values()):
        raise ToolError("La fuente de datos no responde ahora mismo.")
    dates = sorted(p["fecha"] for p in prices.values())
    return _prune({"fecha": dates[-1] if dates else None, **out})


PORTFOLIO_PERIODS = ("1y", "2y", "5y", "10y")


def analizar_cartera(posiciones: list, periodo: str = "5y") -> dict:
    if periodo not in PORTFOLIO_PERIODS:
        raise ToolError(f"Periodo no válido. Usa uno de: {', '.join(PORTFOLIO_PERIODS)}.")
    if not isinstance(posiciones, list) or not 2 <= len(posiciones) <= MAX_POSITIONS:
        raise ToolError(f"La cartera debe tener entre 2 y {MAX_POSITIONS} posiciones.")
    weights: dict[str, float] = {}
    for p in posiciones:
        if not isinstance(p, dict):
            raise ToolError("Cada posición debe ser {ticker, peso}.")
        t = _clean_ticker(p.get("ticker", ""))
        w = num(p.get("peso"), 6)
        if w is None or w <= 0:
            raise ToolError(f"Peso no válido para {t}: debe ser un número positivo (porcentaje o importe).")
        weights[t] = weights.get(t, 0) + w

    series, currencies, sectors = {}, {}, {}
    for t in weights:
        try:
            series[t] = mercado.closes_in_eur(t, periodo)
        except DataError as exc:
            raise ToolError(str(exc)) from exc
        meta = mercado.history(t, periodo)[1]
        currencies[meta.get("currency", "?")] = currencies.get(meta.get("currency", "?"), 0) + weights[t]
        sector = mercado.info(t).get("sector") or ("ETF/fondo" if mercado.info(t).get("quoteType") == "ETF" else "Otro")
        sectors[sector] = sectors.get(sector, 0) + weights[t]

    prices = analisis.align(series)
    if len(prices) < 60:
        raise ToolError("No hay suficiente historial común entre los valores para analizar la cartera.")
    total = sum(weights.values())
    result = analisis.portfolio(prices, weights)
    result["periodo"] = f"{prices.index[0]:%Y-%m-%d} a {prices.index[-1]:%Y-%m-%d} (en EUR, pesos constantes)"
    result["por_divisa_pct"] = {k: pct(v / total) for k, v in sorted(currencies.items(), key=lambda x: -x[1])}
    result["por_sector_pct"] = {k: pct(v / total) for k, v in sorted(sectors.items(), key=lambda x: -x[1])}
    return _prune(result)


def simular_inversion(ticker: str, aportacion_mensual: float = 0, importe_inicial: float = 0, anos: int = 10,
                      *, ctx: ToolContext | None = None) -> dict:
    t = _clean_ticker(ticker)
    monthly, initial = num(aportacion_mensual, 2) or 0, num(importe_inicial, 2) or 0
    if monthly < 0 or initial < 0 or (monthly == 0 and initial == 0):
        raise ToolError("Indica una aportación mensual y/o un importe inicial positivos.")
    try:
        years = int(anos)
    except (TypeError, ValueError) as exc:
        raise ToolError("Los años deben ser un número entero.") from exc
    if not 1 <= years <= 30:
        raise ToolError("Los años deben estar entre 1 y 30.")
    try:
        closes = mercado.closes_in_eur(t, "max")
    except DataError as exc:
        raise ToolError(str(exc)) from exc
    closes = analisis.align({"p": closes})["p"]
    start = closes.index[-1] - pd.DateOffset(years=years)
    window = closes[closes.index >= start]
    result = analisis.simulate_dca(window, initial, monthly)
    if "error" not in result:
        # Los modelos pequeños malinterpretan esta comparación: se la damos ya leída.
        lump, final = result["si_todo_al_principio"], result["valor_final"]
        better = "invertirlo todo al principio" if lump > final else "las aportaciones periódicas"
        result["lectura"] = (f"Con este activo y periodo habría dado más {better} ({max(lump, final):,.0f} € frente a "
                             f"{min(lump, final):,.0f} €). Invertir todo al principio suele ganar cuando el mercado sube; "
                             "aportar cada mes reduce el riesgo de entrar justo antes de una caída.").replace(",", ".")
    if window.index[0] > start + pd.Timedelta(days=31):
        result["aviso"] = f"Solo hay datos desde {window.index[0]:%Y-%m-%d}."
    if ctx is not None and "error" not in result:
        meta = mercado.history(t, "max")[1]
        name = meta.get("shortName") or meta.get("longName") or t
        def eur(v: float) -> str:
            return f"{v:,.0f} €".replace(",", ".")
        plan = " + ".join(x for x in (f"{eur(initial)} al inicio" if initial else "",
                                      f"{eur(monthly)}/mes" if monthly else "") if x)
        title = f"{plan} en {name}: {eur(result['valor_final'])}"
        tir = (f" · TIR {str(result['tir_anual_pct']).replace('.', ',')} % anual"
               if result.get("tir_anual_pct") is not None else "")
        subtitle = f"{result['desde']} a {result['hasta']} · aportado {eur(result['total_aportado'])}{tir} · dividendos reinvertidos"
        path, _ = analisis.dca_path(window, initial, monthly)
        ctx.images.append((graficos.dca_chart(title, path, subtitle), f"Simulación: {plan} en {t}"))
        result["grafico"] = "enviado como imagen"
    return {"ticker": t, "moneda": "EUR", "nota": "Precios ajustados: incluye dividendos reinvertidos, sin comisiones ni impuestos.",
            **result}


SCREEN_CRITERIA = {
    # criterio: (campo de ordenación de Yahoo, ascendente, condiciones extra)
    "dividendo": ("forward_dividend_yield", False, (("gt", "forward_dividend_yield", 0),)),
    "per_bajo": ("peratio.lasttwelvemonths", True, (("gt", "peratio.lasttwelvemonths", 0),)),
    "suben_hoy": ("percentchange", False, ()),
    "bajan_hoy": ("percentchange", True, ()),
    "suben_12m": ("fiftytwowkpercentchange", False, ()),
    "bajan_12m": ("fiftytwowkpercentchange", True, ()),
    "mas_grandes": ("intradaymarketcap", False, ()),
}
SCREEN_DISPLAY_ORDER = {  # criterio: (campo mostrado por el que se ordena, descendente)
    "dividendo": ("dividendo_pct", True), "per_bajo": ("per", False), "suben_hoy": ("dia_pct", True),
    "bajan_hoy": ("dia_pct", False), "suben_12m": ("12m_pct", True), "bajan_12m": ("12m_pct", False),
    "mas_grandes": ("capitalizacion_millones", True),
}
SECTORS = {"energia": "Energy", "electricas": "Utilities", "financiero": "Financial Services",
           "tecnologia": "Technology", "salud": "Healthcare", "industria": "Industrials",
           "consumo_ciclico": "Consumer Cyclical", "consumo_basico": "Consumer Defensive",
           "inmobiliario": "Real Estate", "materiales": "Basic Materials", "telecomunicaciones": "Communication Services"}


def buscar_acciones(criterio: str, bolsa: str = "es", sector: str | None = None, limite: int = 8) -> dict:
    if criterio not in SCREEN_CRITERIA:
        raise ToolError(f"criterio debe ser uno de: {', '.join(SCREEN_CRITERIA)}.")
    if bolsa not in mercado.SCREEN_MARKETS:
        raise ToolError("bolsa debe ser 'es' (Bolsa española) o 'us' (EE. UU.).")
    if sector is not None and sector not in SECTORS:
        raise ToolError(f"sector debe ser uno de: {', '.join(SECTORS)}.")
    limit = max(1, min(int(num(limite, 0) or 8), 15))
    field_name, ascending, extra = SCREEN_CRITERIA[criterio]
    quotes = mercado.screen(bolsa, field_name, ascending, SECTORS.get(sector) if sector else None, extra)
    rows = []
    for q in quotes:
        symbol = q.get("symbol") or ""
        # Latibex: empresas latinoamericanas que cotizan en Madrid (XCMIG.MC...). Se excluyen.
        if bolsa == "es" and symbol.startswith("X") and symbol.endswith(".MC"):
            continue
        rows.append({"ticker": symbol, "nombre": q.get("longName") or q.get("shortName"),
                     "precio": num(q.get("regularMarketPrice"), 4), "moneda": q.get("currency"),
                     "dia_pct": num(q.get("regularMarketChangePercent")),
                     "12m_pct": num(q.get("fiftyTwoWeekChangePercent")), "per": num(q.get("trailingPE"), 1),
                     "dividendo_pct": num(q.get("dividendYield")),
                     "capitalizacion_millones": num((q.get("marketCap") or 0) / 1e6, 0) or None})
    # Yahoo ordena por un campo (p. ej. dividendo futuro) y muestra otro (dividendo actual): se reordena por
    # el dato que ve el usuario para que la lista sea coherente.
    key, descending = SCREEN_DISPLAY_ORDER[criterio]
    rows = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r[key], reverse=descending)[:limit]
    if not rows:
        raise ToolError("Ningún valor cumple esos criterios.")
    for r in rows:
        if (r.get("dividendo_pct") or 0) > 10:
            r["aviso"] = "Dividendo anormalmente alto: puede incluir un pago extraordinario o una caída fuerte del precio."
    return _prune({"criterio": criterio, "bolsa": bolsa, "fecha": datetime.now(TZ).strftime("%Y-%m-%d"),
                   "resultados": rows,
                   "nota": "Datos en vivo de hoy. dividendo_pct es la estimación de Yahoo (orientativa): confírmala con ficha_empresa."})


# --- Cartera personal --------------------------------------------------------------------

def portfolio_view(db: Database, user_id: int) -> dict:
    """Valoración de la cartera del usuario en euros (también la usan /cartera y el resumen diario)."""
    lots = db.lots(user_id)
    if not lots:
        return {"posiciones": [], "nota": "Cartera vacía. Añade compras con /compra TICKER ACCIONES PRECIO."}
    grouped: dict[str, dict] = {}
    for lot in lots:
        g = grouped.setdefault(lot["ticker"], {"shares": 0.0, "cost": 0.0, "currency": lot["currency"]})
        g["shares"] += lot["shares"]
        g["cost"] += lot["shares"] * lot["price"]
    prices = mercado.last_prices(list(grouped))
    rates: dict[str, float] = {}
    positions, total_value, total_cost, today_change = [], 0.0, 0.0, 0.0
    for t, g in grouped.items():
        p = prices.get(t)
        if not p:
            positions.append({"ticker": t, "acciones": g["shares"], "error": "Sin precio ahora mismo."})
            continue
        if g["currency"] not in rates:
            rates[g["currency"]] = mercado.eur_per_unit(g["currency"])
        rate = rates[g["currency"]]
        value, cost = g["shares"] * p["precio"] * rate, g["cost"] * rate
        total_value, total_cost = total_value + value, total_cost + cost
        if p.get("anterior"):
            today_change += g["shares"] * (p["precio"] - p["anterior"]) * rate
        positions.append({"ticker": t, "acciones": num(g["shares"], 4), "coste_medio": num(g["cost"] / g["shares"], 4),
                          "precio": num(p["precio"], 4), "moneda": g["currency"], "valor_eur": num(value),
                          "ganancia_eur": num(value - cost), "ganancia_pct": pct(value / cost - 1) if cost else None,
                          "dia_pct": pct(p["precio"] / p["anterior"] - 1) if p.get("anterior") else None})
    for pos in positions:
        if total_value and pos.get("valor_eur") is not None:
            pos["peso_pct"] = pct(pos["valor_eur"] / total_value)
    positions.sort(key=lambda x: -(x.get("valor_eur") or 0))

    year = datetime.now(TZ).year
    realized = sum(s["gain_eur"] for s in db.sales_in_year(user_id, year))
    unrealized = total_value - total_cost
    result = {
        "posiciones": positions,
        "total": {"valor_eur": num(total_value), "invertido_eur": num(total_cost),
                  "ganancia_eur": num(unrealized), "ganancia_pct": pct(unrealized / total_cost) if total_cost else None,
                  "hoy_eur": num(today_change)},
        f"ventas_{year}": {"ganancia_realizada_eur": num(realized),
                           "irpf_estimado_eur": analisis.spanish_savings_tax(realized, 0)["cuota"] if realized > 0 else 0},
        "nota": "Importes en euros al cambio de hoy (aproximado en valores de otra divisa). Sin comisiones.",
    }
    if realized + unrealized > 0:
        result["irpf_si_vendieras_todo_hoy_eur"] = analisis.spanish_savings_tax(realized + unrealized, 0)["cuota"]
    return _prune(result)


def mi_cartera(accion: str, ticker: str = "", acciones: float | None = None, precio: float | None = None,
               *, ctx: ToolContext) -> dict:
    db, plans = _require_account(ctx)
    if accion == "ver":
        return portfolio_view(db, ctx.user_id)
    if accion not in ("comprar", "vender"):
        raise ToolError("accion debe ser 'ver', 'comprar' o 'vender'.")
    t = _clean_ticker(ticker)
    shares = num(acciones, 6)
    if shares is None or not 0 < shares <= 1e9:
        raise ToolError("Indica un número de acciones positivo.")
    closes, _, meta = _closes(t, "5d")
    price = num(precio, 6) if precio is not None else num(closes.iloc[-1], 6)
    if price is None or price <= 0:
        raise ToolError("El precio debe ser un número positivo (por acción, en la moneda del valor).")

    if accion == "comprar":
        held = db.portfolio_tickers(ctx.user_id)
        plan = plans.plan_for(ctx.user_id)
        if t not in held and len(held) >= plan.positions:
            raise ToolError(f"Límite de {plan.positions} valores en cartera del plan {plan.name}. Con /premium tienes más.")
        db.add_lot(ctx.user_id, t, shares, price, meta.get("currency") or "EUR")
        lots = db.lots(ctx.user_id, t)
        total = sum(lot["shares"] for lot in lots)
        avg = sum(lot["shares"] * lot["price"] for lot in lots) / total
        return {"ok": True, "compra": {"ticker": t, "acciones": shares, "precio": price, "moneda": meta.get("currency")},
                "posicion": {"acciones": num(total, 4), "coste_medio": num(avg, 4)}}

    lots = db.lots(ctx.user_id, t)
    held_shares = sum(lot["shares"] for lot in lots)
    if shares > held_shares + 1e-9:
        raise ToolError(f"Solo tienes {num(held_shares, 4)} acciones de {t}.")
    currency = lots[0]["currency"]
    rate = mercado.eur_per_unit(currency)
    gain = db.sell(ctx.user_id, t, shares, price, currency, rate)
    result = {"ok": True, "venta": {"ticker": t, "acciones": shares, "precio": price, "moneda": currency},
              "ganancia": num(gain), "ganancia_eur": num(gain * rate),
              "nota": "Calculado por FIFO (primero se venden las acciones más antiguas), como exige Hacienda."}
    if gain < 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=61)).isoformat()
        if any(lot["bought_at"] >= cutoff for lot in db.lots(ctx.user_id, t)):
            result["aviso_regla_dos_meses"] = ("Has comprado acciones de este valor en los últimos 2 meses y aún las tienes: "
                                               "Hacienda no te deja compensar esta pérdida hasta que las vendas.")
    return result


# --- RAG: búsqueda en documentos ------------------------------------------------------------

SEC_REFRESH_DAYS = 30  # cada cuánto se comprueba si la SEC tiene un informe anual más nuevo
_ingest_locks: dict[str, threading.Lock] = {}
_ingest_guard = threading.Lock()


def ensure_annual_report(kb: KnowledgeBase, sec: SecClient, ticker: str) -> int:
    """Id del último informe anual indexado de la empresa; si no está (o hay uno nuevo), lo descarga e indexa."""
    with _ingest_guard:
        lock = _ingest_locks.setdefault(ticker, threading.Lock())
    with lock:  # dos preguntas a la vez sobre la misma empresa no lo indexan dos veces
        current = kb.shared_document(ticker, "sec")
        if current:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(current["created_at"])
            if age < timedelta(days=SEC_REFRESH_DAYS) or sec.latest_annual_report(ticker)["url"] == current["url"]:
                return current["id"]
        meta, sections = sec.annual_report_sections(ticker)
        title = f"{meta['company']} — {meta['form']} {(meta['period'] or meta['filed'])[:4]}"
        return kb.add_document(title, sections, "sec", ticker=ticker, url=meta["url"], published=meta["filed"])


def buscar_en_documentos(pregunta: str, ticker: str | None = None, *, ctx: ToolContext) -> dict:
    if ctx.kb is None:
        raise ToolError("La búsqueda en documentos no está activada en este bot.")
    pregunta = str(pregunta).strip()
    if not 3 <= len(pregunta) <= 300:
        raise ToolError("La pregunta debe tener entre 3 y 300 caracteres.")
    doc_ids = [d["id"] for d in ctx.kb.documents(ctx.user_id)]
    notes = []
    if ticker:
        t = _clean_ticker(ticker)
        if "." in t or t.startswith("^"):
            notes.append(f"La SEC solo tiene informes de empresas de EE. UU.; para {t} el usuario puede enviarme el PDF.")
        elif ctx.sec is None:
            notes.append("Los informes de la SEC no están configurados (falta SEC_USER_AGENT).")
        else:
            try:
                doc_ids.append(ensure_annual_report(ctx.kb, ctx.sec, t))
            except DocumentError as exc:
                notes.append(str(exc))
    if not doc_ids:
        raise ToolError("No hay documentos donde buscar. " + " ".join(notes) +
                        " El usuario puede enviarme un PDF (informe anual, presentación de resultados, folleto de un fondo...).")
    hits = ctx.kb.search(pregunta, doc_ids, k=6)
    if not hits:
        raise ToolError("No he encontrado nada relacionado en los documentos.")
    return _prune({
        "busqueda": ctx.kb.mode,
        "fragmentos": [{"ref": i, "documento": h.title, "ubicacion": h.location, "texto": h.text}
                       for i, h in enumerate(hits, 1)],
        "instruccion": "Responde SOLO con estos fragmentos y cita cada dato con [ref]. Si no contienen la respuesta, dilo.",
        "notas": notes,
    })


def calcular_impuestos(ganancias: float = 0, dividendos_intereses: float = 0,
                       perdidas_anos_anteriores: float = 0, retenciones: float = 0) -> dict:
    values = [num(v, 2) for v in (ganancias, dividendos_intereses, perdidas_anos_anteriores, retenciones)]
    if any(v is None for v in values):
        raise ToolError("Todos los importes deben ser números.")
    result = analisis.spanish_savings_tax(*values)
    result["ambito"] = ("IRPF 2026, base del ahorro, territorio común (no País Vasco ni Navarra). "
                        "Estimación orientativa: no incluye deducciones por doble imposición internacional.")
    return result


def convertir_divisa(cantidad: float, origen: str, destino: str) -> dict:
    origen, destino = str(origen).strip().upper(), str(destino).strip().upper()
    if not (_CURRENCY_RE.match(origen) and _CURRENCY_RE.match(destino)):
        raise ToolError("Las divisas deben ser códigos ISO de 3 letras (EUR, USD...).")
    amount = num(cantidad, 6)
    if amount is None:
        raise ToolError("La cantidad debe ser un número.")
    if origen == destino:
        return {"cantidad": amount, "origen": origen, "destino": destino, "resultado": amount, "tipo": 1.0}
    rate = mercado.fx_rate(origen, destino)
    return {"cantidad": amount, "origen": origen, "destino": destino, "tipo": rate["tipo"],
            "resultado": round(amount * rate["tipo"], 2), "fecha": rate["fecha"], "fuente": "BCE"}


def grafico(tickers, periodo: str = "1y", *, ctx: ToolContext) -> dict:
    periodo = _period(periodo)
    tickers = _clean_tickers(tickers)
    if len(tickers) == 1:
        closes, df, meta = _closes(tickers[0], periodo)
        title = meta.get("longName") or meta.get("shortName") or tickers[0]
        long_closes = _closes(tickers[0], "5y")[0] if periodo in ("1mo", "3mo", "6mo", "ytd", "1y", "2y") else None
        png = graficos.price_chart(f"{title} ({tickers[0]})", df, meta.get("currency", ""), PERIODS[periodo], long_closes)
        caption = f"{tickers[0]} · {PERIODS[periodo]}"
        extra = {"variacion_pct": pct(closes.iloc[-1] / closes.iloc[0] - 1)}
    else:
        series = {t: _closes(t, periodo)[0] for t in tickers}
        png = graficos.compare_chart(series, PERIODS[periodo])
        caption = " vs ".join(tickers) + f" · {PERIODS[periodo]}"
        extra = {"variaciones_pct": {t: pct(s.iloc[-1] / s.iloc[0] - 1) for t, s in series.items()}}
    ctx.images.append((png, caption))
    return {"ok": True, "nota": "El gráfico se envía como imagen junto a tu respuesta; no lo describas en detalle.", **extra}


def _require_account(ctx: ToolContext) -> tuple[Database, Plans]:
    if ctx.db is None or ctx.plans is None:
        raise ToolError("Esta función solo está disponible en el bot de Telegram.")
    return ctx.db, ctx.plans


def alertas(accion: str, ticker: str = "", tipo: str = "", valor: float | None = None, id: int | None = None,
            *, ctx: ToolContext) -> dict:
    db, plans = _require_account(ctx)
    if accion == "ver":
        return {"alertas": [{"id": a["id"], "ticker": a["ticker"], "tipo": a["kind"], "valor": a["value"]}
                            for a in db.user_alerts(ctx.user_id)]}
    if accion == "borrar":
        alert_id = num(id, 0)
        if alert_id is None or not db.delete_alert(ctx.user_id, int(alert_id)):
            raise ToolError("No existe esa alerta. Usa accion='ver' para ver los id.")
        return {"ok": True, "borrada": int(alert_id)}
    if accion != "crear":
        raise ToolError("accion debe ser 'crear', 'ver' o 'borrar'.")

    t = _clean_ticker(ticker)
    if tipo not in ALERT_KINDS:
        raise ToolError(f"tipo debe ser uno de {ALERT_KINDS}.")
    value = num(valor, 6)
    if value is None or value <= 0:
        raise ToolError("valor debe ser un número positivo (precio, o % de movimiento diario).")
    plan = plans.plan_for(ctx.user_id)
    if len(db.user_alerts(ctx.user_id)) >= plan.alerts:
        raise ToolError(f"Límite de {plan.alerts} alertas del plan {plan.name}. Con /premium tienes más.")
    closes, _, meta = _closes(t, "5d")
    alert_id = db.add_alert(ctx.user_id, ctx.chat_id, t, tipo, value)
    price = closes.iloc[-1]
    result = {"ok": True, "id": alert_id, "ticker": t, "tipo": tipo, "valor": value,
              "precio_actual": num(price, 4), "moneda": meta.get("currency"),
              "nota": "Se revisa cada 10 minutos con datos que pueden ir 15 min retrasados."}
    if (tipo == "encima" and price >= value) or (tipo == "debajo" and price <= value):
        result["aviso"] = "La condición ya se cumple ahora: saltará en la próxima revisión."
    return result


def seguimiento(accion: str, tickers=None, *, ctx: ToolContext) -> dict:
    db, plans = _require_account(ctx)
    if accion in ("anadir", "quitar"):
        changed = []
        plan = plans.plan_for(ctx.user_id)
        for t in _clean_tickers(tickers or [], limit=MAX_POSITIONS):
            if accion == "anadir":
                if len(db.watchlist(ctx.user_id)) >= plan.watchlist:
                    raise ToolError(f"Límite de {plan.watchlist} valores del plan {plan.name}. Con /premium tienes más.")
                _closes(t, "5d")  # comprueba que el ticker existe
                if db.watch(ctx.user_id, t):
                    changed.append(t)
            elif db.unwatch(ctx.user_id, t):
                changed.append(t)
        return {"ok": True, "cambiados": changed, "lista": db.watchlist(ctx.user_id)}
    if accion != "ver":
        raise ToolError("accion debe ser 'anadir', 'quitar' o 'ver'.")
    tickers = db.watchlist(ctx.user_id)
    prices = mercado.last_prices(tickers)
    return {"lista": [{"ticker": t, "precio": num(p["precio"], 4),
                       "dia_pct": pct(p["precio"] / p["anterior"] - 1) if p.get("anterior") else None}
                      for t, p in prices.items()]}


# --- Esquemas para el modelo -------------------------------------------------------------

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


# Qué es un ticker (SAN.MC, AAPL...) ya lo explican las instrucciones del agente: no se repite en cada herramienta.
_TICKERS = {"type": "array", "items": {"type": "string"}}
_TICKER = {"type": "string"}
_PERIOD = {"type": "string", "enum": list(PERIODS)}

TOOLS = [
    _fn("buscar_ticker", "Busca el ticker de una empresa, índice o ETF por su nombre.",
        {"consulta": {"type": "string"}}, ["consulta"]),
    _fn("cotizacion", "Precio actual y variación del día, mes, año en curso y 12 meses.",
        {"tickers": _TICKERS}, ["tickers"]),
    _fn("ficha_empresa", "Análisis completo de una empresa: valoración (PER, EV/EBITDA), márgenes, ROE, deuda, "
        "crecimiento, dividendo, consenso y precio objetivo de analistas, señales técnicas (medias, RSI, MACD) y próximos resultados.",
        {"ticker": _TICKER}, ["ticker"]),
    _fn("rendimiento", "Rentabilidad, rentabilidad anualizada, volatilidad, Sharpe y máxima caída de uno o varios valores. "
        "Para comparar valores de distintas monedas usa en_euros=true.",
        {"tickers": _TICKERS, "periodo": _PERIOD, "en_euros": {"type": "boolean"}}, ["tickers", "periodo"]),
    _fn("noticias", "Titulares de prensa de los últimos 7 días sobre una empresa o tema.",
        {"consulta": {"type": "string", "description": "Ticker o nombre"}}, ["consulta"]),
    _fn("resumen_mercado", "Cómo van hoy los principales índices, divisas, oro, petróleo, bitcoin, VIX y bono de EE. UU.",
        {}, []),
    _fn("analizar_cartera", "Analiza una cartera en euros: rentabilidad, riesgo, Sharpe, máxima caída, correlaciones, "
        "beneficio de diversificar y concentración por divisa y sector.",
        {"posiciones": {"type": "array", "items": {"type": "object", "properties": {
            "ticker": {"type": "string"}, "peso": {"type": "number", "description": "% o importe"}},
            "required": ["ticker", "peso"]}},
         "periodo": {"type": "string", "enum": list(PORTFOLIO_PERIODS)}}, ["posiciones"]),
    _fn("simular_inversion", "Simula '¿y si hubiera invertido X al mes en...?': aportaciones periódicas con dividendos "
        "reinvertidos, en euros. Devuelve valor final, TIR y comparación con invertirlo todo al principio.",
        {"ticker": _TICKER, "aportacion_mensual": {"type": "number"}, "importe_inicial": {"type": "number"},
         "anos": {"type": "integer", "minimum": 1, "maximum": 30}}, ["ticker", "anos"]),
    _fn("calcular_impuestos", "Estima el IRPF español 2026 de la base del ahorro (venta de acciones/fondos, dividendos, "
        "intereses) con compensación de pérdidas.",
        {"ganancias": {"type": "number", "description": "Saldo neto de ventas del año (negativo si pérdidas)"},
         "dividendos_intereses": {"type": "number"}, "perdidas_anos_anteriores": {"type": "number"},
         "retenciones": {"type": "number", "description": "Retenciones ya practicadas"}}, []),
    _fn("convertir_divisa", "Convierte entre divisas con el tipo oficial del BCE.",
        {"cantidad": {"type": "number"}, "origen": {"type": "string"}, "destino": {"type": "string"}},
        ["cantidad", "origen", "destino"]),
    _fn("grafico", "Genera un gráfico: de precio con medias si es 1 ticker, o comparativo en base 100 si son varios.",
        {"tickers": _TICKERS, "periodo": _PERIOD}, ["tickers"]),
    _fn("alertas", "Crea, ve o borra alertas de precio del usuario. tipo: 'encima'/'debajo' de un precio, "
        "o 'movimiento' si la variación diaria supera valor %.",
        {"accion": {"type": "string", "enum": ["crear", "ver", "borrar"]}, "ticker": _TICKER,
         "tipo": {"type": "string", "enum": list(ALERT_KINDS)}, "valor": {"type": "number"}, "id": {"type": "integer"}},
        ["accion"]),
    _fn("seguimiento", "Gestiona la lista de seguimiento del usuario (la usa el resumen diario).",
        {"accion": {"type": "string", "enum": ["anadir", "quitar", "ver"]}, "tickers": _TICKERS}, ["accion"]),
    _fn("buscar_acciones", "Busca acciones que cumplan un criterio (más dividendo, PER más bajo, las que más suben o "
        "bajan hoy o en 12 meses, las más grandes) en la bolsa española o de EE. UU., opcionalmente por sector.",
        {"criterio": {"type": "string", "enum": list(SCREEN_CRITERIA)}, "bolsa": {"type": "string", "enum": ["es", "us"]},
         "sector": {"type": "string", "enum": list(SECTORS)}, "limite": {"type": "integer"}}, ["criterio"]),
    _fn("mi_cartera", "Cartera real del usuario: 'ver' (valor, ganancias en EUR, IRPF estimado), 'comprar' o 'vender' "
        "(FIFO) acciones. precio por acción en su moneda; si falta, el actual.",
        {"accion": {"type": "string", "enum": ["ver", "comprar", "vender"]}, "ticker": _TICKER,
         "acciones": {"type": "number"}, "precio": {"type": "number"}}, ["accion"]),
    _fn("buscar_en_documentos", "Busca en los documentos del usuario (PDFs que ha enviado) y, si das un ticker de EE. UU., "
        "en el último informe anual oficial (10-K de la SEC, en inglés). Úsala para riesgos, estrategia, negocio, "
        "segmentos o cualquier detalle que esté en informes. Para el 10-K, escribe la pregunta en inglés.",
        {"pregunta": {"type": "string"}, "ticker": _TICKER}, ["pregunta"]),
]

FUNCTIONS = {
    "buscar_ticker": buscar_ticker, "cotizacion": cotizacion, "ficha_empresa": ficha_empresa,
    "rendimiento": rendimiento, "noticias": noticias, "resumen_mercado": resumen_mercado,
    "analizar_cartera": analizar_cartera, "simular_inversion": simular_inversion,
    "calcular_impuestos": calcular_impuestos, "convertir_divisa": convertir_divisa, "grafico": grafico,
    "alertas": alertas, "seguimiento": seguimiento, "buscar_acciones": buscar_acciones, "mi_cartera": mi_cartera,
    "buscar_en_documentos": buscar_en_documentos,
}


def execute_tool(name: str, arguments: str, ctx: ToolContext) -> str:
    """Ejecuta la herramienta pedida por el modelo y devuelve el resultado como texto JSON.

    Los errores también se devuelven como JSON: así el modelo puede explicárselos al
    usuario o corregir los argumentos, en vez de romper la conversación.
    """
    func = FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"error": f"Herramienta desconocida: {name}"})
    try:
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError
    except ValueError:
        return json.dumps({"error": "Argumentos no válidos (se esperaba un objeto JSON)."})
    if "ctx" in inspect.signature(func).parameters:
        args.pop("ctx", None)
        call = lambda: func(**args, ctx=ctx)  # noqa: E731
        check = lambda: inspect.signature(func).bind(**args, ctx=ctx)  # noqa: E731
    else:
        call = lambda: func(**args)  # noqa: E731
        check = lambda: inspect.signature(func).bind(**args)  # noqa: E731
    try:
        check()
    except TypeError as exc:
        return json.dumps({"error": f"Argumentos incorrectos para {name}: {exc}"}, ensure_ascii=False)
    try:
        result = call()
    except (ToolError, DataError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception:
        log.exception("Fallo inesperado en la herramienta %s", name)
        return json.dumps({"error": "La fuente de datos no responde ahora mismo. Inténtalo más tarde."}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)
