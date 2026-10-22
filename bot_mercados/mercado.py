"""Acceso a datos de mercado (Yahoo Finance, BCE y Google News), con caché en memoria.

Es el único módulo que habla con las fuentes de datos: para pasar a un proveedor con
licencia comercial (ver docs/MONETIZACION.md) solo habría que reescribir este archivo.
"""

import email.utils
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from .analisis import align

USER_AGENT = "bot-mercados/0.2"


class DataError(Exception):
    """Fallo esperado al obtener datos (ticker inexistente, divisa no soportada...)."""


class _TTLCache:
    def __init__(self):
        self._data: dict = {}
        self._lock = threading.Lock()

    def get_or_set(self, key, ttl: float, loader):
        now = time.monotonic()
        with self._lock:
            hit = self._data.get(key)
            if hit and hit[0] > now:
                return hit[1]
        value = loader()
        with self._lock:
            self._data[key] = (now + ttl, value)
            if len(self._data) > 2000:  # evita que crezca sin límite
                for k in sorted(self._data, key=lambda k: self._data[k][0])[:500]:
                    del self._data[k]
        return value


_cache = _TTLCache()


def _http_get(url: str, timeout: float = 15) -> bytes:
    # Sin User-Agent propio, Cloudflare bloquea las peticiones de Python (error 1010).
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read()


def history(ticker: str, period: str = "1y", interval: str = "1d") -> tuple[pd.DataFrame, dict]:
    """Velas OHLCV (precios ajustados por dividendos y splits) y metadatos (moneda, nombre)."""
    def load():
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        meta = (t.history_metadata or {}) if not df.empty else {}
        return df, meta
    # Ojo: el DataFrame se comparte entre llamadas gracias a la caché; no modificarlo.
    return _cache.get_or_set(("hist", ticker, period, interval), 600, load)


def info(ticker: str) -> dict:
    return _cache.get_or_set(("info", ticker), 6 * 3600, lambda: yf.Ticker(ticker).info or {})


def calendar(ticker: str) -> dict:
    def load():
        try:
            return yf.Ticker(ticker).calendar or {}
        except Exception:  # los ETF e índices no tienen calendario
            return {}
    return _cache.get_or_set(("cal", ticker), 6 * 3600, load)


def dividends(ticker: str) -> pd.Series:
    return _cache.get_or_set(("div", ticker), 6 * 3600, lambda: yf.Ticker(ticker).dividends)


def search(query: str) -> list[dict]:
    def load():
        s = yf.Search(query, max_results=8, news_count=0, lists_count=0, recommended=0)
        return s.quotes
    return _cache.get_or_set(("search", query.lower()), 24 * 3600, load)


def last_prices(tickers: list[str]) -> dict[str, dict]:
    """Último precio y cierre anterior de varios valores en una sola descarga (para alertas y resúmenes)."""
    if not tickers:
        return {}
    df = yf.download(tickers, period="5d", interval="1d", progress=False, auto_adjust=True,
                     group_by="column", threads=True)
    if df is None or df.empty:
        return {}
    closes = df["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    out = {}
    for t in tickers:
        if t not in closes:
            continue
        s = closes[t].dropna()
        if len(s) >= 1:
            out[t] = {
                "precio": float(s.iloc[-1]),
                "anterior": float(s.iloc[-2]) if len(s) > 1 else None,
                "inicio": float(s.iloc[0]) if len(s) > 1 else None,  # cierre de hace ~5 sesiones
                "fecha": pd.Timestamp(s.index[-1]).strftime("%Y-%m-%d"),
            }
    return out


def closes_in_eur(ticker: str, period: str) -> pd.Series:
    """Cierres ajustados convertidos a euros con el tipo de cambio de cada día."""
    df, meta = history(ticker, period)
    if df.empty:
        raise DataError(f"Sin datos para {ticker}.")
    closes = df["Close"].dropna()
    currency = meta.get("currency") or "EUR"
    if currency in ("GBp", "GBX"):  # Londres cotiza en peniques
        closes, currency = closes / 100, "GBP"
    if currency == "EUR":
        return closes
    fx, _ = history(f"{currency}EUR=X", period)
    if fx.empty:
        raise DataError(f"No hay tipo de cambio {currency}/EUR para {ticker}.")
    both = align({"p": closes, "fx": fx["Close"]})
    return both["p"] * both["fx"]


def fx_rate(origin: str, target: str) -> dict:
    """Tipo de cambio oficial del BCE vía Frankfurter (gratis y sin clave)."""
    def load():
        url = f"https://api.frankfurter.dev/v1/latest?base={origin}&symbols={target}"
        try:
            data = json.loads(_http_get(url))
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 422):
                raise DataError(f"Divisa no soportada ({origin} o {target}).") from exc
            raise
        rate = data.get("rates", {}).get(target)
        if rate is None:
            raise DataError(f"Divisa no soportada: {target}.")
        return {"tipo": rate, "fecha": data.get("date")}
    return _cache.get_or_set(("fx", origin, target), 3600, load)


def eur_per_unit(currency: str) -> float:
    """Euros que vale una unidad de `currency` (los peniques de Londres, GBp, se pasan a libras)."""
    if currency == "EUR":
        return 1.0
    if currency in ("GBp", "GBX"):
        return fx_rate("GBP", "EUR")["tipo"] / 100
    return fx_rate(currency.upper(), "EUR")["tipo"]


# Mercados del buscador de acciones: región de Yahoo, bolsas admitidas y capitalización mínima.
SCREEN_MARKETS = {
    "es": {"region": "es", "exchanges": ["MCE"], "min_cap": 3e8},
    "us": {"region": "us", "exchanges": ["NMS", "NYQ", "NGM", "NCM"], "min_cap": 1e10},
}


def screen(market: str, sort_field: str, ascending: bool, sector: str | None = None,
           extra: tuple[tuple[str, str, float], ...] = (), size: int = 40) -> list[dict]:
    """Buscador de acciones de Yahoo, ya filtrado por mercado, tamaño mínimo y (opcional) sector.

    extra: condiciones adicionales como (operador, campo, valor), p. ej. ("gt", "peratio.lasttwelvemonths", 0).
    """
    def load():
        cfg = SCREEN_MARKETS[market]
        EQ = yf.EquityQuery
        conditions = [EQ("eq", ["region", cfg["region"]]), EQ("is-in", ["exchange", *cfg["exchanges"]]),
                      EQ("gte", ["intradaymarketcap", cfg["min_cap"]])]
        if sector:
            conditions.append(EQ("eq", ["sector", sector]))
        conditions += [EQ(op, [field, value]) for op, field, value in extra]
        result = yf.screen(EQ("and", conditions), sortField=sort_field, sortAsc=ascending, size=size)
        return result.get("quotes", [])
    return _cache.get_or_set(("screen", market, sort_field, ascending, sector, extra), 900, load)


def news(query: str, limit: int = 6) -> list[dict]:
    """Titulares de los últimos 7 días desde el RSS de Google News en español."""
    def load():
        q = urllib.parse.quote(f"{query} when:7d")
        raw = _http_get(f"https://news.google.com/rss/search?q={q}&hl=es&gl=ES&ceid=ES:es")
        items = []
        for item in ET.fromstring(raw).iter("item"):
            title = item.findtext("title") or ""
            source = item.findtext("source") or ""
            if source and title.endswith(f" - {source}"):
                title = title[: -len(source) - 3]
            published = item.findtext("pubDate")
            when = email.utils.parsedate_to_datetime(published) if published else None
            items.append({
                "titular": title.strip(),
                "medio": source,
                "fecha": when.astimezone(timezone.utc).strftime("%Y-%m-%d") if when else None,
                "enlace": item.findtext("link"),
            })
        items.sort(key=lambda x: x["fecha"] or "", reverse=True)
        return items
    return _cache.get_or_set(("news", query.lower()), 1800, load)[:limit]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
