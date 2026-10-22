"""Pruebas de las herramientas con un mercado simulado (no usan internet)."""

import json

import numpy as np
import pandas as pd
import pytest

from bot_mercados import mercado, tools
from bot_mercados.db import Database
from bot_mercados.planes import Plans
from bot_mercados.tools import ToolContext


def _df(closes, tz="Europe/Madrid"):
    index = pd.date_range("2025-01-01", periods=len(closes), freq="B", tz=tz)
    return pd.DataFrame({"Close": closes, "Volume": np.full(len(closes), 1000.0)}, index=index)


MARKET = {
    "AAA.MC": (_df(np.linspace(10, 11, 300)), {"currency": "EUR", "longName": "Empresa A, S.A.", "shortName": "EMPRESA A"}),
    "BBB": (_df(np.linspace(100, 80, 300), tz="America/New_York"), {"currency": "USD", "longName": "Company B Inc."}),
    "USDEUR=X": (_df(np.full(300, 0.9), tz="Europe/London"), {"currency": "EUR"}),
}


@pytest.fixture
def fake_market(monkeypatch):
    def history(ticker, period="1y", interval="1d"):
        return MARKET.get(ticker, (pd.DataFrame(), {}))
    monkeypatch.setattr(mercado, "history", history)
    monkeypatch.setattr(mercado, "info", lambda t: {"longName": "Empresa A, S.A.", "trailingPE": 12.0,
                                                     "profitMargins": 0.2, "dividendYield": 3.1,
                                                     "totalDebt": 500, "totalCash": 200, "ebitda": 100,
                                                     "targetMeanPrice": 13.2, "recommendationKey": "buy",
                                                     "trailingEps": 1.0, "forwardEps": 1.1, "sector": "Industrials"})
    monkeypatch.setattr(mercado, "calendar", lambda t: {})
    now = pd.Timestamp.now(tz="Europe/Madrid")
    divs = pd.Series([0.4, 0.2, 0.3], index=[now - pd.Timedelta(days=d) for d in (400, 200, 100)])
    monkeypatch.setattr(mercado, "dividends", lambda t: divs)
    monkeypatch.setattr(mercado, "last_prices", lambda ts: {t: {"precio": 11.0, "anterior": 10.0, "inicio": 10.0,
                                                                "fecha": "2026-01-01"} for t in ts})


@pytest.fixture
def ctx():
    db = Database(":memory:")
    db.ensure_user(1, 1, "Ana")
    return ToolContext(user_id=1, chat_id=1, db=db, plans=Plans(db))


def call(name, ctx=None, **args):
    return json.loads(tools.execute_tool(name, json.dumps(args), ctx or ToolContext(0, 0)))


def test_cotizacion(fake_market):
    out = call("cotizacion", tickers=["aaa.mc", "NOEXISTE"])["cotizaciones"]
    assert out[0]["ticker"] == "AAA.MC" and out[0]["precio"] == 11.0 and out[0]["moneda"] == "EUR"
    assert out[0]["vs_maximo_52s_pct"] == 0
    assert "error" in out[1]


def test_ficha_calcula_ratios_derivados(fake_market):
    out = call("ficha_empresa", ticker="AAA.MC")
    assert out["salud"]["deuda_neta"] == 300 and out["salud"]["deuda_neta_ebitda"] == 3.0
    assert out["analistas"]["recomendacion"] == "compra"
    assert out["analistas"]["potencial_pct"] == 20.0
    assert out["crecimiento"]["crecimiento_bpa_esperado_pct"] == 10.0
    # 0,2 + 0,3 cobrados en el último año sobre un precio de 11 (el pago de hace 400 días no cuenta)
    assert out["dividendo"]["rentabilidad_12m_pct"] == 4.55 and out["dividendo"]["pagos_12m"] == 2
    assert "ev_ebitda" not in out["valoracion"]  # los campos vacíos se eliminan
    assert out["tecnico"]["tendencia"] == "alcista"


def test_rendimiento_en_euros_convierte_divisa(fake_market):
    out = call("rendimiento", tickers=["BBB", "AAA.MC"], periodo="1y", en_euros=True)
    b = out["valores"][0]
    assert b["rentabilidad_pct"] == -20.0  # tipo de cambio constante: no altera la rentabilidad
    assert out["ranking"] == ["AAA.MC", "BBB"]


def test_cartera_y_concentracion(fake_market):
    out = call("analizar_cartera", posiciones=[{"ticker": "AAA.MC", "peso": 60}, {"ticker": "BBB", "peso": 40}],
               periodo="1y")
    assert out["por_divisa_pct"] == {"EUR": 60.0, "USD": 40.0}
    assert out["aviso_concentracion"].startswith("AAA.MC")


def test_cartera_valida_entrada(fake_market):
    assert "error" in call("analizar_cartera", posiciones=[{"ticker": "AAA.MC", "peso": 1}])
    assert "error" in call("analizar_cartera", posiciones=[{"ticker": "AAA.MC", "peso": -1}, {"ticker": "BBB", "peso": 1}])
    assert "error" in call("analizar_cartera", posiciones=[{"ticker": "AAA.MC", "peso": 1}, {"ticker": "BBB", "peso": 1}],
                           periodo="3y")


def test_simulacion(fake_market):
    out = call("simular_inversion", ticker="AAA.MC", aportacion_mensual=100, anos=1)
    assert out["moneda"] == "EUR" and out["valor_final"] > out["total_aportado"]
    assert "error" in call("simular_inversion", ticker="AAA.MC", anos=1)  # sin importes
    assert "error" in call("simular_inversion", ticker="AAA.MC", aportacion_mensual=10, anos=50)


def test_impuestos():
    out = call("calcular_impuestos", ganancias=15000)
    assert out["cuota"] == 3030 and "territorio común" in out["ambito"]


def test_grafico_genera_png(fake_market):
    ctx = ToolContext(0, 0)
    tools.execute_tool("grafico", json.dumps({"tickers": ["AAA.MC"], "periodo": "1y"}), ctx)
    tools.execute_tool("grafico", json.dumps({"tickers": ["AAA.MC", "BBB"], "periodo": "1y"}), ctx)
    assert len(ctx.images) == 2
    assert all(png.startswith(b"\x89PNG") for png, _ in ctx.images)


def test_alertas_respetan_el_plan(fake_market, ctx):
    for _ in range(3):
        assert call("alertas", ctx, accion="crear", ticker="AAA.MC", tipo="debajo", valor=5)["ok"]
    out = call("alertas", ctx, accion="crear", ticker="AAA.MC", tipo="debajo", valor=5)
    assert "Límite de 3 alertas" in out["error"]
    assert len(call("alertas", ctx, accion="ver")["alertas"]) == 3
    assert call("alertas", ctx, accion="borrar", id=1)["ok"]
    assert "error" in call("alertas", ctx, accion="borrar", id=1)


def test_alerta_avisa_si_ya_se_cumple(fake_market, ctx):
    out = call("alertas", ctx, accion="crear", ticker="AAA.MC", tipo="encima", valor=5)
    assert "aviso" in out


def test_alertas_no_disponibles_sin_cuenta(fake_market):
    assert "Telegram" in call("alertas", accion="ver")["error"]


def test_seguimiento(fake_market, ctx):
    assert call("seguimiento", ctx, accion="anadir", tickers=["AAA.MC", "NOEXISTE"]).get("error")
    assert call("seguimiento", ctx, accion="ver")["lista"][0]["ticker"] == "AAA.MC"
    assert call("seguimiento", ctx, accion="quitar", tickers=["AAA.MC"])["lista"] == []


def test_resumen_mercado_en_puntos_para_bonos(fake_market):
    out = call("resumen_mercado")
    bono = next(r for r in out["Riesgo"] if r["nombre"].startswith("Bono"))
    assert bono["dia_puntos"] == 1.0 and "dia_pct" not in bono


@pytest.mark.parametrize("bad", [[], ["A B"], ["X;rm -rf"], ["T"] * (tools.MAX_TICKERS + 1)])
def test_tickers_invalidos(bad):
    with pytest.raises(tools.ToolError):
        tools._clean_tickers(bad)


def test_execute_tool_errores_se_devuelven_como_json(fake_market):
    assert "desconocida" in call("borrar_todo")["error"]
    assert "JSON" in json.loads(tools.execute_tool("cotizacion", "no es json", ToolContext(0, 0)))["error"]
    assert "Argumentos incorrectos" in call("cotizacion", x=1)["error"]
    assert "Periodo" in call("rendimiento", tickers=["AAA.MC"], periodo="9y")["error"]


def test_noticias_limpia_nombre_y_filtra_paginas_de_cotizacion(fake_market, monkeypatch):
    queries = []

    def news(query, limit=6):
        queries.append(query)
        return [{"titular": "Precio de acciones, noticias, cotización e historial de X", "medio": "Yahoo", "fecha": "2026-01-01"},
                {"titular": "Empresa A gana más", "medio": "Expansión", "fecha": "2026-01-02", "enlace": "https://x"}]
    monkeypatch.setattr(mercado, "news", news)
    out = call("noticias", consulta="AAA.MC")
    assert queries[0] == '"Empresa A"'
    assert [n["titular"] for n in out["noticias"]] == ["Empresa A gana más"]
    assert "enlace" not in out["noticias"][0]


def test_esquemas_coinciden_con_funciones():
    names = {t["function"]["name"] for t in tools.TOOLS}
    assert names == set(tools.FUNCTIONS)


def test_esquemas_compactos():
    # Cada token cuenta contra el límite gratuito por minuto: vigila que no crezcan sin control.
    assert len(json.dumps(tools.TOOLS, ensure_ascii=False)) < 7000
