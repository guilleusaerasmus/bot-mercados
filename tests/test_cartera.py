"""Pruebas de la cartera personal (FIFO, IRPF, regla de los dos meses), el buscador y la migración de la base de datos."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot_mercados import analisis, mercado, tools
from bot_mercados.db import Database
from bot_mercados.planes import FREE, Plans
from bot_mercados.tools import ToolContext


def test_fifo_vende_primero_lo_mas_antiguo():
    lots = [{"id": 1, "shares": 10, "price": 5.0}, {"id": 2, "shares": 10, "price": 8.0}]
    updates, gain = analisis.fifo_sell(lots, 15, 10.0)
    assert updates == [(1, 0), (2, 5)]
    assert gain == pytest.approx(10 * 5 + 5 * 2)  # 10 a 5 € y 5 a 8 €, vendidas a 10 €


def test_fifo_sin_acciones_suficientes():
    with pytest.raises(ValueError):
        analisis.fifo_sell([{"id": 1, "shares": 1, "price": 1.0}], 2, 1.0)


def test_dca_path_cuadra_con_la_simulacion():
    prices = pd.Series(np.linspace(10, 20, 400), index=pd.date_range("2020-01-01", periods=400, freq="D"))
    path, flows = analisis.dca_path(prices, 1000, 100)
    result = analisis.simulate_dca(prices, 1000, 100)
    assert path["aportado"].iloc[-1] == result["total_aportado"]
    assert path["valor"].iloc[-1] == pytest.approx(result["valor_final"], abs=0.01)
    assert len(flows) == result["aportaciones"]


def test_migracion_anade_columnas_sin_perder_datos(tmp_path):
    path = tmp_path / "viejo.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (user_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, name TEXT, "
                 "created_at TEXT NOT NULL, premium_until TEXT, referred_by INTEGER, "
                 "referral_rewarded INTEGER NOT NULL DEFAULT 0, quiz_ok INTEGER NOT NULL DEFAULT 0, "
                 "quiz_total INTEGER NOT NULL DEFAULT 0)")
    conn.execute("INSERT INTO users (user_id, chat_id, name, created_at) VALUES (1, 1, 'Ana', '2026-01-01')")
    conn.commit()
    conn.close()
    db = Database(path)
    assert db.get_user(1)["announcements"] == 1
    assert db.announcement_chats() == [1]
    Database(path)  # abrirla otra vez no debe fallar


# --- cartera con un mercado simulado ---

PRICES = {"SAN.MC": (11.0, 10.0, "EUR"), "AAPL": (200.0, 190.0, "USD")}


@pytest.fixture
def market(monkeypatch):
    def history(ticker, period="1y", interval="1d"):
        if ticker not in PRICES:
            return pd.DataFrame(), {}
        price, _, currency = PRICES[ticker]
        idx = pd.date_range("2026-01-01", periods=5, freq="B", tz="Europe/Madrid")
        return pd.DataFrame({"Close": [price] * 5}, index=idx), {"currency": currency, "longName": ticker}
    monkeypatch.setattr(mercado, "history", history)
    monkeypatch.setattr(mercado, "last_prices", lambda ts: {t: {"precio": PRICES[t][0], "anterior": PRICES[t][1],
                                                                "fecha": "2026-01-07"} for t in ts if t in PRICES})
    monkeypatch.setattr(mercado, "eur_per_unit", lambda cur: {"EUR": 1.0, "USD": 0.9}[cur])


@pytest.fixture
def ctx():
    db = Database(":memory:")
    db.ensure_user(1, 1, "Ana")
    return ToolContext(user_id=1, chat_id=1, db=db, plans=Plans(db))


def call(name, ctx, **args):
    return json.loads(tools.execute_tool(name, json.dumps(args), ctx))


def test_compra_y_valoracion_en_euros(market, ctx):
    assert call("mi_cartera", ctx, accion="comprar", ticker="SAN.MC", acciones=100, precio=10)["ok"]
    out = call("mi_cartera", ctx, accion="comprar", ticker="SAN.MC", acciones=100, precio=12)
    assert out["posicion"] == {"acciones": 200, "coste_medio": 11.0}
    call("mi_cartera", ctx, accion="comprar", ticker="AAPL", acciones=10, precio=150)

    view = call("mi_cartera", ctx, accion="ver")
    total = view["total"]
    assert total["valor_eur"] == pytest.approx(200 * 11 + 10 * 200 * 0.9)
    assert total["ganancia_eur"] == pytest.approx(0 + 10 * 50 * 0.9)
    assert total["hoy_eur"] == pytest.approx(200 * 1 + 10 * 10 * 0.9)
    weights = {p["ticker"]: p["peso_pct"] for p in view["posiciones"]}
    assert sum(weights.values()) == pytest.approx(100, abs=0.05)
    assert view["irpf_si_vendieras_todo_hoy_eur"] == pytest.approx(450 * 0.19)


def test_venta_fifo_e_irpf_del_ano(market, ctx):
    call("mi_cartera", ctx, accion="comprar", ticker="SAN.MC", acciones=100, precio=10)
    out = call("mi_cartera", ctx, accion="vender", ticker="SAN.MC", acciones=40, precio=12)
    assert out["ganancia_eur"] == 80.0 and "FIFO" in out["nota"]
    view = call("mi_cartera", ctx, accion="ver")
    ventas = next(v for k, v in view.items() if k.startswith("ventas_"))
    assert ventas["ganancia_realizada_eur"] == 80.0 and ventas["irpf_estimado_eur"] == pytest.approx(15.2)
    assert view["posiciones"][0]["acciones"] == 60


def test_no_se_vende_lo_que_no_se_tiene(market, ctx):
    call("mi_cartera", ctx, accion="comprar", ticker="SAN.MC", acciones=10, precio=10)
    assert "Solo tienes 10" in call("mi_cartera", ctx, accion="vender", ticker="SAN.MC", acciones=11)["error"]


def test_aviso_regla_de_los_dos_meses(market, ctx):
    ctx.db.add_lot(1, "SAN.MC", 10, 15.0, "EUR", datetime.now(timezone.utc) - timedelta(days=200))
    ctx.db.add_lot(1, "SAN.MC", 10, 14.0, "EUR")  # compra reciente que se queda en cartera
    out = call("mi_cartera", ctx, accion="vender", ticker="SAN.MC", acciones=10, precio=11)
    assert out["ganancia_eur"] == -40.0 and "aviso_regla_dos_meses" in out


def test_venta_con_perdida_sin_recompra_no_avisa(market, ctx):
    ctx.db.add_lot(1, "SAN.MC", 10, 15.0, "EUR", datetime.now(timezone.utc) - timedelta(days=200))
    out = call("mi_cartera", ctx, accion="vender", ticker="SAN.MC", acciones=10, precio=11)
    assert "aviso_regla_dos_meses" not in out


def test_limite_de_valores_en_cartera(market, ctx, monkeypatch):
    monkeypatch.setitem(PRICES, "OTRO", (1.0, 1.0, "EUR"))
    monkeypatch.setitem(PRICES, "T0", (1.0, 1.0, "EUR"))
    for i in range(FREE.positions):
        ctx.db.add_lot(1, f"T{i}", 1, 1.0, "EUR")
    assert "Límite" in call("mi_cartera", ctx, accion="comprar", ticker="OTRO", acciones=1)["error"]
    # ampliar una posición que ya tienes no cuenta como valor nuevo
    assert call("mi_cartera", ctx, accion="comprar", ticker="T0", acciones=1, precio=1)["ok"]


def test_destinatarios_resumen_incluye_cartera(ctx):
    ctx.db.add_premium_days(1, 5)
    ctx.db.add_lot(1, "SAN.MC", 1, 1.0, "EUR")
    assert [r["user_id"] for r in ctx.db.premium_watchers()] == [1]


# --- buscador ---

def test_buscador_excluye_latibex_y_respeta_limite(monkeypatch):
    calls = []

    def screen(market, field, asc, sector=None, extra=()):
        calls.append((market, field, asc, sector, extra))
        return [{"symbol": "XCMIG.MC", "shortName": "Latam"}] + [
            {"symbol": f"A{i}.MC", "longName": f"Empresa {i}", "regularMarketPrice": 10 + i, "currency": "EUR",
             "dividendYield": 5 - i * 0.1, "trailingPE": 10, "marketCap": 2e9} for i in range(20)]
    monkeypatch.setattr(mercado, "screen", screen)
    out = call("buscar_acciones", ToolContext(0, 0), criterio="dividendo", sector="electricas", limite=3)
    assert [r["ticker"] for r in out["resultados"]] == ["A0.MC", "A1.MC", "A2.MC"]
    assert out["resultados"][0]["capitalizacion_millones"] == 2000
    assert calls[0][:4] == ("es", "forward_dividend_yield", False, "Utilities")


def test_buscador_valida_parametros():
    assert "criterio" in call("buscar_acciones", ToolContext(0, 0), criterio="magia")["error"]
    assert "bolsa" in call("buscar_acciones", ToolContext(0, 0), criterio="dividendo", bolsa="jp")["error"]
    assert "sector" in call("buscar_acciones", ToolContext(0, 0), criterio="dividendo", sector="ovnis")["error"]


def test_simulador_genera_grafico(monkeypatch):
    closes = pd.Series(np.linspace(10, 20, 800), index=pd.date_range("2023-01-02", periods=800, freq="B", tz="UTC"))
    monkeypatch.setattr(mercado, "closes_in_eur", lambda t, p: closes)
    monkeypatch.setattr(mercado, "history", lambda t, p="1y", i="1d": (pd.DataFrame({"Close": closes}), {"shortName": "Indice"}))
    ctx = ToolContext(0, 0)
    out = json.loads(tools.execute_tool("simular_inversion", json.dumps({"ticker": "^X", "aportacion_mensual": 100, "anos": 2}), ctx))
    assert out["grafico"] == "enviado como imagen"
    png, caption = ctx.images[0]
    assert png.startswith(b"\x89PNG") and "100 €/mes" in caption
