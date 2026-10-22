"""Pruebas de los cálculos financieros con series sintéticas cuyo resultado se conoce."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot_mercados import analisis


def series(values, start="2020-01-01", freq="B"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


def test_num_y_pct_limpian_valores_raros():
    assert analisis.num(float("nan")) is None
    assert analisis.num(float("inf")) is None
    assert analisis.num("abc") is None
    assert analisis.num(np.float64(1.23456), 3) == 1.235
    assert analisis.pct(0.1234) == 12.34


def test_rsi_extremos():
    up = series(np.arange(1, 60))
    down = series(np.arange(60, 1, -1))
    assert analisis.rsi(up).iloc[-1] == pytest.approx(100)
    assert analisis.rsi(down).iloc[-1] == pytest.approx(0)


def test_technical_summary_tendencia_alcista():
    closes = series(np.linspace(50, 150, 300))
    out = analisis.technical_summary(closes, series(np.full(300, 1000.0)))
    assert out["tendencia"] == "alcista"
    assert out["rsi_lectura"] == "sobrecompra (>70)"
    assert out["distancia_maximo_52s_pct"] == 0
    assert out["volumen_vs_media_3m"] == 1.0


def test_technical_summary_detecta_cruce():
    values = np.concatenate([np.linspace(100, 60, 250), np.linspace(60, 120, 80)])
    out = analisis.technical_summary(series(values))
    assert out["cruce_reciente"]["tipo"].startswith("cruce dorado")


def test_technical_summary_historial_corto():
    assert "error" in analisis.technical_summary(series([1.0] * 10))


def test_performance_rentabilidad_anualizada():
    closes = pd.Series([100.0, 200.0], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    out = analisis.performance(closes)
    assert out["rentabilidad_pct"] == 100.0
    assert out["rentabilidad_anualizada_pct"] == pytest.approx(41.4, abs=0.1)  # sqrt(2) - 1


def test_max_drawdown_fechas():
    closes = series([100, 120, 60, 90, 130])
    dd = analisis.max_drawdown(closes)
    assert dd["caida_pct"] == -50.0
    assert dd["desde"] == "2020-01-02" and dd["hasta"] == "2020-01-03"


def test_align_une_zonas_horarias_y_festivos():
    madrid = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-02 09:00", periods=3, freq="D", tz="Europe/Madrid"))
    ny = pd.Series([10.0, 30.0], index=pd.to_datetime(["2024-01-02 09:30", "2024-01-04 09:30"]).tz_localize("America/New_York"))
    out = analisis.align({"m": madrid, "n": ny})
    assert list(out["n"]) == [10.0, 10.0, 30.0]  # el festivo de NY arrastra el último precio
    assert out.index.tz is None


def test_cartera_diversificacion():
    rng = np.random.default_rng(1)
    a = series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    b = series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    prices = pd.DataFrame({"A": a, "B": b})
    out = analisis.portfolio(prices, {"A": 3, "B": 1})
    assert out["activos"]["A"]["peso_pct"] == 75.0
    assert out["reduccion_riesgo_por_diversificacion_pct"] > 10  # activos independientes
    assert out["aviso_concentracion"].startswith("A pesa")
    same = analisis.portfolio(pd.DataFrame({"A": a, "B": a}), {"A": 1, "B": 1})
    assert same["reduccion_riesgo_por_diversificacion_pct"] == pytest.approx(0, abs=0.01)


def test_xirr_simple():
    irr = analisis.xirr([(date(2020, 1, 1), -100), (date(2021, 1, 1), 110)])
    assert irr == pytest.approx(0.10, abs=0.001)
    assert analisis.xirr([(date(2020, 1, 1), -100)]) is None


def test_dca_precio_constante_no_gana_nada():
    prices = series([10.0] * 400, freq="D")
    out = analisis.simulate_dca(prices, initial=1000, monthly=100)
    assert out["total_aportado"] == out["valor_final"]
    assert out["tir_anual_pct"] == pytest.approx(0, abs=0.01)
    assert out["aportaciones"] == 14  # enero 2020 a febrero 2021


def test_dca_precio_que_se_duplica():
    prices = series(np.linspace(10, 20, 366), freq="D")
    out = analisis.simulate_dca(prices, initial=0, monthly=100)
    assert out["valor_final"] > out["total_aportado"]
    assert out["si_todo_al_principio"] == pytest.approx(out["total_aportado"] * 2, rel=0.001)


@pytest.mark.parametrize("base,expected", [
    (6_000, 1_140), (15_000, 3_030), (50_000, 10_380), (250_000, 58_380), (400_000, 101_880),
])
def test_escala_del_ahorro(base, expected):
    # 15.000 € -> 3.030 € es el ejemplo publicado; el resto sale de aplicar 19/21/23/27/30 %.
    assert analisis.savings_tax(base)[0] == pytest.approx(expected)


def test_impuestos_compensa_perdidas_con_dividendos_hasta_25():
    out = analisis.spanish_savings_tax(gains=-2000, income=4000)
    assert out["base_ahorro"] == 3000  # 4.000 - 25 % de 4.000
    assert out["perdidas_pendientes_para_4_anos"] == 1000


def test_impuestos_perdidas_anteriores_y_retenciones():
    out = analisis.spanish_savings_tax(gains=5000, income=1000, previous_losses=6000, withholdings=190)
    assert out["base_ahorro"] == 750  # 5.000 compensados del todo y 250 (25 %) de los dividendos
    assert out["cuota"] == pytest.approx(142.5)
    assert out["resultado"] == pytest.approx(-47.5)
    assert out["resultado_texto"] == "a devolver"
    assert out["perdidas_pendientes_para_4_anos"] == 750
