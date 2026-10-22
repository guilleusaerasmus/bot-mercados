"""Pruebas del modelo de negocio: planes, límites, invitaciones, pagos y quiz."""

from datetime import datetime, timedelta, timezone

import pytest

from bot_mercados import formato, quiz
from bot_mercados.db import Database
from bot_mercados.planes import FREE, PREMIUM, REFERRAL_DAYS, REFERRAL_MAX_REWARDS, Plans


@pytest.fixture
def db():
    return Database(":memory:")


def test_alta_de_usuario_una_sola_vez(db):
    assert db.ensure_user(1, 10, "Ana") is True
    assert db.ensure_user(1, 11, "Ana María") is False
    assert db.get_user(1)["chat_id"] == 11


def test_limite_diario_de_preguntas(db):
    db.ensure_user(1, 1, "Ana")
    plans = Plans(db)
    results = [plans.try_consume_query(1)[0] for _ in range(FREE.ai_queries_per_day + 1)]
    assert results.count(True) == FREE.ai_queries_per_day and results[-1] is False


def test_devolver_pregunta_fallida(db):
    db.ensure_user(1, 1, "Ana")
    db.count_query(1, "2026-01-01")
    db.uncount_query(1, "2026-01-01")
    db.uncount_query(1, "2026-01-01")
    assert db.queries_on(1, "2026-01-01") == 0


def test_premium_y_admin(db):
    db.ensure_user(1, 1, "Ana")
    plans = Plans(db, admin_ids=frozenset({99}))
    assert plans.plan_for(1) is FREE
    db.add_premium_days(1, 30)
    assert plans.plan_for(1) is PREMIUM
    assert plans.plan_for(99).ai_queries_per_day > PREMIUM.ai_queries_per_day


def test_dias_premium_se_acumulan(db):
    db.ensure_user(1, 1, "Ana")
    first = db.add_premium_days(1, 3)
    second = db.add_premium_days(1, 3)
    assert second - first == timedelta(days=3)


def test_premium_caducado(db):
    db.ensure_user(1, 1, "Ana")
    db.set_premium_until(1, datetime.now(timezone.utc) - timedelta(seconds=1))
    assert not db.is_premium(1)


def test_invitacion_se_recompensa_una_vez(db):
    db.ensure_user(1, 1, "Anfitrión")
    db.ensure_user(2, 2, "Invitado", referred_by=1)
    plans = Plans(db)
    referrer, until = plans.reward_referral(2)
    assert referrer == 1 and db.is_premium(1)
    assert until - datetime.now(timezone.utc) > timedelta(days=REFERRAL_DAYS - 1)
    assert plans.reward_referral(2) is None


def test_no_vale_invitarse_a_si_mismo(db):
    db.ensure_user(1, 1, "Ana", referred_by=1)
    assert db.get_user(1)["referred_by"] is None


def test_tope_de_invitaciones(db):
    db.ensure_user(1, 1, "Anfitrión")
    plans = Plans(db)
    for i in range(REFERRAL_MAX_REWARDS + 2):
        db.ensure_user(100 + i, 100 + i, f"amigo{i}", referred_by=1)
        plans.reward_referral(100 + i)
    assert db.rewarded_referrals(1) == REFERRAL_MAX_REWARDS


def test_pago_duplicado_no_cuenta_dos_veces(db):
    db.ensure_user(1, 1, "Ana")
    assert db.record_payment("ch1", 1, 250, "XTR", "premium_30d", True, None)
    assert not db.record_payment("ch1", 1, 250, "XTR", "premium_30d", True, None)
    assert db.stats("2026-01-01")["stars_cobradas"] == 250
    db.mark_refunded("ch1")
    assert db.stats("2026-01-01")["stars_cobradas"] == 0


def test_alerta_solo_la_borra_su_dueno(db):
    db.ensure_user(1, 1, "Ana")
    alert_id = db.add_alert(1, 1, "SAN.MC", "debajo", 10)
    assert not db.delete_alert(2, alert_id)
    assert db.delete_alert(1, alert_id)


def test_destinatarios_del_resumen(db):
    db.ensure_user(1, 1, "Premium")
    db.ensure_user(2, 2, "Gratis")
    db.add_premium_days(1, 5)
    db.watch(1, "SAN.MC")
    db.watch(2, "SAN.MC")
    assert [r["user_id"] for r in db.premium_watchers()] == [1]


def test_preguntas_del_quiz_cumplen_limites_de_telegram():
    for q in quiz.QUESTIONS:
        assert len(q.text) <= 300 and len(q.explanation) <= 200
        assert 2 <= len(q.options) <= 10 and all(len(o) <= 100 for o in q.options)
        assert 0 <= q.correct < len(q.options)


def test_barajar_mantiene_la_respuesta_correcta():
    import random
    q = quiz.QUESTIONS[0]
    for seed in range(20):
        options, correct = quiz.shuffled(q, random.Random(seed))
        assert options[correct] == q.options[q.correct]


def test_markdown_de_la_ia_a_html_de_telegram():
    html = formato.ai_to_html("### Resumen\n**Santander** sube <5 %> & `SAN.MC`\n* uno\n- dos")
    assert html == "<b>Resumen</b>\n<b>Santander</b> sube &lt;5 %&gt; &amp; <code>SAN.MC</code>\n• uno\n• dos"
    assert formato.plain("**hola** `x`") == "hola x"


def test_trocear_mensajes_por_parrafos():
    text = "\n\n".join(f"Párrafo {i} " + "x" * 900 for i in range(10))
    parts = formato.split_message(text, limit=3500)
    assert all(len(p) <= 3500 for p in parts) and "\n\n".join(parts) == text
    assert formato.split_message("y" * 8000, limit=3500)[0] == "y" * 3500


def test_formato_numeros_a_la_espanola():
    assert formato.n(1234567.891) == "1.234.567,89"
    assert formato.n(None) == "—"
    assert formato.big(2_500_000_000) == "2,5 mil M"
    assert formato.change(1.5).startswith("🟢 +1,50")
