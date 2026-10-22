"""Pruebas de la capa de Telegram con objetos simulados (sin conectarse a Telegram)."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_mercados import mercado
from bot_mercados.agent import Agent
from bot_mercados.config import Config
from bot_mercados.db import Database
from bot_mercados.telegram_bot import PAYLOAD, MarketBot, build_application


def make_config(**overrides) -> Config:
    values = dict(provider="groq", base_url="https://api.groq.com/openai/v1", api_key="x", models=("m",),
                  reasoning_effort=None, telegram_token="123:ABC", allowed_user_ids=frozenset(),
                  admin_user_ids=frozenset({99}), db_path=Path(":memory:"), premium_price_stars=250,
                  support_contact="@soporte", daily_summary_time="18:00")
    values.update(overrides)
    return Config(**values)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


class FakeMessage:
    def __init__(self, successful_payment=None):
        self.successful_payment = successful_payment
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


@pytest.fixture
def bot():
    db = Database(":memory:")
    return MarketBot(make_config(), Agent(None, ["m"]), db)


def test_build_application_registra_comandos_y_tareas():
    app = build_application(make_config(), Agent(None, ["m"]), Database(":memory:"))
    commands = {c for h in app.handlers[0] for c in getattr(h, "commands", ())}
    assert {"start", "premium", "paysupport", "terms", "cancelar", "alerta", "quiz", "stats"} <= commands
    assert {j.name for j in app.job_queue.jobs()} == {"alertas", "resumen"}
    with_channel = build_application(make_config(channel_id="@canal"), Agent(None, ["m"]), Database(":memory:"))
    assert "canal" in {j.name for j in with_channel.job_queue.jobs()}


def test_alertas_saltan_y_se_desactivan(bot, monkeypatch):
    db = bot.db
    db.ensure_user(1, 1, "Ana")
    low = db.add_alert(1, 1, "SAN.MC", "debajo", 10)
    high = db.add_alert(1, 1, "SAN.MC", "encima", 100)
    move = db.add_alert(1, 1, "SAN.MC", "movimiento", 5)
    monkeypatch.setattr(mercado, "last_prices", lambda ts: {"SAN.MC": {"precio": 9.5, "anterior": 10.1, "fecha": "x"}})
    context = SimpleNamespace(bot=FakeBot())

    asyncio.run(bot.check_alerts(context))
    texts = [t for _, t in context.bot.sent]
    assert len(texts) == 2 and "bajado a 9,50" in texts[0] and "-5,94 %" in texts[1]
    active = {a["id"] for a in db.active_alerts()}
    assert low not in active and high in active and move in active

    asyncio.run(bot.check_alerts(context))  # la de movimiento solo avisa una vez al día
    assert len(context.bot.sent) == 2


def _payment(**kw):
    values = dict(currency="XTR", total_amount=250, invoice_payload=PAYLOAD, telegram_payment_charge_id="ch1",
                  subscription_expiration_date=None, is_recurring=True, is_first_recurring=True)
    values.update(kw)
    return SimpleNamespace(**values)


def _update(message, user_id=1):
    return SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id, first_name="Ana"),
                           effective_chat=SimpleNamespace(id=user_id))


def test_pago_activa_premium_hasta_la_fecha_de_telegram(bot):
    expires = datetime.now(timezone.utc) + timedelta(days=30)
    message = FakeMessage(_payment(subscription_expiration_date=int(expires.timestamp())))
    asyncio.run(bot.paid(_update(message), SimpleNamespace()))
    assert bot.db.is_premium(1)
    assert abs((bot.db.premium_until(1) - expires).total_seconds()) < 2
    assert "Premium activado" in message.replies[0]

    asyncio.run(bot.paid(_update(FakeMessage(_payment(subscription_expiration_date=int(expires.timestamp())))),
                         SimpleNamespace()))
    assert bot.db.stats("x")["pagos"] == 1  # el aviso repetido de Telegram no cuenta dos veces


def test_renovacion_no_pierde_dias_regalados(bot):
    bot.db.ensure_user(1, 1, "Ana")
    gifted = bot.db.add_premium_days(1, 60)
    expires = datetime.now(timezone.utc) + timedelta(days=30)
    asyncio.run(bot.paid(_update(FakeMessage(_payment(telegram_payment_charge_id="ch2", subscription_expiration_date=int(expires.timestamp())))),
                         SimpleNamespace()))
    assert bot.db.premium_until(1) == gifted


# --- voz, documentos, inline, anuncios y errores ---

class FakeFile:
    def __init__(self, data):
        self.data = data

    async def download_as_bytearray(self):
        return bytearray(self.data)


class FakeMedia(SimpleNamespace):
    async def get_file(self):
        return FakeFile(self.data)


class RichMessage(FakeMessage):
    def __init__(self, text=None, voice=None, document=None, caption=None):
        super().__init__()
        self.text, self.voice, self.audio, self.document, self.caption = text, voice, None, document, caption
        self.photos = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)
        return StatusMessage(self.replies)

    async def reply_photo(self, photo, **kwargs):
        self.photos.append(photo)


class StatusMessage:
    def __init__(self, sink):
        self.sink = sink

    async def edit_text(self, text, **kwargs):
        self.sink.append(text)


class FakeChat(SimpleNamespace):
    async def send_action(self, action):
        pass


def _rich_update(message, user_id=1):
    return SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=user_id, first_name="Ana"),
                           effective_chat=FakeChat(id=user_id))


class FakeAgent:
    def __init__(self):
        self.questions = []

    def reply(self, chat_id, text, ctx):
        from bot_mercados.agent import Reply
        self.questions.append(text)
        return Reply(f"respuesta a: {text}")


def test_nota_de_voz_se_transcribe_y_se_responde(bot):
    class Transcriber:
        def transcribe(self, audio, filename):
            assert audio == b"ogg" and filename == "voz.ogg"
            return "¿cómo va el IBEX?"
    bot.transcriber, bot.agent = Transcriber(), FakeAgent()
    message = RichMessage(voice=FakeMedia(duration=5, data=b"ogg", file_name=None))
    asyncio.run(bot.voice(_rich_update(message), SimpleNamespace(bot=FakeBot())))
    assert message.replies[0] == "🎙️ «¿cómo va el IBEX?»" and bot.agent.questions == ["¿cómo va el IBEX?"]
    assert bot.db.queries_on(1, __import__("bot_mercados.planes", fromlist=["today"]).today()) == 1


def test_nota_de_voz_demasiado_larga(bot):
    bot.transcriber = SimpleNamespace(transcribe=lambda *a: pytest.fail("no debe transcribir"))
    message = RichMessage(voice=FakeMedia(duration=600, data=b"x", file_name=None))
    asyncio.run(bot.voice(_rich_update(message), SimpleNamespace(bot=FakeBot())))
    assert "muy largo" in message.replies[0]


def test_pdf_se_indexa_y_responde_la_pregunta_adjunta(bot):
    from bot_mercados.rag import KnowledgeBase
    from tests.test_rag import _pdf
    bot.kb, bot.agent = KnowledgeBase(":memory:"), FakeAgent()
    data = _pdf(["El grupo vendio 4.820 millones de euros en el ejercicio, un nueve por ciento mas que el anterior."])
    message = RichMessage(document=FakeMedia(file_name="Informe 2025.pdf", file_size=len(data), data=data),
                          caption="¿Cuánto vendió?")
    asyncio.run(bot.document(_rich_update(message), SimpleNamespace(bot=FakeBot())))
    assert message.replies[-2].startswith("✅ «Informe 2025» listo: 1 páginas")
    assert bot.agent.questions == ["¿Cuánto vendió?"]
    assert [d["title"] for d in bot.kb.documents(1)] == ["Informe 2025"]


def test_pdf_respeta_el_limite_del_plan(bot):
    from bot_mercados.planes import FREE
    from bot_mercados.rag import KnowledgeBase
    bot.kb = KnowledgeBase(":memory:")
    for i in range(FREE.documents):
        bot.kb.add_document(f"d{i}", [("pág. 1", "Texto de relleno suficientemente largo para generar un fragmento.")],
                            "pdf", owner_id=1)
    message = RichMessage(document=FakeMedia(file_name="otro.pdf", file_size=10, data=b"%PDF"))
    asyncio.run(bot.document(_rich_update(message), SimpleNamespace(bot=FakeBot())))
    assert "máximo de 3 documentos" in message.replies[0]


def test_consulta_inline_devuelve_tarjetas(bot, monkeypatch):
    from bot_mercados import tools
    monkeypatch.setattr(tools, "cotizacion", lambda ts: {"cotizaciones": [
        {"ticker": "SAN.MC", "nombre": "Santander", "precio": 11.5, "moneda": "EUR", "dia_pct": 1.0, "1a_pct": 40.0,
         "fecha": "2026-10-22", "vs_maximo_52s_pct": -3.0}]})
    answers = []

    class Query(SimpleNamespace):
        async def answer(self, results, cache_time=0):
            answers.append(results)
    q = Query(query="san.mc", from_user=SimpleNamespace(id=1))
    asyncio.run(bot.inline_query(SimpleNamespace(inline_query=q), SimpleNamespace(bot=SimpleNamespace(username="mibot"))))
    assert answers[0][0].title.startswith("SAN.MC · 11,50 EUR")
    assert "vía @mibot" in answers[0][0].input_message_content.message_text


def test_anuncio_respeta_bajas_y_bloqueos(bot, monkeypatch):
    from telegram.error import Forbidden
    from bot_mercados import telegram_bot
    monkeypatch.setattr(telegram_bot, "BROADCAST_DELAY", 0)
    for uid in (1, 2, 3):
        bot.db.ensure_user(uid, uid, f"u{uid}")
    bot.db.set_announcements(2, False)

    class Bot(FakeBot):
        async def send_message(self, chat_id, text, **kwargs):
            if chat_id == 3:
                raise Forbidden("bot was blocked by the user")
            self.sent.append((chat_id, text))
    context = SimpleNamespace(bot=Bot())
    message = RichMessage(text="/anuncio Nueva función: cartera")
    asyncio.run(bot.broadcast(_rich_update(message, user_id=99), context))
    assert [c for c, _ in context.bot.sent] == [1]
    assert "1 usuarios (1 fallidos)" in message.replies[0]
    assert bot.db.announcement_chats() == [1]  # el que bloqueó deja de recibirlos


def test_aviso_de_errores_al_admin_limitado(bot):
    context = SimpleNamespace(bot=FakeBot(), error=RuntimeError("fallo raro"))
    asyncio.run(bot.on_error(None, context))
    asyncio.run(bot.on_error(None, context))
    assert len(context.bot.sent) == 1 and context.bot.sent[0][0] == 99


def test_precheckout_rechaza_facturas_raras(bot):
    answers = []

    class Query(SimpleNamespace):
        async def answer(self, ok, error_message=None):
            answers.append(ok)

    for payload, amount in ((PAYLOAD, 250), ("otra", 250), (PAYLOAD, 1)):
        q = Query(invoice_payload=payload, currency="XTR", total_amount=amount)
        asyncio.run(bot.precheckout(SimpleNamespace(pre_checkout_query=q), SimpleNamespace()))
    assert answers == [True, False, False]
