"""Interfaz de Telegram: comandos, conversación con la IA, pagos, alertas y tareas programadas."""

import asyncio
import contextlib
import logging
import re
import time
from collections import defaultdict, deque
from datetime import date, datetime, time as dtime, timedelta, timezone
from functools import partial
from html import escape

from telegram import (BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle,
                      InputTextMessageContent, LabeledPrice, Poll, Update)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, InlineQueryHandler,
                          MessageHandler, PollAnswerHandler, PreCheckoutQueryHandler, filters)

from . import formato, graficos, mercado, quiz, tools
from .agent import Agent
from .config import Config
from .db import Database
from .documentos import DocumentError, SecClient, pdf_sections
from .planes import ADMIN, PREMIUM, REFERRAL_DAYS, TZ, Plans, format_until, today, utc_from_timestamp
from .rag import KnowledgeBase
from .tools import ToolContext, ToolError
from .voz import MAX_VOICE_SECONDS, Transcriber, TranscriptionError

log = logging.getLogger(__name__)

PAYLOAD = "premium_30d"
SUBSCRIPTION_SECONDS = 30 * 24 * 3600  # Telegram solo admite suscripciones de 30 días
ALERT_CHECK_SECONDS = 600
COMMANDS_PER_MINUTE = 20
ERROR_NOTICE_SECONDS = 300  # como mucho un aviso de error al administrador cada 5 minutos
BROADCAST_DELAY = 0.05  # ~20 mensajes por segundo, por debajo del límite de Telegram
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # lo máximo que Telegram deja descargar a un bot

COMMANDS = [
    ("mercado", "Cómo van hoy los mercados"),
    ("cotizacion", "Precio de uno o varios valores: /cotizacion SAN.MC AAPL"),
    ("ficha", "Ficha completa de una empresa: /ficha ITX.MC"),
    ("top", "Buscador: /top dividendo, /top bajan_hoy us"),
    ("cartera", "Tu cartera real con ganancias en euros"),
    ("compra", "Apuntar una compra: /compra SAN.MC 100 11,20"),
    ("venta", "Apuntar una venta: /venta SAN.MC 50 12,10"),
    ("documentos", "Tus PDFs para preguntarme sobre ellos"),
    ("grafico", "Gráfico: /grafico SAN.MC 1y o /grafico SAN.MC BBVA.MC"),
    ("noticias", "Últimas noticias: /noticias IBE.MC"),
    ("lista", "Tu lista de seguimiento"),
    ("seguir", "Añadir a tu lista: /seguir AAPL"),
    ("dejar", "Quitar de tu lista: /dejar AAPL"),
    ("alertas", "Ver y borrar tus alertas"),
    ("alerta", "Crear alerta: /alerta SAN.MC debajo 10"),
    ("quiz", "Pregunta de finanzas para aprender"),
    ("premium", "Planes y suscripción Premium"),
    ("cuenta", "Tu plan y uso de hoy"),
    ("invitar", "Invita amigos y gana días Premium"),
    ("avisos", "Recibir o no anuncios: /avisos off"),
    ("reset", "Borrar la conversación con la IA"),
    ("ayuda", "Qué puedo hacer"),
    ("terms", "Términos y privacidad"),
    ("paysupport", "Ayuda con pagos y reembolsos"),
]

HELP = """<b>📈 Bot de Mercados</b> — tu analista de bolsa en Telegram.

<b>Pregúntame en lenguaje normal</b> (usa la IA):
• ¿Qué opinas de Inditex?
• Compara Santander, BBVA y el IBEX en 5 años con un gráfico
• Si hubiera invertido 200 € al mes en el S&P 500 desde hace 10 años, ¿cuánto tendría?
• Analiza mi cartera: 40 % SAN.MC, 30 % AAPL, 30 % IBE.MC
• Vendí acciones con 8.000 € de ganancia y cobré 500 € en dividendos, ¿cuánto pago a Hacienda?
• Avísame si Iberdrola baja de 15 €
• ¿Qué empresas españolas pagan más dividendo?
• He comprado 50 acciones de Inditex a 54 €
• ¿Qué riesgos destaca Apple en su informe anual?
🎙️ También puedes mandarme <b>notas de voz</b>, y 📄 <b>PDFs</b> (informes anuales, presentaciones...)
para preguntarme sobre ellos: te respondo citando la página.

<b>Comandos rápidos</b> (ilimitados, sin IA):
/mercado · /cotizacion · /ficha · /grafico · /top · /cartera · /noticias · /lista · /alertas · /quiz

<b>Compártelo:</b> escribe <code>@{bot} SAN.MC</code> en cualquier chat para enviar una cotización.
/premium para más consultas, alertas y resumen diario · /invitar para ganar días gratis.
<i>Información, no asesoramiento financiero. Datos con posible retraso.</i>"""

TERMS = """<b>Términos y privacidad</b>

1. <b>Servicio informativo y educativo.</b> Nada de lo que dice el bot es una recomendación de inversión ni asesoramiento financiero personalizado. Invertir conlleva riesgo de pérdida.
2. <b>Datos.</b> Proceden de fuentes públicas (Yahoo Finance, BCE, Google News) y pueden tener retrasos o errores. Las respuestas de la IA pueden contener errores: verifica los datos importantes.
3. <b>Premium.</b> Suscripción de 30 días pagada con Telegram Stars que se renueva automáticamente. Puedes cancelarla cuando quieras con /cancelar o desde los ajustes de Telegram; seguirás teniendo Premium hasta el final del periodo pagado.
4. <b>Reembolsos.</b> Si algo no funciona, escribe /paysupport en los 14 días siguientes al pago y lo revisaremos.
5. <b>Privacidad.</b> Guardamos tu ID y nombre de Telegram, tu uso diario, tus alertas, tu lista y tus pagos, solo para prestar el servicio. Tus preguntas se envían al proveedor de IA para responderlas y no se guardan en nuestra base de datos. Puedes pedir el borrado de tus datos por soporte.
6. El soporte de Telegram no puede ayudarte con compras hechas en este bot: usa /paysupport."""


def _ticker_buttons(ticker: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📈 Gráfico", callback_data=f"g:{ticker}"),
        InlineKeyboardButton("🏢 Ficha", callback_data=f"f:{ticker}"),
        InlineKeyboardButton("📰 Noticias", callback_data=f"n:{ticker}"),
        InlineKeyboardButton("👀 Seguir", callback_data=f"s:{ticker}"),
    ]])


class MarketBot:
    def __init__(self, config: Config, agent: Agent, db: Database, transcriber: Transcriber | None = None,
                 kb: KnowledgeBase | None = None, sec: SecClient | None = None):
        self.config = config
        self.agent = agent
        self.db = db
        self.transcriber = transcriber
        self.kb = kb
        self.sec = sec
        self.plans = Plans(db, config.admin_user_ids)
        self._recent: dict[int, deque] = defaultdict(deque)
        # None = aún no se ha avisado. No vale 0.0: monotonic() cuenta desde el arranque de la máquina y en un
        # servidor recién encendido valdría menos de 5 minutos, así que nunca se avisaría.
        self._last_error_notice: float | None = None

    # --- utilidades ---------------------------------------------------------------------

    def _register(self, update: Update, referred_by: int | None = None) -> bool:
        user, chat = update.effective_user, update.effective_chat
        if not user or not chat:
            return False
        return self.db.ensure_user(user.id, chat.id, user.first_name, referred_by)

    def _throttled(self, user_id: int) -> bool:
        """Freno anti-abuso para los comandos sin IA (protege la fuente de datos)."""
        now = time.monotonic()
        recent = self._recent[user_id]
        while recent and now - recent[0] > 60:
            recent.popleft()
        if len(recent) >= COMMANDS_PER_MINUTE:
            return True
        recent.append(now)
        return False

    async def _run(self, func, *args, **kwargs):
        """Las herramientas son síncronas (red, pandas): se ejecutan en un hilo aparte."""
        return await asyncio.to_thread(partial(func, *args, **kwargs))

    async def _guarded(self, update: Update, coro_factory) -> None:
        """Ejecuta un comando con registro, freno y mensajes de error amables."""
        self._register(update)
        message = update.effective_message
        if self._throttled(update.effective_user.id):
            await message.reply_text("Vas muy rápido 🙂 Espera unos segundos.")
            return
        try:
            await coro_factory()
        except ToolError as exc:
            await message.reply_text(f"⚠️ {exc}")
        except Exception:
            log.exception("Error en un comando")
            await message.reply_text("⚠️ La fuente de datos no responde ahora mismo. Inténtalo en un rato.")

    @staticmethod
    def _tickers(args: list[str]) -> list[str]:
        return [a for a in args if a.lower() not in tools.PERIODS]

    @staticmethod
    def _period(args: list[str], default: str = "1y") -> str:
        return next((a.lower() for a in args if a.lower() in tools.PERIODS), default)

    # --- comandos básicos ----------------------------------------------------------------

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        referred_by = None
        if context.args and context.args[0].startswith("ref_") and context.args[0][4:].isdigit():
            referred_by = int(context.args[0][4:])
        is_new = self._register(update, referred_by)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌍 Mercados hoy", callback_data="cmd:mercado"),
             InlineKeyboardButton("🧠 Quiz", callback_data="cmd:quiz")],
            [InlineKeyboardButton("⭐ Premium", callback_data="cmd:premium"),
             InlineKeyboardButton("🎁 Invitar", callback_data="cmd:invitar")],
        ])
        greeting = f"¡Hola, {escape(update.effective_user.first_name or '')}! 👋\n\n" if is_new else ""
        await update.effective_message.reply_text(greeting + self._help(context), parse_mode=ParseMode.HTML,
                                                  reply_markup=keyboard)

    @staticmethod
    def _help(context: ContextTypes.DEFAULT_TYPE) -> str:
        return HELP.replace("{bot}", escape(context.bot.username or "este_bot"))

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(self._help(context), parse_mode=ParseMode.HTML)

    async def terms(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(TERMS, parse_mode=ParseMode.HTML)

    async def support(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        contact = self.config.support_contact or "el administrador del bot"
        await update.effective_message.reply_text(
            f"🛟 <b>Soporte</b>\nEscribe a {escape(contact)} indicando tu ID de usuario "
            f"(<code>{update.effective_user.id}</code>) y qué ha pasado. Si es un pago, incluye la fecha.\n\n"
            "Reembolsos: dentro de los 14 días siguientes al pago si el servicio no ha funcionado. "
            "El soporte de Telegram no puede ayudarte con compras hechas en este bot.",
            parse_mode=ParseMode.HTML)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.agent.reset(update.effective_chat.id)
        await update.effective_message.reply_text("Conversación borrada.")

    # --- comandos de mercado (sin IA) -------------------------------------------------------

    async def market(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            await update.effective_chat.send_action(ChatAction.TYPING)
            data = await self._run(tools.resumen_mercado)
            await update.effective_message.reply_text(formato.market(data), parse_mode=ParseMode.HTML)
        await self._guarded(update, run)

    async def quote(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            tickers = self._tickers(context.args or [])
            if not tickers:
                await update.effective_message.reply_text("Uso: /cotizacion SAN.MC AAPL")
                return
            data = await self._run(tools.cotizacion, tickers)
            first_ok = next((q["ticker"] for q in data["cotizaciones"] if "error" not in q), None)
            await update.effective_message.reply_text(formato.quotes(data), parse_mode=ParseMode.HTML,
                                                      reply_markup=_ticker_buttons(first_ok) if first_ok else None)
        await self._guarded(update, run)

    async def _send_company(self, message, ticker: str) -> None:
        data = await self._run(tools.ficha_empresa, ticker)
        await message.reply_text(formato.company(data), parse_mode=ParseMode.HTML, reply_markup=_ticker_buttons(ticker))

    async def company(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            if not context.args:
                await update.effective_message.reply_text("Uso: /ficha ITX.MC")
                return
            await update.effective_chat.send_action(ChatAction.TYPING)
            await self._send_company(update.effective_message, context.args[0])
        await self._guarded(update, run)

    async def _send_chart(self, message, tickers: list[str], period: str) -> None:
        ctx = ToolContext(user_id=0, chat_id=message.chat_id)
        await self._run(tools.grafico, tickers, period, ctx=ctx)
        for png, caption in ctx.images:
            await message.reply_photo(png, caption=caption)

    async def chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            args = context.args or []
            tickers = self._tickers(args)
            if not tickers:
                await update.effective_message.reply_text(
                    "Uso: /grafico SAN.MC [periodo]  o  /grafico SAN.MC BBVA.MC ^IBEX 5y\n"
                    f"Periodos: {', '.join(tools.PERIODS)}")
                return
            await update.effective_chat.send_action(ChatAction.UPLOAD_PHOTO)
            await self._send_chart(update.effective_message, tickers, self._period(args))
        await self._guarded(update, run)

    async def _send_news(self, message, query: str) -> None:
        data = await self._run(tools.noticias, query, incluir_enlaces=True)
        await message.reply_text(formato.news(data), parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    async def news(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            if not context.args:
                await update.effective_message.reply_text("Uso: /noticias SAN.MC  o  /noticias inteligencia artificial")
                return
            await self._send_news(update.effective_message, " ".join(context.args)[:60])
        await self._guarded(update, run)

    # --- lista y alertas -----------------------------------------------------------------

    def _ctx(self, update: Update) -> ToolContext:
        return ToolContext(user_id=update.effective_user.id, chat_id=update.effective_chat.id, db=self.db,
                           plans=self.plans, kb=self.kb, sec=self.sec)

    @contextlib.asynccontextmanager
    async def _typing(self, chat):
        """Mantiene el «escribiendo…» mientras dura algo lento (Telegram lo borra a los 5 segundos)."""
        async def keep_alive():
            while True:
                try:
                    await chat.send_action(ChatAction.TYPING)
                except TelegramError:
                    pass
                await asyncio.sleep(4.5)
        task = asyncio.create_task(keep_alive())
        try:
            yield
        finally:
            task.cancel()

    # --- documentos (RAG) ---------------------------------------------------------------------

    def _ingest_pdf(self, user_id: int, title: str, data: bytes) -> tuple[int, int]:
        """Lee el PDF y lo indexa (se ejecuta en un hilo: puede tardar varios segundos)."""
        kb = self.kb
        if kb is None:
            raise DocumentError("La lectura de documentos no está activada en este bot.")
        sections, pages = pdf_sections(data)
        try:
            doc_id = kb.add_document(title, sections, "pdf", owner_id=user_id)
        except ValueError as exc:
            raise DocumentError(str(exc)) from exc
        chunks = next(d["chunks"] for d in kb.documents(user_id) if d["id"] == doc_id)
        return pages, chunks

    async def document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        message, uid = update.effective_message, update.effective_user.id
        if self.kb is None:
            await message.reply_text("La lectura de documentos no está activada en este bot.")
            return
        if self._throttled(uid):
            await message.reply_text("Vas muy rápido 🙂 Espera unos segundos.")
            return
        plan = self.plans.plan_for(uid)
        if len(self.kb.documents(uid)) >= plan.documents:
            await message.reply_text(f"Has llegado al máximo de {plan.documents} documentos del plan {plan.name}. "
                                     "Borra alguno con /documentos o pásate a /premium.")
            return
        doc = message.document
        if doc.file_size and doc.file_size > MAX_UPLOAD_BYTES:
            await message.reply_text("El PDF pesa más de 20 MB, el máximo que Telegram deja descargar a los bots.")
            return
        title = re.sub(r"\.pdf$", "", doc.file_name or "Documento", flags=re.I)[:80]
        status = await message.reply_text(f"📄 Leyendo «{title}»…")
        try:
            async with self._typing(update.effective_chat):
                data = bytes(await (await doc.get_file()).download_as_bytearray())
                pages, chunks = await self._run(self._ingest_pdf, uid, title, data)
        except DocumentError as exc:
            await status.edit_text(f"⚠️ {exc}")
            return
        await status.edit_text(f"✅ «{title}» listo: {pages} páginas, {chunks} fragmentos.\n"
                               "Pregúntame lo que quieras sobre él: te contesto citando la página.")
        if message.caption:  # «¿qué riesgos menciona?» escrito junto al PDF
            await self._ask(update, context, message.caption)

    async def documents(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        if self.kb is None:
            await update.effective_message.reply_text("La lectura de documentos no está activada en este bot.")
            return
        uid = update.effective_user.id
        docs, plan = self.kb.documents(uid), self.plans.plan_for(uid)
        if not docs:
            await update.effective_message.reply_text(
                "📚 No tienes documentos. Envíame un PDF (informe anual, presentación de resultados, folleto de un "
                "fondo...) y pregúntame sobre él.\nPara empresas de EE. UU. uso directamente su informe anual oficial "
                "(10-K): pregúntame, p. ej., «¿qué riesgos destaca Apple en su informe anual?».")
            return
        lines = [f"📚 <b>Tus documentos</b> ({len(docs)}/{plan.documents})"]
        lines += [f"• {escape(d['title'])} · {d['chunks']} fragmentos" for d in docs]
        buttons = [[InlineKeyboardButton(f"🗑 {d['title'][:28]}", callback_data=f"dd:{d['id']}")] for d in docs]
        await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML,
                                                  reply_markup=InlineKeyboardMarkup(buttons))

    async def watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            data = await self._run(tools.seguimiento, "ver", ctx=self._ctx(update))
            await update.effective_message.reply_text(formato.watchlist(data), parse_mode=ParseMode.HTML)
        await self._guarded(update, run)

    async def follow(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str = "anadir") -> None:
        async def run():
            if not context.args:
                await update.effective_message.reply_text(f"Uso: /{'seguir' if action == 'anadir' else 'dejar'} SAN.MC AAPL")
                return
            data = await self._run(tools.seguimiento, action, context.args, ctx=self._ctx(update))
            verb = "Añadidos" if action == "anadir" else "Quitados"
            changed = ", ".join(data["cambiados"]) or "ninguno"
            await update.effective_message.reply_text(f"{verb}: {changed}\nTu lista: {', '.join(data['lista']) or 'vacía'}")
        await self._guarded(update, run)

    async def alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        rows = self.db.user_alerts(update.effective_user.id)
        plan = self.plans.plan_for(update.effective_user.id)
        if not rows:
            await update.effective_message.reply_text(
                "No tienes alertas. Crea una con /alerta SAN.MC debajo 10, /alerta AAPL encima 400 "
                "o /alerta TSLA movimiento 5 (avisa si se mueve más de un 5 % en un día). "
                "También puedes pedírmelo en lenguaje normal.")
            return
        keyboard = [[InlineKeyboardButton(f"🗑 Borrar #{a['id']}", callback_data=f"da:{a['id']}")] for a in rows]
        text = f"🔔 <b>Tus alertas</b> ({len(rows)}/{plan.alerts})\n" + "\n".join(formato.alert_line(a) for a in rows)
        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    async def new_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            args = context.args or []
            if len(args) != 3 or args[1].lower() not in tools.ALERT_KINDS:
                await update.effective_message.reply_text(
                    "Uso: /alerta TICKER encima|debajo|movimiento VALOR\nEj.: /alerta SAN.MC debajo 10")
                return
            try:
                value = float(args[2].replace(",", "."))
            except ValueError:
                raise ToolError("El valor debe ser un número.") from None
            data = await self._run(tools.alertas, "crear", args[0], args[1].lower(), value, ctx=self._ctx(update))
            line = formato.alert_line({"id": data["id"], "ticker": data["ticker"], "kind": data["tipo"], "value": data["valor"]})
            extra = f"\n⚠️ {data['aviso']}" if data.get("aviso") else ""
            await update.effective_message.reply_text(
                f"✅ Alerta creada: {line}\nPrecio actual: {formato.n(data['precio_actual'])} {data.get('moneda', '')}{extra}",
                parse_mode=ParseMode.HTML)
        await self._guarded(update, run)

    # --- cartera personal y buscador ---------------------------------------------------------

    async def portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            await update.effective_chat.send_action(ChatAction.TYPING)
            data = await self._run(tools.mi_cartera, "ver", ctx=self._ctx(update))
            await update.effective_message.reply_text(formato.portfolio(data), parse_mode=ParseMode.HTML)
            weights = {p["ticker"]: p["peso_pct"] for p in data.get("posiciones", []) if p.get("peso_pct")}
            if len(weights) >= 2:
                png = await self._run(graficos.allocation_chart, weights, f"{formato.n(data['total']['valor_eur'], 0)} €")
                await update.effective_message.reply_photo(png)
        await self._guarded(update, run)

    async def trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE, action: str = "comprar") -> None:
        async def run():
            args = context.args or []
            command = "compra" if action == "comprar" else "venta"
            if len(args) not in (2, 3):
                await update.effective_message.reply_text(
                    f"Uso: /{command} TICKER ACCIONES [PRECIO]\nEj.: /{command} SAN.MC 100 11,20 "
                    "(sin precio se usa el actual; el precio va en la moneda del valor)")
                return
            try:
                shares = float(args[1].replace(",", "."))
                price = float(args[2].replace(",", ".")) if len(args) == 3 else None
            except ValueError:
                raise ToolError("Las acciones y el precio deben ser números.") from None
            data = await self._run(tools.mi_cartera, action, args[0], shares, price, ctx=self._ctx(update))
            if action == "comprar":
                c, pos = data["compra"], data["posicion"]
                text = (f"✅ Compra apuntada: {formato.n(c['acciones'], 2)} {escape(c['ticker'])} a {formato.n(c['precio'])} "
                        f"{escape(c.get('moneda') or '')}\nAhora tienes {formato.n(pos['acciones'], 2)} acciones, "
                        f"coste medio {formato.n(pos['coste_medio'])}.")
            else:
                v = data["venta"]
                text = (f"✅ Venta apuntada: {formato.n(v['acciones'], 2)} {escape(v['ticker'])} a {formato.n(v['precio'])}\n"
                        f"Resultado: {formato.change(data['ganancia_eur'], '€')} (FIFO, como exige Hacienda)")
                if data.get("aviso_regla_dos_meses"):
                    text += f"\n⚠️ {escape(data['aviso_regla_dos_meses'])}"
            await update.effective_message.reply_text(text + "\n\nMira tu cartera con /cartera.", parse_mode=ParseMode.HTML)
        await self._guarded(update, run)

    async def top(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        async def run():
            criterion, market, sector = "dividendo", "es", None
            for arg in (a.lower() for a in context.args or []):
                if arg in tools.SCREEN_CRITERIA:
                    criterion = arg
                elif arg in mercado.SCREEN_MARKETS:
                    market = arg
                elif arg in tools.SECTORS:
                    sector = arg
                else:
                    await update.effective_message.reply_text(
                        f"Uso: /top [criterio] [es|us] [sector]\nCriterios: {', '.join(tools.SCREEN_CRITERIA)}\n"
                        f"Sectores: {', '.join(tools.SECTORS)}")
                    return
            await update.effective_chat.send_action(ChatAction.TYPING)
            data = await self._run(tools.buscar_acciones, criterion, market, sector)
            await update.effective_message.reply_text(formato.screener(data), parse_mode=ParseMode.HTML)
        await self._guarded(update, run)

    # --- compartir en cualquier chat (modo inline) ------------------------------------------

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.inline_query
        tickers = [t for t in (query.query or "").upper().replace(",", " ").split() if tools.valid_ticker(t)][:3]
        if not tickers or self._throttled(query.from_user.id):
            await query.answer([], cache_time=5)
            return
        try:
            data = await self._run(tools.cotizacion, tickers)
        except Exception:
            log.exception("Fallo en la consulta inline")
            await query.answer([], cache_time=5)
            return
        bot_name = context.bot.username
        button = InlineKeyboardMarkup([[InlineKeyboardButton("🤖 Analízalo gratis", url=f"https://t.me/{bot_name}?start=inline")]])
        results = []
        for item in data["cotizaciones"]:
            if "error" in item:
                continue
            results.append(InlineQueryResultArticle(
                id=item["ticker"][:64],
                title=f"{item['ticker']} · {formato.n(item['precio'])} {item.get('moneda', '')}",
                description=f"Hoy {formato.change(item.get('dia_pct'))} · 12 m {formato.change(item.get('1a_pct'))}",
                input_message_content=InputTextMessageContent(
                    formato.quotes({"cotizaciones": [item]}) + f"\n\n📈 vía @{escape(bot_name)}", parse_mode=ParseMode.HTML),
                reply_markup=button))
        await query.answer(results, cache_time=60)

    # --- quiz ----------------------------------------------------------------------------

    async def send_quiz(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
        _, q = quiz.random_question()
        options, correct = quiz.shuffled(q)
        message = await context.bot.send_poll(
            chat_id, q.text, options, is_anonymous=False, type=Poll.QUIZ, correct_option_id=correct,
            explanation=q.explanation,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🧠 Otra pregunta", callback_data="cmd:quiz")]]))
        pending = context.bot_data.setdefault("quiz", {})
        if len(pending) > 5000:  # encuestas nunca respondidas: evita que crezca sin límite
            pending.clear()
        pending[message.poll.id] = correct

    async def quiz_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        await self.send_quiz(update.effective_chat.id, context)

    async def quiz_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        answer = update.poll_answer
        correct = context.bot_data.get("quiz", {}).pop(answer.poll_id, None)
        if correct is None or not answer.user or not answer.option_ids:
            return
        self.db.record_quiz(answer.user.id, answer.option_ids[0] == correct)

    async def ranking(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        rows = self.db.quiz_ranking()
        if not rows:
            await update.effective_message.reply_text("Aún no hay puntuaciones. ¡Empieza con /quiz!")
            return
        medals = ["🥇", "🥈", "🥉"] + ["▫️"] * 10
        lines = [f"{medals[i]} {escape(r['name'] or 'Anónimo')}: {r['quiz_ok']}/{r['quiz_total']}" for i, r in enumerate(rows)]
        await update.effective_message.reply_text("🏆 <b>Ranking del quiz</b>\n" + "\n".join(lines), parse_mode=ParseMode.HTML)

    # --- cuenta, premium e invitaciones ------------------------------------------------------

    async def account(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        uid = update.effective_user.id
        plan = self.plans.plan_for(uid)
        used = self.db.queries_on(uid, today())
        user = self.db.get_user(uid) or {}
        lines = [f"👤 <b>Tu cuenta</b> · plan <b>{plan.name}</b>"]
        until = self.db.premium_until(uid)
        if plan is PREMIUM and until:
            lines.append(f"Premium hasta el {format_until(until)}")
        lines += [f"Preguntas a la IA hoy: {used}/{plan.ai_queries_per_day}",
                  f"Alertas: {len(self.db.user_alerts(uid))}/{plan.alerts} · Lista: {len(self.db.watchlist(uid))}/{plan.watchlist}",
                  f"Amigos invitados que usan el bot: {self.db.rewarded_referrals(uid)}",
                  f"Quiz: {user.get('quiz_ok', 0)}/{user.get('quiz_total', 0)} aciertos"]
        if plan is not PREMIUM and plan is not ADMIN:
            lines.append("\n⭐ Pásate a /premium o gana días gratis con /invitar.")
        await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

    async def premium(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        price = self.config.premium_price_stars
        link = await context.bot.create_invoice_link(
            title="Bot de Mercados Premium",
            description=f"30 días de Premium: {PREMIUM.ai_queries_per_day} preguntas a la IA al día, "
                        f"{PREMIUM.alerts} alertas y resumen diario. Se renueva cada 30 días; cancela cuando quieras.",
            payload=PAYLOAD, currency="XTR", prices=[LabeledPrice("Premium 30 días", price)],
            subscription_period=SUBSCRIPTION_SECONDS)
        until = self.db.premium_until(update.effective_user.id)
        status = (f"\n✅ Ya eres Premium hasta el {format_until(until)}." if until and until > datetime.now(timezone.utc) else "")
        await update.effective_message.reply_text(
            "⭐ <b>Planes</b>\n\n" + formato.plans_table(price) + status +
            "\nAl suscribirte aceptas los /terms. Cancela cuando quieras con /cancelar.",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(f"⭐ Suscribirme por {price} Stars/mes", url=link)]]))

    async def precheckout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.pre_checkout_query
        # Telegram exige responder en menos de 10 segundos: aquí no se hace nada lento.
        if q.invoice_payload != PAYLOAD or q.currency != "XTR" or q.total_amount != self.config.premium_price_stars:
            await q.answer(ok=False, error_message="Esta factura ha caducado. Vuelve a pedirla con /premium.")
            return
        await q.answer(ok=True)

    async def paid(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        sp = update.effective_message.successful_payment
        uid = update.effective_user.id
        self._register(update)
        expires = utc_from_timestamp(sp.subscription_expiration_date) if sp.subscription_expiration_date else None
        is_new = self.db.record_payment(sp.telegram_payment_charge_id, uid, sp.total_amount, sp.currency,
                                        sp.invoice_payload, bool(sp.is_recurring), expires)
        if not is_new:
            return
        if expires:
            current = self.db.premium_until(uid)
            # Si tenía días regalados (invitaciones) por delante, no se pierden.
            self.db.set_premium_until(uid, max(expires, current) if current else expires)
        else:
            self.db.add_premium_days(uid, 30)
        until = self.db.premium_until(uid)
        log.info("Pago recibido: usuario %s, %s %s", uid, sp.total_amount, sp.currency)
        renewal = "renovado" if sp.is_recurring and not sp.is_first_recurring else "activado"
        await update.effective_message.reply_text(
            f"🎉 ¡Premium {renewal}! Disfrútalo hasta el {format_until(until)}.\n"
            "Añade valores con /seguir para recibir tu resumen diario a las "
            f"{self.config.daily_summary_time} (días laborables).")

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id
        payment = self.db.last_payment(uid)
        if not payment or not payment["is_recurring"]:
            await update.effective_message.reply_text("No tienes ninguna suscripción activa.")
            return
        try:
            await context.bot.edit_user_star_subscription(uid, payment["charge_id"], is_canceled=True)
        except TelegramError as exc:
            log.warning("No se pudo cancelar la suscripción de %s: %s", uid, exc)
            await update.effective_message.reply_text("No he podido cancelarla. Prueba desde los ajustes de Telegram o /paysupport.")
            return
        until = self.db.premium_until(uid)
        await update.effective_message.reply_text(
            f"Suscripción cancelada: no se volverá a cobrar. Sigues siendo Premium hasta el {format_until(until)}." if until
            else "Suscripción cancelada.")

    async def invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        uid = update.effective_user.id
        link = f"https://t.me/{context.bot.username}?start=ref_{uid}"
        await update.effective_message.reply_text(
            f"🎁 <b>Invita y gana Premium</b>\nPor cada amigo que entre con tu enlace y haga su primera pregunta, "
            f"te regalo <b>{REFERRAL_DAYS} días de Premium</b>.\n\nTu enlace:\n{link}\n\n"
            f"Amigos que ya cuentan: {self.db.rewarded_referrals(uid)}",
            parse_mode=ParseMode.HTML, disable_web_page_preview=True)

    # --- administración -------------------------------------------------------------------

    def _is_admin(self, update: Update) -> bool:
        return update.effective_user.id in self.config.admin_user_ids

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update):
            return
        s = self.db.stats(today())
        await update.effective_message.reply_text(
            "📊 <b>Métricas</b>\n" + "\n".join(f"{k.replace('_', ' ')}: {v}" for k, v in s.items()),
            parse_mode=ParseMode.HTML)

    async def gift(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update):
            return
        try:
            uid, days = int(context.args[0]), int(context.args[1])
        except (IndexError, TypeError, ValueError):
            await update.effective_message.reply_text("Uso: /regalar USER_ID DIAS")
            return
        if not self.db.get_user(uid):
            await update.effective_message.reply_text("Ese usuario no ha usado el bot.")
            return
        until = self.db.add_premium_days(uid, days)
        await update.effective_message.reply_text(f"Premium de {uid} hasta el {format_until(until)}.")

    async def refund(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_admin(update):
            return
        if not context.args:
            await update.effective_message.reply_text("Uso: /reembolsar CHARGE_ID")
            return
        payment = self.db.get_payment(context.args[0])
        if not payment or payment["refunded"]:
            await update.effective_message.reply_text("Pago no encontrado o ya reembolsado.")
            return
        try:
            await context.bot.refund_star_payment(payment["user_id"], payment["charge_id"])
        except TelegramError as exc:
            await update.effective_message.reply_text(f"Telegram rechazó el reembolso: {exc}")
            return
        if payment["is_recurring"]:
            try:
                await context.bot.edit_user_star_subscription(payment["user_id"], payment["charge_id"], is_canceled=True)
            except TelegramError as exc:
                log.warning("Reembolso hecho, pero no se pudo cancelar la suscripción: %s", exc)
        self.db.mark_refunded(payment["charge_id"])
        self.db.set_premium_until(payment["user_id"], datetime.now(timezone.utc))
        await update.effective_message.reply_text("Reembolso hecho y Premium retirado.")

    # --- botones -------------------------------------------------------------------------

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        kind, _, value = (query.data or "").partition(":")
        message = query.message
        if kind == "cmd":
            handlers = {"mercado": self.market, "quiz": self.quiz_command, "premium": self.premium, "invitar": self.invite}
            if value in handlers:
                await handlers[value](update, context)
            return
        if kind == "da":
            ok = self.db.delete_alert(update.effective_user.id, int(value)) if value.isdigit() else False
            await message.reply_text(f"Alerta #{value} borrada." if ok else "Esa alerta ya no existe.")
            return
        if kind == "dd":
            ok = bool(self.kb and value.isdigit() and self.kb.delete_document(update.effective_user.id, int(value)))
            await message.reply_text("Documento borrado." if ok else "Ese documento ya no existe.")
            return

        async def run():
            if kind == "g":
                await self._send_chart(message, [value], "1y")
            elif kind == "f":
                await self._send_company(message, value)
            elif kind == "n":
                await self._send_news(message, value)
            elif kind == "s":
                data = await self._run(tools.seguimiento, "anadir", [value], ctx=self._ctx(update))
                await message.reply_text(f"👀 {value} está en tu lista: {', '.join(data['lista'])}")
        await self._guarded(update, run)

    # --- conversación con la IA ----------------------------------------------------------------

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        await self._ask(update, context, update.effective_message.text)

    async def voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Nota de voz -> texto (Whisper) -> misma conversación que un mensaje escrito."""
        self._register(update)
        message, uid = update.effective_message, update.effective_user.id
        if self.transcriber is None:
            await message.reply_text("Con este proveedor de IA no puedo escuchar audios. Escríbeme la pregunta 🙂")
            return
        media = message.voice or message.audio
        if media.duration and media.duration > MAX_VOICE_SECONDS:
            await message.reply_text(f"El audio es muy largo: mándame uno de menos de {MAX_VOICE_SECONDS} segundos.")
            return
        plan = self.plans.plan_for(uid)
        # Se comprueba el cupo antes de transcribir para no gastar audio en preguntas que no se van a responder.
        if self._throttled(uid):
            await message.reply_text("Vas muy rápido 🙂 Espera unos segundos.")
            return
        if self.db.queries_on(uid, today()) >= plan.ai_queries_per_day:
            await self._ask(update, context, "")  # muestra el mensaje de límite diario
            return
        await update.effective_chat.send_action(ChatAction.TYPING)
        audio = bytes(await (await media.get_file()).download_as_bytearray())
        filename = getattr(media, "file_name", None) or "voz.ogg"
        try:
            text = await asyncio.to_thread(self.transcriber.transcribe, audio, filename)
        except TranscriptionError as exc:
            await message.reply_text(str(exc))
            return
        await message.reply_text(f"🎙️ «{text}»")
        await self._ask(update, context, text)

    async def _ask(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        uid, chat_id = update.effective_user.id, update.effective_chat.id
        allowed, remaining, plan = self.plans.try_consume_query(uid) if text else (False, 0, self.plans.plan_for(uid))
        if not allowed:
            buttons = [[InlineKeyboardButton("⭐ Premium", callback_data="cmd:premium"),
                        InlineKeyboardButton("🎁 Invitar amigos", callback_data="cmd:invitar")]]
            await update.effective_message.reply_text(
                f"Has usado tus {plan.ai_queries_per_day} preguntas a la IA de hoy (plan {plan.name}). "
                "Mañana se renuevan. Mientras, los comandos /cotizacion, /ficha, /grafico y /mercado son ilimitados.",
                reply_markup=InlineKeyboardMarkup(buttons) if plan is not ADMIN else None)
            return

        async with self._typing(update.effective_chat):  # indexar un informe de la SEC puede tardar ~30 s
            reply = await asyncio.to_thread(self.agent.reply, chat_id, text, self._ctx(update))
        if not reply.ok:
            # Si la IA falló (saturada, sin conexión...), la pregunta no cuenta para el límite diario.
            self.db.uncount_query(uid, today())
        for part in formato.split_message(reply.text):
            try:
                await update.effective_message.reply_text(formato.ai_to_html(part), parse_mode=ParseMode.HTML)
            except BadRequest:  # HTML que Telegram no acepta: mejor texto plano que no responder
                await update.effective_message.reply_text(formato.plain(part))
        for png, caption in reply.images:
            await update.effective_message.reply_photo(png, caption=caption)
        if remaining <= 2 and plan.ai_queries_per_day < ADMIN.ai_queries_per_day:
            await update.effective_message.reply_text(f"ℹ️ Te quedan {remaining} preguntas a la IA hoy.")

        reward = await asyncio.to_thread(self.plans.reward_referral, uid)
        if reward:
            referrer, until = reward
            referrer_user = self.db.get_user(referrer)
            if referrer_user:
                try:
                    await context.bot.send_message(
                        referrer_user["chat_id"],
                        f"🎁 Un amigo al que invitaste ya usa el bot: +{REFERRAL_DAYS} días de Premium (hasta el {format_until(until)}).")
                except TelegramError:
                    log.info("No se pudo avisar al usuario %s de su recompensa", referrer)

    # --- tareas programadas ---------------------------------------------------------------

    async def check_alerts(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        alerts = self.db.active_alerts()
        if not alerts:
            return
        prices = await self._run(mercado.last_prices, sorted({a["ticker"] for a in alerts}))
        day = today()
        for a in alerts:
            p = prices.get(a["ticker"])
            if not p:
                continue
            price, prev = p["precio"], p.get("anterior")
            move = (price / prev - 1) * 100 if prev else None
            if a["kind"] == "encima" and price >= a["value"]:
                text = f"🔔 {a['ticker']} ha subido a {formato.n(price)} (alerta: por encima de {formato.n(a['value'])})."
            elif a["kind"] == "debajo" and price <= a["value"]:
                text = f"🔔 {a['ticker']} ha bajado a {formato.n(price)} (alerta: por debajo de {formato.n(a['value'])})."
            elif a["kind"] == "movimiento" and move is not None and abs(move) >= a["value"] and a["last_fired"] != day:
                text = f"🔔 {a['ticker']} se mueve un {formato.change(move)} hoy (precio {formato.n(price)})."
            else:
                continue
            # Las alertas de precio se desactivan al saltar; las de movimiento, una vez al día.
            self.db.fire_alert(a["id"], day, deactivate=a["kind"] != "movimiento")
            try:
                await context.bot.send_message(a["chat_id"], text + "\nPídeme «analiza " + a["ticker"] + "» para ver por qué.",
                                               reply_markup=_ticker_buttons(a["ticker"]))
            except TelegramError as exc:
                log.warning("No se pudo enviar la alerta %s: %s", a["id"], exc)

    async def daily_summary(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        recipients = self.db.premium_watchers()
        admins = [{"user_id": a, "chat_id": u["chat_id"]} for a in self.config.admin_user_ids
                  if (u := self.db.get_user(a)) and (self.db.watchlist(a) or self.db.portfolio_tickers(a))]
        recipients += [r for r in admins if r["user_id"] not in {x["user_id"] for x in recipients}]
        if not recipients:
            return
        market = await self._run(tools.resumen_mercado)
        headline = " · ".join(f"{r['nombre']} {formato.change(r.get('dia_pct'))}" for r in market.get("Índices", [])[:4])
        for r in recipients:
            try:
                uid = r["user_id"]
                parts = [f"☀️ <b>Tu resumen de hoy</b>\n{escape(headline)}"]
                held = self.db.portfolio_tickers(uid)
                if held:
                    total = (await self._run(tools.portfolio_view, self.db, uid)).get("total", {})
                    parts.append(f"💼 Tu cartera: {formato.n(total.get('valor_eur'))} € · hoy "
                                 f"{formato.change(total.get('hoy_eur'), '€')} · total {formato.change(total.get('ganancia_pct'))}")
                watched = self.db.watchlist(uid)
                if watched:
                    data = await self._run(tools.seguimiento, "ver", ctx=ToolContext(uid, r["chat_id"], self.db, self.plans))
                    parts.append(formato.watchlist(data))
                events = await self._upcoming_events(sorted(set(held) | set(watched)))
                if events:
                    parts.append("\n".join(events))
                parts.append("<i>Pregúntame por cualquiera de ellos para un análisis. /cartera para el detalle.</i>")
                await context.bot.send_message(r["chat_id"], "\n\n".join(parts), parse_mode=ParseMode.HTML)
            except Exception:
                log.exception("Fallo en el resumen diario de %s", r["user_id"])

    async def _upcoming_events(self, tickers: list[str], days: int = 7) -> list[str]:
        """Resultados y fechas ex-dividendo de los próximos días para esos valores."""
        start = datetime.now(TZ).date()
        horizon = start + timedelta(days=days)
        events = []
        for t in tickers:
            cal = await self._run(mercado.calendar, t)
            for label, key in (("resultados", "Earnings Date"), ("ex-dividendo", "Ex-Dividend Date")):
                value = cal.get(key)
                value = value[0] if isinstance(value, list) and value else value
                if isinstance(value, date) and start <= value <= horizon:
                    events.append(f"📅 {t}: {label} el {value:%d/%m}")
        return events

    async def channel_post(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Publica el cierre del día en el canal público: contenido gratis que atrae usuarios al bot."""
        market = await self._run(tools.resumen_mercado)
        ctx = ToolContext(user_id=0, chat_id=0)
        await self._run(tools.grafico, ["^IBEX", "^STOXX50E", "^GSPC"], "1mo", ctx=ctx)
        cta = (f"\n\n🤖 ¿Dudas sobre una empresa? Pregúntale a @{context.bot.username}: análisis, gráficos, "
               "alertas y simulaciones gratis.\n<i>Información, no asesoramiento.</i>")
        await context.bot.send_message(self.config.channel_id, formato.market(market) + cta, parse_mode=ParseMode.HTML,
                                       disable_web_page_preview=True)
        for png, caption in ctx.images:
            await context.bot.send_photo(self.config.channel_id, png, caption="Último mes: " + caption)

    async def post_init(self, app: Application) -> None:
        await app.bot.set_my_commands([BotCommand(c, d[:256]) for c, d in COMMANDS])

    async def announcements(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._register(update)
        arg = (context.args or [""])[0].lower()
        if arg not in ("on", "off"):
            await update.effective_message.reply_text("Uso: /avisos on (recibir novedades del bot) o /avisos off")
            return
        self.db.set_announcements(update.effective_user.id, arg == "on")
        await update.effective_message.reply_text("Recibirás los anuncios del bot." if arg == "on"
                                                  else "No volverás a recibir anuncios. Las alertas y resúmenes siguen igual.")

    async def broadcast(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/anuncio TEXTO (solo administradores): envía una novedad a quien no se haya dado de baja."""
        if not self._is_admin(update):
            return
        text = (update.effective_message.text or "").partition(" ")[2].strip()
        if not text:
            await update.effective_message.reply_text("Uso: /anuncio TEXTO")
            return
        body = f"📣 {escape(text)}\n\n<i>/avisos off para no recibir anuncios.</i>"
        sent = failed = 0
        for chat_id in self.db.announcement_chats():
            try:
                await context.bot.send_message(chat_id, body, parse_mode=ParseMode.HTML)
                sent += 1
            except Forbidden:  # el usuario bloqueó el bot: no se le vuelve a intentar
                self.db.set_announcements(chat_id, False)
                failed += 1
            except TelegramError:
                failed += 1
            await asyncio.sleep(BROADCAST_DELAY)
        await update.effective_message.reply_text(f"Anuncio enviado a {sent} usuarios ({failed} fallidos).")

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("Error procesando una actualización", exc_info=context.error)
        now = time.monotonic()
        last = self._last_error_notice
        if not self.config.admin_user_ids or (last is not None and now - last < ERROR_NOTICE_SECONDS):
            return
        self._last_error_notice = now
        text = f"⚠️ Error en el bot: {type(context.error).__name__}: {str(context.error)[:300]}\nDetalles en el registro."
        for admin in self.config.admin_user_ids:
            try:
                await context.bot.send_message(admin, text)
            except TelegramError:
                pass


def build_application(config: Config, agent: Agent, db: Database, transcriber: Transcriber | None = None,
                      kb: KnowledgeBase | None = None, sec: SecClient | None = None) -> Application:
    bot = MarketBot(config, agent, db, transcriber, kb, sec)
    # Si hay lista de usuarios permitidos, el resto se ignora (modo privado / pruebas).
    allowed = filters.User(user_id=config.allowed_user_ids) if config.allowed_user_ids else filters.ALL

    # concurrent_updates: sin esto, una pregunta lenta a la IA bloquearía al resto de usuarios.
    app = Application.builder().token(config.telegram_token).concurrent_updates(16).post_init(bot.post_init).build()

    commands = {
        ("start",): bot.start, ("ayuda", "help"): bot.help, ("terms", "terminos", "privacidad"): bot.terms,
        ("paysupport", "support", "soporte"): bot.support, ("reset",): bot.reset, ("mercado",): bot.market,
        ("cotizacion", "precio"): bot.quote, ("ficha",): bot.company, ("grafico",): bot.chart,
        ("noticias",): bot.news, ("lista",): bot.watchlist, ("seguir",): bot.follow,
        ("dejar",): partial(bot.follow, action="quitar"), ("alertas",): bot.alerts, ("alerta",): bot.new_alert,
        ("quiz",): bot.quiz_command, ("ranking",): bot.ranking, ("cuenta",): bot.account,
        ("premium",): bot.premium, ("cancelar",): bot.cancel, ("invitar",): bot.invite,
        ("stats",): bot.stats, ("regalar",): bot.gift, ("reembolsar",): bot.refund,
        ("cartera",): bot.portfolio, ("compra", "comprar"): bot.trade,
        ("venta", "vender"): partial(bot.trade, action="vender"), ("top", "buscar"): bot.top,
        ("avisos",): bot.announcements, ("anuncio",): bot.broadcast, ("documentos", "docs"): bot.documents,
    }
    for names, handler in commands.items():
        app.add_handler(CommandHandler(list(names), handler, filters=allowed))
    app.add_handler(CallbackQueryHandler(bot.button))
    app.add_handler(InlineQueryHandler(bot.inline_query))
    app.add_handler(MessageHandler((filters.VOICE | filters.AUDIO) & allowed, bot.voice))
    app.add_handler(MessageHandler(filters.Document.PDF & allowed, bot.document))
    app.add_handler(PollAnswerHandler(bot.quiz_answer))
    app.add_handler(PreCheckoutQueryHandler(bot.precheckout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, bot.paid))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & allowed, bot.chat))
    app.add_error_handler(bot.on_error)

    hour, minute = (int(x) for x in config.daily_summary_time.split(":"))
    app.job_queue.run_repeating(bot.check_alerts, interval=ALERT_CHECK_SECONDS, first=30, name="alertas")
    # days: 0 = domingo ... 6 = sábado (convención de python-telegram-bot); 1-5 = lunes a viernes.
    app.job_queue.run_daily(bot.daily_summary, time=dtime(hour, minute, tzinfo=TZ), days=(1, 2, 3, 4, 5), name="resumen")
    if config.channel_id:
        app.job_queue.run_daily(bot.channel_post, time=dtime(hour, minute, tzinfo=TZ), days=(1, 2, 3, 4, 5), name="canal")
    return app
