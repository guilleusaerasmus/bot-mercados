"""Punto de entrada.

    python -m bot_mercados consola              -> chatea con el agente en la terminal (no necesita Telegram)
    python -m bot_mercados telegram             -> arranca el bot de Telegram
    python -m bot_mercados indexar a.pdf b.pdf  -> añade PDFs a los documentos de la consola (RAG)
    python -m bot_mercados evaluar-rag          -> mide la calidad de la búsqueda en documentos
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

from .config import ConfigError, data_dir, load_config, models_dir
from .rag import Embedder, KnowledgeBase

CONSOLE_USER = 0


def build_knowledge_base(folder: Path, use_embeddings: bool = True) -> KnowledgeBase:
    embedder = Embedder(cache_dir=models_dir()) if use_embeddings and Embedder.available() else None
    return KnowledgeBase(folder / "documentos.db", embedder)


def run_console(agent, db, admin_ids: frozenset[int], charts_dir: Path, kb, sec) -> None:
    from .planes import Plans
    from .tools import ToolContext

    db.ensure_user(CONSOLE_USER, CONSOLE_USER, "Consola")
    plans = Plans(db, admin_ids | {CONSOLE_USER})  # en consola no hay límites
    print("Bot de mercados en consola. Escribe 'salir' para terminar, 'reset' para borrar la conversación.")
    print(f"Documentos para preguntar: {len(kb.documents(CONSOLE_USER))} (añade PDFs con: python -m bot_mercados indexar archivo.pdf)")
    while True:
        try:
            text = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"salir", "exit", "quit"}:
            return
        if text.lower() == "reset":
            agent.reset(CONSOLE_USER)
            print("Conversación borrada.")
            continue
        if not text:
            continue
        ctx = ToolContext(user_id=CONSOLE_USER, chat_id=CONSOLE_USER, db=db, plans=plans, kb=kb, sec=sec)
        reply = agent.reply(CONSOLE_USER, text, ctx)
        print(f"\nBot: {reply.text}")
        for i, (png, caption) in enumerate(reply.images):
            charts_dir.mkdir(parents=True, exist_ok=True)
            path = charts_dir / f"grafico_{datetime.now():%Y%m%d_%H%M%S}_{i}.png"
            path.write_bytes(png)
            print(f"[Gráfico «{caption}» guardado en {path}]")


def index_files(paths: list[str]) -> int:
    from .documentos import DocumentError, pdf_sections

    kb = build_knowledge_base(data_dir())
    print(f"Búsqueda {kb.mode}. Indexando {len(paths)} archivo(s)...")
    errors = 0
    for raw in paths:
        path = Path(raw)
        try:
            sections, pages = pdf_sections(path.read_bytes())
            kb.add_document(path.stem, sections, "pdf", owner_id=CONSOLE_USER)
            print(f"  OK    {path.name}: {pages} páginas")
        except (OSError, DocumentError, ValueError) as exc:
            print(f"  ERROR {path.name}: {exc}", file=sys.stderr)
            errors += 1
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bot_mercados", description="Bot de mercados con IA (Groq/Grok).")
    parser.add_argument("modo", choices=["consola", "telegram", "indexar", "evaluar-rag"])
    parser.add_argument("archivos", nargs="*", help="PDFs para el modo indexar")
    parser.add_argument("-v", "--verbose", action="store_true", help="muestra qué herramientas usa el agente")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose or args.modo == "telegram" else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx registra cada petición (con el token de Telegram en la URL): lo silenciamos.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    # yfinance avisa de cada ticker sin datos y de deprecaciones de pandas; esos casos
    # ya los gestionan las herramientas, así que no ensuciamos la consola.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    # yfinance instala al importarse su propio filtro («mostrar DeprecationWarning») delante de los
    # demás, y pandas 3.1 avisa con Pandas4Warning (un DeprecationWarning): hay que importarlo antes
    # de añadir el nuestro para que el nuestro quede primero y gane.
    from . import mercado  # noqa: F401
    warnings.filterwarnings("ignore", module=r"yfinance(\..*)?$")

    if args.modo == "evaluar-rag":
        from . import rag_eval
        rag_eval.main(Embedder(cache_dir=models_dir()) if Embedder.available() else None)
        return 0
    if args.modo == "indexar":
        if not args.archivos:
            parser.error("indica al menos un PDF: python -m bot_mercados indexar informe.pdf")
        return index_files(args.archivos)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return 1

    import openai

    from .agent import Agent
    from .db import Database
    from .documentos import SecClient

    client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=60, max_retries=1)
    agent = Agent(client, config.models, config.reasoning_effort)
    db = Database(config.db_path)
    kb = build_knowledge_base(config.db_path.parent, config.rag_embeddings)
    sec = SecClient(config.sec_user_agent) if config.sec_user_agent else None

    if args.modo == "consola":
        run_console(agent, db, config.admin_user_ids, config.db_path.parent / "graficos", kb, sec)
        return 0

    if not config.telegram_token:
        print("Error de configuración: falta TELEGRAM_TOKEN en el archivo .env.", file=sys.stderr)
        return 1
    from telegram import Update

    from .telegram_bot import build_application
    from .voz import Transcriber

    transcriber = Transcriber(client, config.stt_model) if config.stt_model else None
    logging.info("Bot arrancado con %s / %s (voz: %s, documentos: %s, SEC: %s). Base de datos: %s. Ctrl+C para parar.",
                 config.provider, ", ".join(config.models), config.stt_model or "no", kb.mode,
                 "sí" if sec else "no", config.db_path)
    build_application(config, agent, db, transcriber, kb, sec).run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
