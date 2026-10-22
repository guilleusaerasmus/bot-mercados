"""Configuración leída de variables de entorno (archivo .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Ambos proveedores exponen una API compatible con la de OpenAI:
# cambiar de uno a otro es cambiar la URL base, la clave y el modelo.
PROVIDERS = {
    # En Groq cada modelo tiene su propio cupo gratuito: si uno se agota, se pasa al siguiente.
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "default_models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
             "stt_model": "whisper-large-v3-turbo"},
    "xai": {"base_url": "https://api.x.ai/v1", "default_models": [], "stt_model": None},
}
REASONING_EFFORTS = {"low", "medium", "high"}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    provider: str
    base_url: str
    api_key: str
    models: tuple[str, ...]
    reasoning_effort: str | None
    telegram_token: str | None
    allowed_user_ids: frozenset[int]
    admin_user_ids: frozenset[int]
    db_path: Path
    premium_price_stars: int
    support_contact: str
    daily_summary_time: str
    channel_id: str | None = None
    stt_model: str | None = None  # modelo de voz a texto; None = notas de voz desactivadas
    sec_user_agent: str | None = None  # identificación ante la SEC; None = sin informes 10-K
    rag_embeddings: bool = True  # False = RAG solo con palabras clave (BM25)


def _ids(name: str) -> frozenset[int]:
    raw = os.getenv(name, "")
    try:
        return frozenset(int(x) for x in raw.replace(" ", "").split(",") if x)
    except ValueError as exc:
        raise ConfigError(f"{name} debe ser una lista de números separados por comas.") from exc


def data_dir() -> Path:
    """Carpeta de datos (base de datos, documentos y modelos), sin exigir el resto de la configuración."""
    load_dotenv()
    return Path(os.getenv("DB_PATH", "").strip() or Path.home() / ".bot-mercados" / "bot.db").parent


def models_dir() -> Path:
    """Caché de modelos de embeddings (~250 MB), compartida aunque cambie la base de datos."""
    load_dotenv()
    return Path(os.getenv("MODELS_DIR", "").strip() or Path.home() / ".bot-mercados" / "modelos")


def load_config() -> Config:
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER debe ser uno de {sorted(PROVIDERS)}, no '{provider}'.")

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("Falta LLM_API_KEY en el archivo .env.")

    raw_models = os.getenv("LLM_MODELS", "") or os.getenv("LLM_MODEL", "")
    models = tuple(m.strip() for m in raw_models.split(",") if m.strip()) or tuple(PROVIDERS[provider]["default_models"])
    if not models:
        raise ConfigError(f"Con LLM_PROVIDER={provider} hay que indicar LLM_MODELS en el .env.")

    effort = os.getenv("LLM_REASONING_EFFORT", "low").strip().lower() or None
    if effort and effort not in REASONING_EFFORTS:
        raise ConfigError(f"LLM_REASONING_EFFORT debe ser uno de {sorted(REASONING_EFFORTS)}.")

    try:
        price = int(os.getenv("PREMIUM_PRICE_STARS", "250"))
    except ValueError as exc:
        raise ConfigError("PREMIUM_PRICE_STARS debe ser un número entero.") from exc
    if not 1 <= price <= 10_000:
        raise ConfigError("PREMIUM_PRICE_STARS debe estar entre 1 y 10000.")

    summary_time = os.getenv("DAILY_SUMMARY_TIME", "18:00").strip()
    try:
        hour, minute = (int(x) for x in summary_time.split(":"))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError
    except ValueError as exc:
        raise ConfigError("DAILY_SUMMARY_TIME debe tener el formato HH:MM.") from exc

    return Config(
        provider=provider,
        base_url=PROVIDERS[provider]["base_url"],
        api_key=api_key,
        models=models,
        # reasoning_effort solo existe en los modelos gpt-oss de Groq.
        reasoning_effort=effort if provider == "groq" else None,
        telegram_token=os.getenv("TELEGRAM_TOKEN", "").strip() or None,
        allowed_user_ids=_ids("ALLOWED_USER_IDS"),
        admin_user_ids=_ids("ADMIN_USER_IDS"),
        db_path=Path(os.getenv("DB_PATH", "").strip() or Path.home() / ".bot-mercados" / "bot.db"),
        premium_price_stars=price,
        support_contact=os.getenv("SUPPORT_CONTACT", "").strip(),
        daily_summary_time=summary_time,
        channel_id=os.getenv("CHANNEL_ID", "").strip() or None,
        stt_model=os.getenv("STT_MODEL", "").strip() or PROVIDERS[provider]["stt_model"],
        sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip() or None,
        rag_embeddings=os.getenv("RAG_EMBEDDINGS", "true").strip().lower() not in ("false", "0", "no", "off"),
    )
