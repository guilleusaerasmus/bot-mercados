"""Pruebas de la lectura de configuración (sin leer el .env real)."""

import pytest

from bot_mercados import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_MODELS", "LLM_REASONING_EFFORT", "TELEGRAM_TOKEN",
                 "ALLOWED_USER_IDS", "ADMIN_USER_IDS", "DB_PATH", "PREMIUM_PRICE_STARS", "DAILY_SUMMARY_TIME", "CHANNEL_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "clave")


def test_valores_por_defecto():
    c = config.load_config()
    assert c.provider == "groq" and c.models[0] == "openai/gpt-oss-120b" and len(c.models) == 3
    assert c.reasoning_effort == "low" and c.premium_price_stars == 250 and c.channel_id is None
    assert c.db_path.name == "bot.db"


def test_xai_exige_modelo(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    with pytest.raises(config.ConfigError):
        config.load_config()
    monkeypatch.setenv("LLM_MODELS", "grok-x")
    c = config.load_config()
    assert c.models == ("grok-x",) and c.reasoning_effort is None and c.base_url == "https://api.x.ai/v1"


@pytest.mark.parametrize("name,value", [
    ("ADMIN_USER_IDS", "12,abc"), ("PREMIUM_PRICE_STARS", "0"), ("PREMIUM_PRICE_STARS", "mucho"),
    ("DAILY_SUMMARY_TIME", "25:00"), ("LLM_REASONING_EFFORT", "max"), ("LLM_PROVIDER", "openai"),
])
def test_valores_invalidos(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_falta_clave(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(config.ConfigError, match="LLM_API_KEY"):
        config.load_config()
