"""Reglas del modelo de negocio: qué incluye cada plan y cómo se consumen los límites.

Las consultas a la IA cuestan dinero (tokens). Los comandos directos (/cotizacion, /grafico,
/mercado...) no usan la IA, así que son ilimitados en los dos planes: dan valor gratis sin coste
variable y la IA queda como el gancho para pasar a Premium.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .db import Database

TZ = ZoneInfo("Europe/Madrid")
REFERRAL_DAYS = 3  # días de Premium por cada amigo invitado que use el bot
REFERRAL_MAX_REWARDS = 10  # tope de amigos recompensados por persona


@dataclass(frozen=True)
class Plan:
    name: str
    ai_queries_per_day: int
    alerts: int
    watchlist: int
    positions: int
    documents: int
    daily_summary: bool


FREE = Plan("Gratis", ai_queries_per_day=10, alerts=3, watchlist=5, positions=5, documents=3, daily_summary=False)
PREMIUM = Plan("Premium", ai_queries_per_day=60, alerts=30, watchlist=50, positions=50, documents=30, daily_summary=True)
ADMIN = Plan("Admin", ai_queries_per_day=10_000, alerts=1_000, watchlist=1_000, positions=1_000, documents=1_000, daily_summary=True)


def today() -> str:
    """El día cambia a medianoche de España, no de UTC."""
    return datetime.now(TZ).strftime("%Y-%m-%d")


class Plans:
    def __init__(self, db: Database, admin_ids: frozenset[int] = frozenset()):
        self.db = db
        self.admin_ids = admin_ids

    def plan_for(self, user_id: int) -> Plan:
        if user_id in self.admin_ids:
            return ADMIN
        return PREMIUM if self.db.is_premium(user_id) else FREE

    def try_consume_query(self, user_id: int) -> tuple[bool, int, Plan]:
        """Cuenta una consulta a la IA si quedan. Devuelve (permitida, restantes hoy, plan)."""
        plan = self.plan_for(user_id)
        day = today()
        if self.db.queries_on(user_id, day) >= plan.ai_queries_per_day:
            return False, 0, plan
        used = self.db.count_query(user_id, day)
        return True, max(plan.ai_queries_per_day - used, 0), plan

    def reward_referral(self, user_id: int) -> tuple[int, datetime] | None:
        """Si el usuario vino invitado y es su primera consulta, regala días al que le invitó.

        Se recompensa al usar el bot (no al entrar) para que las cuentas falsas no sirvan de nada.
        """
        user = self.db.get_user(user_id)
        if not user or not user["referred_by"] or user["referral_rewarded"]:
            return None
        referrer = user["referred_by"]
        if not self.db.get_user(referrer) or self.db.rewarded_referrals(referrer) >= REFERRAL_MAX_REWARDS:
            return None
        if self.db.mark_referral_rewarded(user_id) is None:
            return None
        return referrer, self.db.add_premium_days(referrer, REFERRAL_DAYS)


def format_until(dt: datetime) -> str:
    return dt.astimezone(TZ).strftime("%d/%m/%Y")


def utc_from_timestamp(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)
