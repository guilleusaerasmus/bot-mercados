"""Base de datos SQLite: usuarios, planes, uso diario, alertas, lista de seguimiento y pagos."""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .analisis import fifo_sell

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    name TEXT,
    created_at TEXT NOT NULL,
    premium_until TEXT,
    referred_by INTEGER,
    referral_rewarded INTEGER NOT NULL DEFAULT 0,
    quiz_ok INTEGER NOT NULL DEFAULT 0,
    quiz_total INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    user_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    queries INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('encima', 'debajo', 'movimiento')),
    value REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_fired TEXT,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (user_id, ticker)
);
CREATE TABLE IF NOT EXISTS payments (
    charge_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    currency TEXT NOT NULL,
    payload TEXT,
    is_recurring INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT,
    created_at TEXT NOT NULL,
    refunded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS alerts_active ON alerts (active);
CREATE TABLE IF NOT EXISTS lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    shares REAL NOT NULL CHECK (shares > 0),
    price REAL NOT NULL CHECK (price > 0),
    currency TEXT NOT NULL,
    bought_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lots_user ON lots (user_id, ticker, bought_at);
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    shares REAL NOT NULL,
    sell_price REAL NOT NULL,
    gain REAL NOT NULL,
    currency TEXT NOT NULL,
    gain_eur REAL NOT NULL,
    sold_at TEXT NOT NULL
);
"""

# Columnas añadidas después de la primera versión: se crean al arrancar si faltan, para que
# una base de datos con usuarios y pagos reales se actualice sin perder nada.
MIGRATIONS = {
    "users": {"announcements": "INTEGER NOT NULL DEFAULT 1"},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Database:
    """Una conexión compartida protegida con un cerrojo: el bot la usa desde varios hilos."""

    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            for table, columns in MIGRATIONS.items():
                existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
                for name, definition in columns.items():
                    if name not in existing:
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _exec(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock, self._conn:
            return self._conn.execute(sql, params)

    def _all(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params=()) -> dict | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def _scalar(self, sql: str, params=()) -> int:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()[0]

    # --- usuarios y planes ---

    def ensure_user(self, user_id: int, chat_id: int, name: str | None, referred_by: int | None = None) -> bool:
        """Registra al usuario si es nuevo. Devuelve True si acaba de darse de alta."""
        if referred_by == user_id:
            referred_by = None
        cur = self._exec(
            "INSERT OR IGNORE INTO users (user_id, chat_id, name, created_at, referred_by) VALUES (?, ?, ?, ?, ?)",
            (user_id, chat_id, name, _iso(_now()), referred_by),
        )
        if cur.rowcount == 0:
            self._exec("UPDATE users SET chat_id = ?, name = ? WHERE user_id = ?", (chat_id, name, user_id))
            return False
        return True

    def get_user(self, user_id: int) -> dict | None:
        return self._one("SELECT * FROM users WHERE user_id = ?", (user_id,))

    def premium_until(self, user_id: int) -> datetime | None:
        user = self.get_user(user_id)
        return _parse(user["premium_until"]) if user else None

    def is_premium(self, user_id: int, now: datetime | None = None) -> bool:
        until = self.premium_until(user_id)
        return bool(until and until > (now or _now()))

    def set_premium_until(self, user_id: int, until: datetime) -> None:
        self._exec("UPDATE users SET premium_until = ? WHERE user_id = ?", (_iso(until), user_id))

    def add_premium_days(self, user_id: int, days: int) -> datetime:
        """Suma días a partir del final del premium actual (o de ahora si no tiene)."""
        now = _now()
        current = self.premium_until(user_id)
        until = max(current, now) if current else now
        until = (until + timedelta(days=days)).replace(microsecond=0)  # se guarda con precisión de segundos
        self.set_premium_until(user_id, until)
        return until

    def mark_referral_rewarded(self, user_id: int) -> int | None:
        """Marca la invitación como recompensada (una sola vez) y devuelve quién invitó."""
        cur = self._exec(
            "UPDATE users SET referral_rewarded = 1 WHERE user_id = ? AND referred_by IS NOT NULL AND referral_rewarded = 0",
            (user_id,),
        )
        if cur.rowcount == 0:
            return None
        return self._scalar("SELECT referred_by FROM users WHERE user_id = ?", (user_id,))

    def rewarded_referrals(self, user_id: int) -> int:
        return self._scalar("SELECT COUNT(*) FROM users WHERE referred_by = ? AND referral_rewarded = 1", (user_id,))

    # --- uso diario ---

    def count_query(self, user_id: int, day: str) -> int:
        self._exec(
            "INSERT INTO usage (user_id, day, queries) VALUES (?, ?, 1) "
            "ON CONFLICT (user_id, day) DO UPDATE SET queries = queries + 1",
            (user_id, day),
        )
        return self.queries_on(user_id, day)

    def uncount_query(self, user_id: int, day: str) -> None:
        self._exec("UPDATE usage SET queries = MAX(queries - 1, 0) WHERE user_id = ? AND day = ?", (user_id, day))

    def queries_on(self, user_id: int, day: str) -> int:
        row = self._one("SELECT queries FROM usage WHERE user_id = ? AND day = ?", (user_id, day))
        return row["queries"] if row else 0

    def total_queries(self, user_id: int) -> int:
        return self._scalar("SELECT COALESCE(SUM(queries), 0) FROM usage WHERE user_id = ?", (user_id,))

    # --- alertas ---

    def add_alert(self, user_id: int, chat_id: int, ticker: str, kind: str, value: float) -> int:
        cur = self._exec(
            "INSERT INTO alerts (user_id, chat_id, ticker, kind, value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, chat_id, ticker, kind, value, _iso(_now())),
        )
        return int(cur.lastrowid or 0)

    def user_alerts(self, user_id: int) -> list[dict]:
        return self._all("SELECT * FROM alerts WHERE user_id = ? AND active = 1 ORDER BY id", (user_id,))

    def active_alerts(self) -> list[dict]:
        return self._all("SELECT * FROM alerts WHERE active = 1")

    def delete_alert(self, user_id: int, alert_id: int) -> bool:
        cur = self._exec("UPDATE alerts SET active = 0 WHERE id = ? AND user_id = ? AND active = 1", (alert_id, user_id))
        return cur.rowcount > 0

    def fire_alert(self, alert_id: int, day: str, deactivate: bool) -> None:
        self._exec("UPDATE alerts SET last_fired = ?, active = ? WHERE id = ?", (day, 0 if deactivate else 1, alert_id))

    # --- lista de seguimiento ---

    def watch(self, user_id: int, ticker: str) -> bool:
        cur = self._exec("INSERT OR IGNORE INTO watchlist (user_id, ticker, added_at) VALUES (?, ?, ?)",
                         (user_id, ticker, _iso(_now())))
        return cur.rowcount > 0

    def unwatch(self, user_id: int, ticker: str) -> bool:
        return self._exec("DELETE FROM watchlist WHERE user_id = ? AND ticker = ?", (user_id, ticker)).rowcount > 0

    def watchlist(self, user_id: int) -> list[str]:
        return [r["ticker"] for r in self._all("SELECT ticker FROM watchlist WHERE user_id = ? ORDER BY added_at", (user_id,))]

    def premium_watchers(self, now: datetime | None = None) -> list[dict]:
        """Usuarios premium con algo en su lista o en su cartera (destinatarios del resumen diario)."""
        return self._all(
            "SELECT user_id, chat_id FROM users WHERE premium_until > ? AND ("
            "user_id IN (SELECT user_id FROM watchlist) OR user_id IN (SELECT user_id FROM lots)) ORDER BY user_id",
            (_iso(now or _now()),),
        )

    # --- cartera personal (lotes de compra y ventas realizadas) ---

    def add_lot(self, user_id: int, ticker: str, shares: float, price: float, currency: str,
                bought_at: datetime | None = None) -> int:
        cur = self._exec(
            "INSERT INTO lots (user_id, ticker, shares, price, currency, bought_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, ticker, shares, price, currency, _iso(bought_at or _now())),
        )
        return int(cur.lastrowid or 0)

    def lots(self, user_id: int, ticker: str | None = None) -> list[dict]:
        """Lotes con acciones, del más antiguo al más reciente (orden FIFO)."""
        if ticker:
            return self._all("SELECT * FROM lots WHERE user_id = ? AND ticker = ? ORDER BY bought_at, id", (user_id, ticker))
        return self._all("SELECT * FROM lots WHERE user_id = ? ORDER BY ticker, bought_at, id", (user_id,))

    def portfolio_tickers(self, user_id: int) -> list[str]:
        return [r["ticker"] for r in self._all("SELECT DISTINCT ticker FROM lots WHERE user_id = ? ORDER BY ticker", (user_id,))]

    def sell(self, user_id: int, ticker: str, shares: float, sell_price: float, currency: str,
             eur_per_unit: float) -> float:
        """Vende por FIFO en una sola transacción (lee y actualiza los lotes bajo el mismo cerrojo).

        Devuelve la ganancia en la moneda del valor. Lanza ValueError si no hay acciones suficientes.
        """
        with self._lock, self._conn:
            lots = [dict(r) for r in self._conn.execute(
                "SELECT id, shares, price FROM lots WHERE user_id = ? AND ticker = ? ORDER BY bought_at, id",
                (user_id, ticker)).fetchall()]
            lot_updates, gain = fifo_sell(lots, shares, sell_price)
            for lot_id, left in lot_updates:
                if left <= 1e-9:
                    self._conn.execute("DELETE FROM lots WHERE id = ? AND user_id = ?", (lot_id, user_id))
                else:
                    self._conn.execute("UPDATE lots SET shares = ? WHERE id = ? AND user_id = ?", (left, lot_id, user_id))
            self._conn.execute(
                "INSERT INTO sales (user_id, ticker, shares, sell_price, gain, currency, gain_eur, sold_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, ticker, shares, sell_price, gain, currency, gain * eur_per_unit, _iso(_now())),
            )
        return gain

    def sales_in_year(self, user_id: int, year: int) -> list[dict]:
        return self._all("SELECT * FROM sales WHERE user_id = ? AND sold_at >= ? AND sold_at < ? ORDER BY sold_at",
                         (user_id, f"{year}-01-01", f"{year + 1}-01-01"))

    # --- anuncios ---

    def set_announcements(self, user_id: int, enabled: bool) -> None:
        self._exec("UPDATE users SET announcements = ? WHERE user_id = ?", (int(enabled), user_id))

    def announcement_chats(self) -> list[int]:
        return [r["chat_id"] for r in self._all("SELECT chat_id FROM users WHERE announcements = 1")]

    # --- quiz ---

    def record_quiz(self, user_id: int, correct: bool) -> None:
        self._exec("UPDATE users SET quiz_total = quiz_total + 1, quiz_ok = quiz_ok + ? WHERE user_id = ?",
                   (1 if correct else 0, user_id))

    def quiz_ranking(self, limit: int = 10) -> list[dict]:
        return self._all(
            "SELECT name, quiz_ok, quiz_total FROM users WHERE quiz_total > 0 ORDER BY quiz_ok DESC, quiz_total ASC LIMIT ?",
            (limit,),
        )

    # --- pagos ---

    def record_payment(self, charge_id: str, user_id: int, amount: int, currency: str, payload: str,
                       is_recurring: bool, expires_at: datetime | None) -> bool:
        """Guarda el pago. Devuelve False si ya estaba (Telegram puede reenviar un mismo aviso)."""
        cur = self._exec(
            "INSERT OR IGNORE INTO payments (charge_id, user_id, amount, currency, payload, is_recurring, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (charge_id, user_id, amount, currency, payload, int(is_recurring),
             _iso(expires_at) if expires_at else None, _iso(_now())),
        )
        return cur.rowcount > 0

    def get_payment(self, charge_id: str) -> dict | None:
        return self._one("SELECT * FROM payments WHERE charge_id = ?", (charge_id,))

    def last_payment(self, user_id: int) -> dict | None:
        return self._one("SELECT * FROM payments WHERE user_id = ? AND refunded = 0 ORDER BY created_at DESC LIMIT 1",
                         (user_id,))

    def mark_refunded(self, charge_id: str) -> None:
        self._exec("UPDATE payments SET refunded = 1 WHERE charge_id = ?", (charge_id,))

    # --- métricas de negocio ---

    def stats(self, today: str) -> dict:
        now = _iso(_now())
        week_ago = (_now() - timedelta(days=7)).strftime("%Y-%m-%d")
        q = self._scalar
        return {
            "usuarios": q("SELECT COUNT(*) FROM users"),
            "nuevos_7d": q("SELECT COUNT(*) FROM users WHERE created_at >= ?", (week_ago,)),
            "activos_hoy": q("SELECT COUNT(*) FROM usage WHERE day = ?", (today,)),
            "activos_7d": q("SELECT COUNT(DISTINCT user_id) FROM usage WHERE day >= ?", (week_ago,)),
            "consultas_hoy": q("SELECT COALESCE(SUM(queries), 0) FROM usage WHERE day = ?", (today,)),
            "premium_activos": q("SELECT COUNT(*) FROM users WHERE premium_until > ?", (now,)),
            "stars_cobradas": q("SELECT COALESCE(SUM(amount), 0) FROM payments WHERE refunded = 0 AND currency = 'XTR'"),
            "pagos": q("SELECT COUNT(*) FROM payments WHERE refunded = 0"),
            "alertas_activas": q("SELECT COUNT(*) FROM alerts WHERE active = 1"),
            "invitados": q("SELECT COUNT(*) FROM users WHERE referred_by IS NOT NULL"),
        }
