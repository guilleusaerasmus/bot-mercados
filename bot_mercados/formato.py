"""Mensajes en HTML de Telegram: comandos directos y respuestas de la IA."""

import re
from html import escape

from .planes import FREE, PREMIUM, Plan


def ai_to_html(text: str) -> str:
    """El markdown sencillo que escriben los modelos -> HTML de Telegram.

    Los modelos usan **negrita** aunque se les pida texto plano; en Telegram sin formato se verían
    los asteriscos. Primero se escapa todo (el texto de la IA nunca se interpreta como HTML).
    """
    html = escape(text)
    html = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", html, flags=re.M)
    html = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", html)
    html = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", html)
    return re.sub(r"^(\s*)[*\-•]\s+", r"\1• ", html, flags=re.M)


def plain(text: str) -> str:
    """Versión sin marcas, por si Telegram rechaza el HTML."""
    return re.sub(r"\*\*|`|^#{1,6}\s*", "", text, flags=re.M)


def split_message(text: str, limit: int = 3500) -> list[str]:
    """Trocea por párrafos para no partir una etiqueta HTML por la mitad (Telegram admite 4096)."""
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        while len(paragraph) > limit:
            parts.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        if current and len(current) + 2 + len(paragraph) > limit:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def n(value, digits: int = 2) -> str:
    """Número a la española: 1.234,56."""
    if value is None:
        return "—"
    text = f"{value:,.{digits}f}"
    return text.replace(",", "·").replace(".", ",").replace("·", ".")


def big(value) -> str:
    if value is None:
        return "—"
    for limit, suffix in ((1e12, " billones"), (1e9, " mil M"), (1e6, " M")):
        if abs(value) >= limit:
            return n(value / limit, 1) + suffix
    return n(value, 0)


def change(value, unit: str = "%") -> str:
    if value is None:
        return "—"
    icon = "🟢" if value > 0 else "🔴" if value < 0 else "⚪"
    return f"{icon} {'+' if value > 0 else ''}{n(value)} {unit}"


def quotes(data: dict) -> str:
    lines = []
    for q in data["cotizaciones"]:
        if "error" in q:
            lines.append(f"⚠️ <b>{escape(q['ticker'])}</b>: {escape(q['error'])}")
            continue
        lines.append(
            f"<b>{escape(q.get('nombre') or q['ticker'])}</b> ({escape(q['ticker'])})\n"
            f"💰 <b>{n(q.get('precio'))} {escape(q.get('moneda', ''))}</b>  ·  hoy {change(q.get('dia_pct'))}\n"
            f"1 mes {change(q.get('1m_pct'))}  ·  año {change(q.get('ano_actual_pct'))}  ·  12 m {change(q.get('1a_pct'))}\n"
            f"<i>A {n(abs(q.get('vs_maximo_52s_pct') or 0))} % de su máximo anual · {q.get('fecha')}</i>"
        )
    return "\n\n".join(lines)


def market(data: dict) -> str:
    parts = [f"🌍 <b>Mercados</b> · {data.get('fecha', '')}"]
    for group, rows in data.items():
        if group == "fecha" or not rows:
            continue
        parts.append(f"\n<b>{escape(group)}</b>")
        for r in rows:
            if "dia_puntos" in r:
                parts.append(f"{escape(r['nombre'])}: {n(r['valor'])}  ({change(r.get('dia_puntos'), 'pts')})")
            else:
                parts.append(f"{escape(r['nombre'])}: {n(r['valor'])}  {change(r.get('dia_pct'))}")
    return "\n".join(parts)


def _row(label: str, value: str) -> str:
    return f"• {label}: {value}"


def company(d: dict) -> str:
    e = d.get("empresa", {})
    v, r, s = d.get("valoracion", {}), d.get("rentabilidad", {}), d.get("salud", {})
    c, dv, a, t = d.get("crecimiento", {}), d.get("dividendo", {}), d.get("analistas", {}), d.get("tecnico", {})
    cur = escape(d.get("moneda", ""))
    out = [f"🏢 <b>{escape(e.get('nombre') or e.get('ticker', ''))}</b> ({escape(e.get('ticker', ''))})",
           f"<i>{escape(' · '.join(x for x in (e.get('sector'), e.get('pais')) if x))}</i>",
           f"💰 {n(d.get('precio'))} {cur} · Capitalización {big(d.get('capitalizacion'))} {cur}"]
    if v:
        out += ["\n<b>Valoración</b>", _row("PER", f"{n(v.get('per'), 1)} (estimado {n(v.get('per_estimado'), 1)})"),
                _row("EV/EBITDA", n(v.get("ev_ebitda"), 1)), _row("Precio/valor contable", n(v.get("precio_valor_contable"), 2))]
    if r:
        out += ["\n<b>Rentabilidad del negocio</b>", _row("Margen neto", f"{n(r.get('margen_neto_pct'), 1)} %"),
                _row("ROE", f"{n(r.get('roe_pct'), 1)} %")]
    if s.get("deuda_neta") is not None:
        if s["deuda_neta"] < 0:  # más caja que deuda: mejor decirlo así que con un número negativo
            out += ["\n<b>Deuda</b>", _row("Caja neta", f"{big(-s['deuda_neta'])} {cur} (tiene más caja que deuda)")]
        else:
            ratio = f" ({n(s['deuda_neta_ebitda'], 1)}x EBITDA)" if s.get("deuda_neta_ebitda") is not None else ""
            out += ["\n<b>Deuda</b>", _row("Deuda neta", f"{big(s['deuda_neta'])} {cur}{ratio}")]
    if c:
        out += ["\n<b>Crecimiento</b>", _row("Ventas", f"{n(c.get('ingresos_pct'), 1)} %"),
                _row("Beneficio", f"{n(c.get('beneficios_pct'), 1)} %")]
    if dv.get("rentabilidad_12m_pct"):
        out += ["\n<b>Dividendo</b>",
                _row("Rentabilidad 12 meses", f"{n(dv['rentabilidad_12m_pct'])} % ({n(dv.get('pagado_12m_por_accion'), 3)} {cur} "
                                              f"en {dv.get('pagos_12m', 0)} pagos)"),
                _row("Payout", f"{n(dv.get('payout_pct'), 0)} %")]
        if dv.get("proximo_ex_dividendo"):
            out.append(_row("Próximo ex-dividendo", escape(dv["proximo_ex_dividendo"])))
    if a.get("numero"):
        out += ["\n<b>Analistas</b>", _row("Consenso", f"{escape(str(a.get('recomendacion')))} ({a['numero']} analistas)"),
                _row("Precio objetivo", f"{n(a.get('precio_objetivo'))} {cur} ({change(a.get('potencial_pct'))})")]
    if t and "error" not in t:
        out += ["\n<b>Técnico</b>", _row("Tendencia", escape(t.get("tendencia", "—"))),
                _row("RSI 14", f"{n(t.get('rsi_14'), 0)} ({escape(t.get('rsi_lectura', ''))})"),
                _row("Desde máximo 52 sem.", f"{n(t.get('distancia_maximo_52s_pct'), 1)} %")]
        if t.get("cruce_reciente"):
            out.append(_row("Señal", f"{escape(t['cruce_reciente']['tipo'])} el {t['cruce_reciente']['fecha']}"))
    if d.get("proximos_resultados", {}).get("fecha"):
        out.append(f"\n📅 Próximos resultados: {escape(d['proximos_resultados']['fecha'])}")
    out.append("\n<i>Información, no asesoramiento. Pregúntame «¿qué opinas de "
               f"{escape(e.get('ticker', ''))}?» para un análisis explicado.</i>")
    return "\n".join(out)


def news(data: dict) -> str:
    lines = [f"📰 <b>Noticias: {escape(data['busqueda'])}</b>"]
    for item in data["noticias"]:
        title = escape(item["titular"])
        if item.get("enlace"):
            title = f'<a href="{escape(item["enlace"], quote=True)}">{title}</a>'
        lines.append(f"• {title}\n  <i>{escape(item.get('medio') or '')} · {item.get('fecha') or ''}</i>")
    return "\n".join(lines)


def watchlist(data: dict) -> str:
    if not data["lista"]:
        return "Tu lista está vacía. Añade valores con /seguir TICKER (p. ej. /seguir SAN.MC)."
    lines = ["👀 <b>Tu lista</b>"]
    for r in data["lista"]:
        lines.append(f"<b>{escape(r['ticker'])}</b>: {n(r.get('precio'))}  {change(r.get('dia_pct'))}")
    return "\n".join(lines)


def portfolio(data: dict) -> str:
    if not data.get("posiciones"):
        return ("💼 Tu cartera está vacía.\nAñade compras con /compra TICKER ACCIONES PRECIO, p. ej. "
                "/compra SAN.MC 100 11,20 (o dímelo: «he comprado 100 Santander a 11,20»).")
    t = data["total"]
    lines = [f"💼 <b>Tu cartera</b>: {n(t.get('valor_eur'))} €",
             f"Ganancia: {change(t.get('ganancia_eur'), '€')} ({change(t.get('ganancia_pct'))})",
             f"Hoy: {change(t.get('hoy_eur'), '€')}\n"]
    for p in data["posiciones"]:
        if "error" in p:
            lines.append(f"⚠️ <b>{escape(p['ticker'])}</b>: {escape(p['error'])}")
            continue
        lines.append(f"<b>{escape(p['ticker'])}</b> · {n(p['acciones'], 2)} acc. · {n(p.get('peso_pct'), 1)} %\n"
                     f"   {n(p['precio'])} {escape(p.get('moneda', ''))} (coste {n(p['coste_medio'])}) · "
                     f"{change(p.get('ganancia_pct'))} · hoy {change(p.get('dia_pct'))}")
    sales = next((v for k, v in data.items() if k.startswith("ventas_")), None)
    if sales and sales.get("ganancia_realizada_eur"):
        lines.append(f"\n🧾 Ventas este año: {change(sales['ganancia_realizada_eur'], '€')} · "
                     f"IRPF estimado {n(sales.get('irpf_estimado_eur'))} €")
    if data.get("irpf_si_vendieras_todo_hoy_eur"):
        lines.append(f"Si vendieras todo hoy, el IRPF del año sería ~{n(data['irpf_si_vendieras_todo_hoy_eur'])} €")
    lines.append("\n<i>Al cambio de hoy, sin comisiones. Vende con /venta TICKER ACCIONES PRECIO.</i>")
    return "\n".join(lines)


SCREEN_TITLES = {"dividendo": "Más dividendo", "per_bajo": "PER más bajo", "suben_hoy": "Las que más suben hoy",
                 "bajan_hoy": "Las que más bajan hoy", "suben_12m": "Las que más suben en 12 meses",
                 "bajan_12m": "Las que más bajan en 12 meses", "mas_grandes": "Las más grandes"}


def screener(data: dict) -> str:
    market = "Bolsa española" if data["bolsa"] == "es" else "EE. UU."
    lines = [f"🔎 <b>{SCREEN_TITLES.get(data['criterio'], data['criterio'])}</b> · {market}"]
    for i, r in enumerate(data["resultados"], 1):
        extra = []
        if r.get("per") is not None:
            extra.append(f"PER {n(r['per'], 1)}")
        if r.get("dividendo_pct"):
            extra.append(f"div. {n(r['dividendo_pct'], 1)} %")
        lines.append(f"{i}. <b>{escape(r['ticker'])}</b> {escape((r.get('nombre') or '')[:28])}\n"
                     f"   {n(r.get('precio'))} {escape(r.get('moneda', ''))} · hoy {change(r.get('dia_pct'))} · "
                     f"12 m {change(r.get('12m_pct'))}" + (f" · {' · '.join(extra)}" if extra else ""))
    lines.append("\n<i>Dividendo estimado por Yahoo (orientativo). Usa /ficha para el detalle.</i>")
    return "\n".join(lines)


ALERT_TEXT = {"encima": "suba por encima de", "debajo": "baje por debajo de", "movimiento": "se mueva más de un ±"}


def alert_line(a: dict) -> str:
    unit = " %" if a["kind"] == "movimiento" else ""
    return f"#{a['id']} {escape(a['ticker'])}: cuando {ALERT_TEXT[a['kind']]} {n(a['value'])}{unit}"


def plans_table(price_stars: int) -> str:
    def col(p: Plan) -> str:
        return (f"• {p.ai_queries_per_day} preguntas a la IA al día (escritas o por voz)\n• {p.alerts} alertas de precio\n"
                f"• Cartera de {p.positions} valores y lista de {p.watchlist}\n"
                f"• {p.documents} PDFs para preguntarme sobre ellos\n"
                f"• Resumen diario automático: {'sí ✅' if p.daily_summary else 'no'}")
    return (f"<b>Gratis</b>\n{col(FREE)}\n• Comandos /cotizacion, /grafico, /ficha, /mercado: ilimitados\n\n"
            f"<b>⭐ Premium — {price_stars} Stars / 30 días</b>\n{col(PREMIUM)}\n• Todo lo gratis, sin esperas\n")
