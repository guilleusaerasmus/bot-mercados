"""El agente: habla con el modelo y ejecuta las herramientas que este pida.

Bucle de tool calling:
  1. Enviamos al modelo la conversación y la lista de herramientas.
  2. Si responde con "tool_calls", ejecutamos cada herramienta y le devolvemos los resultados.
  3. Repetimos hasta que responda con texto normal (la respuesta final) o se agoten los pasos.

Si un modelo agota su cupo (HTTP 429), se repite el paso con el siguiente de la lista:
en el plan gratuito de Groq cada modelo tiene su propio límite.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import openai

from .tools import TOOLS, ToolContext, execute_tool

log = logging.getLogger(__name__)

WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

SYSTEM_PROMPT = """Eres "Bot de Mercados", un analista financiero didáctico que responde en español. Hoy es {today}.
Reglas:
- Para cualquier precio, ratio, rentabilidad, noticia o cálculo usa SIEMPRE las herramientas; nunca inventes cifras.
- Empresas españolas: sufijo .MC (ITX.MC Inditex, SAN.MC Santander, IBE.MC Iberdrola). Índices: ^IBEX, ^GSPC
  (S&P 500), ^IXIC (Nasdaq), ^STOXX50E, ^GDAXI. Si dudas, usa buscar_ticker.
- Para "¿qué opinas de X?" o "analiza X" usa ficha_empresa y resume: qué hace, valoración, calidad, deuda,
  dividendo, qué dicen los analistas y señales técnicas, destacando puntos fuertes y riesgos
  (p. ej. payout > 100 %: reparte más de lo que gana; deuda neta > 3x EBITDA salvo bancos; PER muy alto).
- Interpreta los números (¿un PER de 30 es alto para su sector? ¿qué implica una caída del 40 %?) y explica la
  jerga en una frase cuando aparezca. Indica la fecha de los datos y la moneda.
- Si te piden comparar o ver la evolución, ofrece o genera un gráfico.
- Para riesgos, estrategia, segmentos, plantilla o cualquier detalle de informes anuales o de los PDFs del usuario,
  usa buscar_en_documentos: responde solo con lo que digan los fragmentos y cita cada dato como [1], [2]
  indicando documento y página o apartado. Si los fragmentos no lo dicen, reconócelo.
- No añadas de memoria datos que no den las herramientas (cifras, fechas, marcas, eventos): para describir una
  empresa usa su campo resumen. Si un dato no está, dilo. No expliques por qué sube o baja algo salvo que lo diga
  una noticia (herramienta noticias), y comprueba que lo que dices cuadra con los números.
- Respuestas claras y breves. Formato permitido: **negrita** y listas con guiones. Nada de tablas ni títulos con #.
- Puedes informar y educar, pero no des recomendaciones personalizadas de compra o venta: es información,
  no asesoramiento financiero. Si te las piden, da los datos a favor y en contra y deja la decisión al usuario."""


@dataclass
class Reply:
    text: str
    images: list[tuple[bytes, str]] = field(default_factory=list)
    ok: bool = True  # False si la IA no pudo responder (no debe contar para el límite diario)


class AllModelsBusy(Exception):
    pass


class MalformedToolCall(Exception):
    pass


class Agent:
    def __init__(self, client: openai.OpenAI, models: list[str] | tuple[str, ...], reasoning_effort: str | None = None,
                 max_history: int = 8, max_steps: int = 6):
        self.client = client
        self.models = list(models)
        self.reasoning_effort = reasoning_effort
        self.max_history = max_history  # mensajes (usuario + asistente) que se recuerdan por chat
        self.max_steps = max_steps  # llamadas al modelo como máximo por pregunta
        self._histories: dict[int, list[dict]] = {}
        self._lock = threading.Lock()

    def reset(self, chat_id: int) -> None:
        with self._lock:
            self._histories.pop(chat_id, None)

    def reply(self, chat_id: int, text: str, ctx: ToolContext | None = None) -> Reply:
        ctx = ctx or ToolContext(user_id=chat_id, chat_id=chat_id)
        with self._lock:
            history = list(self._histories.get(chat_id, []))

        try:
            answer = self._run(history + [{"role": "user", "content": text}], ctx)
        except AllModelsBusy:
            return Reply("Ahora mismo hay mucha demanda y he alcanzado el límite de la IA. Prueba en un minuto; "
                         "mientras, los comandos /cotizacion, /grafico y /mercado siguen funcionando.", ok=False)
        except MalformedToolCall:
            return Reply("No he conseguido completar la consulta. Prueba a reformularla o a dividirla en partes.", ok=False)
        except openai.APIConnectionError:
            return Reply("No consigo conectar con el modelo de IA. Revisa la conexión a internet.", ok=False)
        except openai.AuthenticationError:
            log.error("La clave LLM_API_KEY no es válida.")
            return Reply("La clave de la API del modelo no es válida (revisa LLM_API_KEY).", ok=False)
        except openai.APIStatusError as exc:
            log.error("Error de la API del modelo (%s): %s", exc.status_code, exc.message)
            return Reply("El modelo ha devuelto un error. Prueba a reformular la pregunta.", ok=False)

        # Solo guardamos la pregunta y la respuesta final, no los pasos intermedios con
        # herramientas: así el historial gasta menos tokens del límite gratuito.
        with self._lock:
            turns = self._histories.get(chat_id, []) + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer},
            ]
            self._histories[chat_id] = turns[-self.max_history:]
        return Reply(answer, ctx.images)

    def _complete(self, messages: list[dict]):
        """Llama al primer modelo con cupo disponible.

        Si el modelo genera una llamada a herramienta mal formada (p. ej. inventa el nombre «graph»
        en vez de «grafico»), Groq rechaza la petición con 400 tool_use_failed: es aleatorio, así que
        se reintenta una vez y, si vuelve a pasar, se prueba el siguiente modelo.
        """
        tool_failures = 0
        for model in self.models:
            kwargs: dict = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
            if self.reasoning_effort and model.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
            for attempt in range(2):
                try:
                    return self.client.chat.completions.create(**kwargs)
                except openai.RateLimitError:
                    log.warning("Cupo agotado en %s; pruebo el siguiente modelo.", model)
                    break
                except openai.BadRequestError as exc:
                    if "tool_use_failed" not in str(exc):
                        raise
                    tool_failures += 1
                    log.warning("Llamada a herramienta mal formada en %s (intento %d): %s", model, attempt + 1, exc.message)
        if tool_failures:
            raise MalformedToolCall
        raise AllModelsBusy

    def _run(self, conversation: list[dict], ctx: ToolContext) -> str:
        now = datetime.now(ZoneInfo("Europe/Madrid"))
        # El nombre del día a mano: strftime("%A") depende del idioma del sistema.
        today = f"{WEEKDAYS[now.weekday()]} {now:%d/%m/%Y}"
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT.format(today=today)}]
        messages += conversation

        for _ in range(self.max_steps):
            msg = self._complete(messages).choices[0].message
            if not msg.tool_calls:
                return (msg.content or "").strip() or "No tengo respuesta para eso."

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in msg.tool_calls
                ],
            })
            for call in msg.tool_calls:
                log.info("Herramienta %s(%s)", call.function.name, call.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": execute_tool(call.function.name, call.function.arguments, ctx),
                })

        return "La consulta necesitaba demasiados pasos. Prueba a dividirla en preguntas más concretas."
