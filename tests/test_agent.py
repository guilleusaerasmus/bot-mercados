"""Pruebas del bucle del agente con un modelo simulado (no llaman a ninguna API)."""

import json
from types import SimpleNamespace

import httpx
import openai

from bot_mercados import agent as agent_mod
from bot_mercados.agent import Agent
from bot_mercados.tools import ToolContext


def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _rate_limit():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError("limit", response=httpx.Response(429, request=request), body=None)


class FakeClient:
    """Imita client.chat.completions.create devolviendo respuestas preparadas."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(json.loads(json.dumps({k: v for k, v in kwargs.items() if k != "tools"})))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_bucle_con_herramienta(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_tool", lambda name, args, ctx: '{"precio": 11.0}')
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "cotizacion", {"tickers": ["SAN.MC"]})]),
        _response(content="Santander cotiza a 11 EUR."),
    ])
    reply = Agent(client, ["modelo"]).reply(1, "¿Santander?")

    assert reply.text == "Santander cotiza a 11 EUR." and reply.ok
    second = client.calls[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": '{"precio": 11.0}'}


def test_imagenes_de_herramientas_llegan_a_la_respuesta(monkeypatch):
    def fake_tool(name, args, ctx):
        ctx.images.append((b"png", "titulo"))
        return '{"ok": true}'
    monkeypatch.setattr(agent_mod, "execute_tool", fake_tool)
    client = FakeClient([_response(tool_calls=[_tool_call("c1", "grafico", {})]), _response(content="Aquí lo tienes")])
    reply = Agent(client, ["m"]).reply(1, "gráfico", ToolContext(1, 1))
    assert reply.images == [(b"png", "titulo")]


def test_cambia_de_modelo_si_se_agota_el_cupo():
    client = FakeClient([_rate_limit(), _response(content="ok")])
    reply = Agent(client, ["grande", "pequeno"]).reply(1, "hola")
    assert reply.text == "ok"
    assert [c["model"] for c in client.calls] == ["grande", "pequeno"]


def _tool_use_failed():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"code": "tool_use_failed", "message": "attempted to call tool 'graph' which was not in request.tools"}}
    return openai.BadRequestError("Error code: 400 - tool_use_failed", response=httpx.Response(400, request=request),
                                  body=body)


def test_llamada_mal_formada_se_reintenta():
    client = FakeClient([_tool_use_failed(), _response(content="ok")])
    assert Agent(client, ["a", "b"]).reply(1, "gráfico").text == "ok"
    assert [c["model"] for c in client.calls] == ["a", "a"]  # reintento con el mismo modelo


def test_llamada_mal_formada_persistente_cambia_de_modelo_y_avisa():
    client = FakeClient([_tool_use_failed(), _tool_use_failed(), _response(content="ok")])
    assert Agent(client, ["a", "b"]).reply(1, "x").text == "ok"
    assert [c["model"] for c in client.calls] == ["a", "a", "b"]
    reply = Agent(FakeClient([_tool_use_failed()] * 2), ["a"]).reply(1, "x")
    assert not reply.ok and "reformular" in reply.text


def test_otros_errores_400_no_se_reintentan():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = openai.BadRequestError("context_length_exceeded", response=httpx.Response(400, request=request), body=None)
    client = FakeClient([error, _response(content="no debería llegar")])
    assert not Agent(client, ["a"]).reply(1, "x").ok and len(client.calls) == 1


def test_todos_los_modelos_agotados():
    reply = Agent(FakeClient([_rate_limit(), _rate_limit()]), ["a", "b"]).reply(1, "hola")
    assert not reply.ok and "límite" in reply.text


def test_reasoning_effort_solo_en_gpt_oss():
    client = FakeClient([_response(content="a"), _response(content="b")])
    Agent(client, ["openai/gpt-oss-120b"], reasoning_effort="low").reply(1, "x")
    Agent(client, ["qwen/qwen3.8-27b"], reasoning_effort="low").reply(2, "x")
    assert client.calls[0]["extra_body"] == {"reasoning_effort": "low"}
    assert "extra_body" not in client.calls[1]


def test_historial_guarda_solo_pregunta_y_respuesta(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_tool", lambda name, args, ctx: "{}")
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "cotizacion", {"tickers": ["X"]})]),
        _response(content="uno"),
        _response(content="dos"),
    ])
    bot = Agent(client, ["m"])
    bot.reply(7, "primera")
    bot.reply(7, "segunda")
    assert [m["role"] for m in client.calls[-1]["messages"]] == ["system", "user", "assistant", "user"]


def test_historial_separado_por_chat_y_reset():
    client = FakeClient([_response(content="a"), _response(content="b"), _response(content="c")])
    bot = Agent(client, ["m"])
    bot.reply(1, "hola")
    bot.reply(2, "hola")
    assert len(client.calls[1]["messages"]) == 2  # el chat 2 no ve el historial del chat 1
    bot.reset(1)
    bot.reply(1, "otra")
    assert len(client.calls[2]["messages"]) == 2


def test_historial_limitado():
    client = FakeClient([_response(content=str(i)) for i in range(10)])
    bot = Agent(client, ["m"], max_history=4)
    for i in range(10):
        bot.reply(1, f"p{i}")
    assert len(client.calls[-1]["messages"]) == 1 + 4 + 1  # system + 4 de historial + pregunta actual


def test_limite_de_pasos(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_tool", lambda name, args, ctx: "{}")
    loop = [_response(tool_calls=[_tool_call(f"c{i}", "cotizacion", {"tickers": ["X"]})]) for i in range(3)]
    assert "demasiados pasos" in Agent(FakeClient(loop), ["m"], max_steps=3).reply(1, "bucle").text


def test_una_pregunta_fallida_no_se_guarda():
    bot = Agent(FakeClient([_rate_limit()]), ["m"])
    bot.reply(1, "hola")
    assert bot._histories == {}
