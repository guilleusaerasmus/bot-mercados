"""Notas de voz: transcripción con Whisper a través de la misma API compatible con OpenAI (Groq)."""

import logging

import openai

log = logging.getLogger(__name__)

MAX_VOICE_SECONDS = 90  # una pregunta hablada cabe de sobra; evita gastar cupo de audio


class TranscriptionError(Exception):
    pass


class Transcriber:
    def __init__(self, client: openai.OpenAI, model: str):
        self.client = client
        self.model = model

    def transcribe(self, audio: bytes, filename: str = "voz.ogg") -> str:
        """Devuelve el texto de un audio en español (Telegram envía las notas de voz en OGG/Opus)."""
        try:
            result = self.client.audio.transcriptions.create(
                model=self.model, file=(filename, audio), language="es", response_format="json", temperature=0.0)
        except openai.RateLimitError as exc:
            raise TranscriptionError("He alcanzado el límite de audio por ahora. Escríbeme la pregunta.") from exc
        except openai.APIError as exc:
            log.warning("Fallo al transcribir: %s", exc)
            raise TranscriptionError("No he podido entender el audio. ¿Me lo escribes?") from exc
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            raise TranscriptionError("No he oído nada en el audio. ¿Me lo escribes?")
        return text
