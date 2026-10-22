"""Pruebas del RAG: troceado, BM25, fusión, base de conocimiento, fuentes y herramienta (sin internet)."""

import io
import json
import zlib
from collections import Counter

import numpy as np
import pytest
from matplotlib.figure import Figure

from bot_mercados import documentos, rag, rag_eval, tools
from bot_mercados.documentos import DocumentError
from bot_mercados.rag import KnowledgeBase
from bot_mercados.tools import ToolContext


class FakeEmbedder:
    """Embeddings deterministas por palabras: suficiente para probar el flujo sin descargar el modelo."""

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in rag.tokenize(text):
                out[i, zlib.crc32(word.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# --- piezas sueltas ---

def test_split_text_respeta_frases_y_tamano():
    text = " ".join(f"Frase número {i} sobre los resultados del grupo en el ejercicio." for i in range(40))
    chunks = rag.split_text(text, max_chars=300)
    assert len(chunks) > 5 and all(len(c) <= 300 for c in chunks)
    assert all(c.endswith(".") for c in chunks)  # no corta frases por la mitad
    assert chunks[1].split(". ")[0] in chunks[0]  # una frase de solapamiento


def test_split_text_descarta_restos_sin_texto():
    assert rag.split_text("12\n\n— 3 —\n\n4.500 | 3.200 | 1.100") == []


def test_tokenize_normaliza_y_lematiza():
    assert rag.tokenize("Las DEUDAS") == rag.tokenize("deuda")
    assert "riesg" in rag.tokenize("Los riesgos")[0]
    assert rag.tokenize("4.820 millones")[0] == "4.820"


def test_bm25_premia_terminos_raros():
    docs = [Counter(rag.tokenize(t)) for t in ("el yuan y el dolar", "el dolar", "ventas del grupo")]
    scores = rag.bm25_scores(rag.tokenize("yuan"), docs, [sum(d.values()) for d in docs])
    assert scores.argmax() == 0 and scores[1] == 0


def test_rrf_combina_rankings():
    fused = rag.rrf([[1, 2, 3], [2, 1, 3]])
    assert fused[1] == fused[2] > fused[3]


# --- base de conocimiento ---

def test_busqueda_y_citas_con_ubicacion():
    kb = KnowledgeBase(":memory:", FakeEmbedder())
    doc = kb.add_document("Informe", [("pág. 1", "Las ventas crecieron un 9 % hasta 4.820 millones de euros."),
                                      ("pág. 2", "El dividendo será de 0,84 euros por acción, pagado en mayo.")], "pdf",
                          owner_id=1)
    hits = kb.search("¿Qué dividendo pagan por acción?", [doc], k=2)
    assert hits[0].location == "pág. 2" and hits[0].title == "Informe"


def test_documentos_son_privados_por_usuario():
    kb = KnowledgeBase(":memory:")
    doc = kb.add_document("Mío", [("pág. 1", "Texto suficientemente largo para ser un fragmento útil del documento.")],
                          "pdf", owner_id=1)
    assert [d["id"] for d in kb.documents(1)] == [doc] and kb.documents(2) == []
    assert not kb.delete_document(2, doc)  # otro usuario no puede borrarlo
    assert kb.delete_document(1, doc) and kb.documents(1) == []


def test_documento_sin_texto():
    with pytest.raises(ValueError):
        KnowledgeBase(":memory:").add_document("Vacío", [("pág. 1", "   ")], "pdf", owner_id=1)


def test_evaluacion_bm25_minima():
    kb, doc = rag_eval.build(None)
    result = rag_eval.evaluate(kb, doc, "bm25")
    # BM25 acierta las preguntas literales; las paráfrasis e inglés son trabajo de los vectores.
    assert result["hit@3"] >= 0.6


# --- fuentes ---

def test_html_to_text_quita_lo_oculto():
    html = ('<html><head><title>x</title><style>p{}</style></head><body><div style="display: none">'
            '<ix:header>dato XBRL oculto</ix:header></div><p>Item 1A. Risk Factors</p><p>Texto&nbsp;visible'
            '<br>siguiente</p><script>alert(1)</script></body></html>')
    text = documentos.html_to_text(html)
    assert "oculto" not in text and "alert" not in text and "x" not in text.split()
    assert "Texto visible\nsiguiente" in text


def test_secciones_de_10k_con_titulos_largos():
    text = ("Portada\nItem 1. Business\nHacemos cosas.\nItem 5. Market for Registrant’s Common Equity, Related "
            "Stockholder Matters and Issuer Purchases of Equity Securities\nRecompra de 110 billion.\nItem 7. MD&A\nVentas.")
    sections = documentos.sec_sections(text)
    labels = [loc for loc, _ in sections]
    assert labels[0] is None and labels[1] == "Item 1. Business"
    assert labels[2].startswith("Item 5. Market for Registrant") and len(labels[2]) < 90
    assert "110 billion" in sections[2][1]


def _pdf(pages: list[str]) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for text in pages:
            fig = Figure()
            fig.text(0.1, 0.5, text)
            pdf.savefig(fig)
    return buf.getvalue()


def test_pdf_por_paginas():
    sections, pages = documentos.pdf_sections(_pdf(["Primera pagina con ventas", "Segunda pagina con riesgos"]))
    assert pages == 2 and sections[1][0] == "pág. 2" and "riesgos" in sections[1][1]


def test_pdf_invalido():
    with pytest.raises(DocumentError):
        documentos.pdf_sections(b"esto no es un pdf")


# --- herramienta para la IA ---

class FakeSec:
    def __init__(self):
        self.calls = 0

    def latest_annual_report(self, ticker):
        return {"url": "https://sec/aapl.htm"}

    def annual_report_sections(self, ticker):
        self.calls += 1
        meta = {"company": "Apple Inc.", "form": "10-K", "filed": "2025-10-31", "period": "2025-09-27",
                "url": "https://sec/aapl.htm"}
        return meta, [("Item 1A. Risk Factors", "The Company relies on single-source partners in Asia to manufacture "
                                                "many components, which exposes it to supply disruptions.")]


def call(ctx, **args):
    return json.loads(tools.execute_tool("buscar_en_documentos", json.dumps(args), ctx))


def test_herramienta_indexa_el_10k_una_sola_vez():
    kb, sec = KnowledgeBase(":memory:", FakeEmbedder()), FakeSec()
    ctx = ToolContext(1, 1, kb=kb, sec=sec)
    out = call(ctx, pregunta="supply risks in Asia", ticker="aapl")
    assert out["fragmentos"][0]["documento"] == "Apple Inc. — 10-K 2025"
    assert out["fragmentos"][0]["ubicacion"] == "Item 1A. Risk Factors" and "[ref]" in out["instruccion"]
    call(ctx, pregunta="otra pregunta sobre Asia", ticker="AAPL")
    assert sec.calls == 1  # el segundo usuario/pregunta reutiliza el índice


def test_herramienta_sin_documentos_explica_que_hacer():
    ctx = ToolContext(1, 1, kb=KnowledgeBase(":memory:"), sec=None)
    out = call(ctx, pregunta="¿Qué riesgos tiene?", ticker="SAN.MC")
    assert "PDF" in out["error"]


def test_herramienta_desactivada():
    assert "no está activada" in call(ToolContext(1, 1), pregunta="hola que tal")["error"]
