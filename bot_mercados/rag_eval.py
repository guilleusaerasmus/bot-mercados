"""Evaluación de la recuperación del RAG: ¿encuentra el fragmento correcto?

Un informe anual inventado con 12 párrafos y 20 preguntas con su respuesta esperada: con las mismas
palabras que el texto, parafraseadas (sin palabras en común), en inglés y con términos exactos
(nombres, lugares, cifras).
Se comparan tres modos:
  - bm25: solo palabras clave
  - vector: solo embeddings
  - hybrid: ambas combinadas con Reciprocal Rank Fusion
Métricas: hit@1 (la respuesta es el primer resultado), hit@3 y MRR (media de 1/posición).

    python -m bot_mercados evaluar-rag
"""

from .rag import Embedder, KnowledgeBase

REPORT = [
    "Las ventas netas del ejercicio 2025 alcanzaron 4.820 millones de euros, un 9,4 % más que el año anterior, "
    "impulsadas por la venta online y la apertura de 35 tiendas en Asia.",
    "El beneficio neto atribuido fue de 612 millones de euros, con un margen neto del 12,7 %. El margen bruto "
    "mejoró 80 puntos básicos hasta el 57,9 % gracias a menores costes logísticos.",
    "El consejo de administración propondrá a la junta general un dividendo de 0,84 euros por acción, pagadero en "
    "dos tramos en mayo y noviembre, lo que supone destinar el 60 % del beneficio a los accionistas.",
    "La compañía cerró el ejercicio sin deuda financiera y con una posición de caja neta de 1.150 millones de euros. "
    "La liquidez disponible, incluidas las líneas de crédito no dispuestas, asciende a 2.000 millones.",
    "Entre los principales riesgos destacan la volatilidad de los tipos de cambio, en especial del dólar y del yuan, "
    "la dependencia de proveedores en el sudeste asiático y la posible caída del consumo en Europa.",
    "En materia de ciberseguridad, el grupo invirtió 18 millones en proteger sus sistemas tras un intento de "
    "intrusión detectado en marzo que no afectó a datos de clientes.",
    "El 78 % del algodón utilizado procede de fuentes orgánicas o recicladas y la empresa se compromete a reducir "
    "un 50 % sus emisiones de CO2 antes de 2030.",
    "La plantilla media fue de 41.300 empleados, de los que el 74 % son mujeres. La rotación voluntaria bajó al 11 %.",
    "El plan estratégico 2026-2028 prevé invertir 900 millones en tecnología, automatizar los centros logísticos de "
    "Zaragoza y Valencia y lanzar la marca en Estados Unidos.",
    "Para 2026 la dirección espera un crecimiento de las ventas de entre el 6 % y el 8 % a tipo de cambio constante "
    "y un margen bruto estable.",
    "Existe un procedimiento judicial abierto en Italia por una supuesta infracción de diseño; los asesores legales "
    "consideran remoto que termine en una sanción relevante.",
    "La junta aprobó un programa de recompra de acciones propias de hasta 300 millones de euros que se ejecutará "
    "durante 2026.",
]

QUESTIONS = [  # (pregunta, párrafo que la responde, tipo)
    ("¿Cuánto dinero va a repartir a los accionistas por cada título?", 2, "paráfrasis"),
    ("¿Tiene deudas la empresa?", 3, "paráfrasis"),
    ("¿Qué peligros amenazan al negocio?", 4, "paráfrasis"),
    ("¿Cuántas personas trabajan en la compañía?", 7, "paráfrasis"),
    ("¿Cuál fue el margen bruto?", 1, "literal"),
    ("¿Han sufrido algún ataque informático?", 5, "paráfrasis"),
    ("¿Qué planes tienen para Estados Unidos?", 8, "literal"),
    ("¿Cuánto esperan crecer el año que viene?", 9, "paráfrasis"),
    ("What is the dividend per share?", 2, "inglés"),
    ("How much cash does the company have?", 3, "inglés"),
    ("¿Hay algún juicio pendiente?", 10, "paráfrasis"),
    ("¿Van a recomprar acciones propias?", 11, "literal"),
    ("¿Qué porcentaje del algodón es reciclado?", 6, "literal"),
    ("¿Cuánto vendieron en 2025?", 0, "literal"),
    # Términos exactos (nombres, lugares, cifras): donde BM25 suele ganar a los vectores.
    ("¿Qué dicen del yuan?", 4, "término exacto"),
    ("¿Qué pasa con Zaragoza y Valencia?", 8, "término exacto"),
    ("¿Qué ocurrió en marzo?", 5, "término exacto"),
    ("¿Qué es lo de Italia?", 10, "término exacto"),
    ("¿A qué corresponden los 41.300?", 7, "término exacto"),
    ("¿Qué pasa con el CO2?", 6, "término exacto"),
]


def build(embedder: Embedder | None) -> tuple[KnowledgeBase, int]:
    """Base de conocimiento en memoria con el informe de prueba (cada párrafo es un fragmento)."""
    kb = KnowledgeBase(":memory:", embedder)
    return kb, kb.add_document("Informe anual de prueba", [(f"§{i}", p) for i, p in enumerate(REPORT)], "eval")


def evaluate(kb: KnowledgeBase, doc_id: int, mode: str) -> dict:
    hits1 = hits3 = 0
    reciprocal = 0.0
    failures = []
    for question, expected, kind in QUESTIONS:
        locations = [h.location for h in kb.search(question, [doc_id], k=len(REPORT), mode=mode)]
        rank = locations.index(f"§{expected}") + 1 if f"§{expected}" in locations else None
        hits1 += rank == 1
        hits3 += rank is not None and rank <= 3
        reciprocal += 1 / rank if rank else 0
        if rank != 1:
            failures.append((question, kind, rank))
    n = len(QUESTIONS)
    return {"modo": mode, "hit@1": hits1 / n, "hit@3": hits3 / n, "mrr": reciprocal / n, "fallos": failures}


def main(embedder: Embedder | None) -> None:
    kb, doc_id = build(embedder)
    modes = ["bm25", "vector", "hybrid"] if embedder else ["bm25"]
    print(f"Evaluación de recuperación: {len(QUESTIONS)} preguntas sobre {len(REPORT)} párrafos\n")
    print(f"{'modo':<8} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    results = [evaluate(kb, doc_id, m) for m in modes]
    for r in results:
        print(f"{r['modo']:<8} {r['hit@1']:>6.0%} {r['hit@3']:>6.0%} {r['mrr']:>6.2f}")
    for r in results:
        if r["fallos"]:
            print(f"\nNo acierta a la primera ({r['modo']}):")
            for question, kind, rank in r["fallos"]:
                print(f"  - [{kind}] {question} -> posición {rank or 'no encontrada'}")
