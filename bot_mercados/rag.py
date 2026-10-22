"""RAG (Retrieval-Augmented Generation): buscar en documentos y dar al modelo los fragmentos relevantes.

Flujo:
  1. Ingesta: el documento (PDF del usuario o informe anual de la SEC) se trocea en fragmentos de
     ~550 caracteres respetando frases, y cada fragmento se convierte en un vector (embedding) con
     un modelo local multilingüe. Todo se guarda en SQLite.
  2. Recuperación híbrida: para cada pregunta se ordenan los fragmentos por
       - similitud de significado (coseno entre embeddings), que entiende sinónimos e idiomas, y
       - BM25 (palabras clave), que acierta con nombres propios, cifras y siglas,
     y las dos listas se combinan con Reciprocal Rank Fusion (RRF).
  3. Generación: el agente recibe los mejores fragmentos numerados y responde citándolos [1], [2]...

Si fastembed no está instalado, funciona solo con BM25.
"""

import logging
import math
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CHUNK_CHARS = 550  # el modelo de embeddings solo lee ~128 tokens: fragmentos más largos se truncarían
MIN_CHUNK_CHARS = 40
RRF_K = 60  # constante habitual de Reciprocal Rank Fusion
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÑ¿¡\"“(0-9])")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = set("""
a al algo ante como con contra cual cuando de del desde donde e el ella ellas ellos en entre era es esa ese eso esta
este esto fue ha han hay la las le les lo los mas me mi mucho muy no nos o os otra otro para pero poco por que quien
se ser si sin sobre su sus tambien te tiene tu un una unas uno unos y ya
the of and to in for on with by as at from or an be is are was were this that these those it its our we their
which has have had not but been will would can may also such any all other than into more
""".split())


# --- Troceado -----------------------------------------------------------------------------

def _split_long(sentence: str, max_chars: int) -> list[str]:
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def split_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Trocea respetando párrafos y frases, con una frase de solapamiento entre fragmentos.

    El solapamiento evita que una idea partida por la mitad no aparezca entera en ningún fragmento.
    """
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        for sentence in _SENTENCE_END.split(paragraph):
            sentences.extend(_split_long(sentence, max_chars) if len(sentence) > max_chars else [sentence])

    chunks, current = [], []
    for sentence in filter(None, sentences):
        if current and len(" ".join(current)) + 1 + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            last = current[-1]
            current = [last] if len(last) + 1 + len(sentence) <= max_chars and len(last) < max_chars // 3 else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    # Fuera números de página, cabeceras sueltas y restos de tablas sin texto.
    return [c for c in chunks if sum(ch.isalpha() for ch in c) >= MIN_CHUNK_CHARS // 2]


try:  # lematizadores Snowball (vienen con fastembed): «deudas» y «deuda» -> «deud»
    from py_rust_stemmers import SnowballStemmer
    _STEMMERS = [SnowballStemmer("spanish"), SnowballStemmer("english")]
except ImportError:
    _STEMMERS = []


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes, sin palabras vacías y lematizado (español y luego inglés).

    En la evaluación (rag_eval.py) la lematización sube el acierto de BM25 del 55 % al 70 %.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = [w for w in _WORD.findall(text) if len(w) > 1 and w not in STOPWORDS]
    for stemmer in _STEMMERS:
        words = [w if w[0].isdigit() else stemmer.stem_word(w) for w in words]
    return words


# --- Puntuaciones -----------------------------------------------------------------------------

def bm25_scores(query: list[str], docs: list[Counter], lengths: list[int], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 clásico: premia los términos raros de la pregunta que aparecen en el fragmento."""
    n = len(docs)
    scores = np.zeros(n)
    if not n or not query:
        return scores
    avg_len = (sum(lengths) / n) or 1.0
    for term in set(query):
        df = sum(1 for tf in docs if term in tf)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, tf in enumerate(docs):
            f = tf.get(term)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lengths[i] / avg_len))
    return scores


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: suma 1/(k + posición) de cada lista. No necesita normalizar puntuaciones."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + position + 1)
    return fused


# --- Embeddings ---------------------------------------------------------------------------------

class Embedder:
    """Modelo de embeddings local (ONNX vía fastembed). Se carga la primera vez que se usa."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return False
        return True

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                log.info("Cargando el modelo de embeddings %s", self.model_name)
                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir) if self.cache_dir else None)
        vectors = np.array(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


# --- Base de conocimiento -----------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id INTEGER,              -- NULL = documento compartido (p. ej. informes de la SEC)
    ticker TEXT,
    title TEXT NOT NULL,
    source TEXT NOT NULL,          -- 'pdf' o 'sec'
    url TEXT,
    published TEXT,
    chunks INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_owner ON documents (owner_id);
CREATE INDEX IF NOT EXISTS documents_ticker ON documents (ticker, source);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    location TEXT,                 -- «pág. 12» o «Item 1A. Risk Factors»
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id, position);
"""


@dataclass
class Hit:
    document_id: int
    title: str
    location: str | None
    text: str
    score: float


@dataclass
class _DocIndex:
    """Fragmentos de un documento listos para buscar (en memoria, para no releer SQLite en cada pregunta)."""
    title: str
    locations: list[str | None]
    texts: list[str]
    term_freqs: list[Counter]
    lengths: list[int]
    vectors: np.ndarray | None


class KnowledgeBase:
    def __init__(self, path: str | Path, embedder: Embedder | None = None):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._cache: dict[int, _DocIndex] = {}
        with self._lock, self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)

    @property
    def mode(self) -> str:
        return "híbrida (vectores + BM25)" if self.embedder else "palabras clave (BM25)"

    # --- ingesta ---

    def add_document(self, title: str, sections: list[tuple[str | None, str]], source: str,
                     owner_id: int | None = None, ticker: str | None = None, url: str | None = None,
                     published: str | None = None) -> int:
        """Trocea, calcula embeddings y guarda. `sections` = [(ubicación, texto)], p. ej. una por página."""
        pieces = [(location, chunk) for location, text in sections for chunk in split_text(text)]
        if not pieces:
            raise ValueError("El documento no tiene texto que se pueda leer (¿es un PDF escaneado?).")
        vectors = self.embedder.embed([chunk for _, chunk in pieces]) if self.embedder else None
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO documents (owner_id, ticker, title, source, url, published, chunks, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (owner_id, ticker, title, source, url, published, len(pieces), now))
            doc_id = int(cur.lastrowid or 0)
            self._conn.executemany(
                "INSERT INTO chunks (document_id, position, location, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [(doc_id, i, loc, chunk, vectors[i].tobytes() if vectors is not None else None)
                 for i, (loc, chunk) in enumerate(pieces)])
        log.info("Documento %s «%s» indexado: %d fragmentos", doc_id, title, len(pieces))
        return doc_id

    def documents(self, owner_id: int) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT id, title, source, ticker, chunks, created_at FROM documents WHERE owner_id = ? ORDER BY id",
                (owner_id,)).fetchall()]

    def shared_document(self, ticker: str, source: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE owner_id IS NULL AND ticker = ? AND source = ? "
                                     "ORDER BY published DESC, id DESC LIMIT 1", (ticker, source)).fetchone()
        return dict(row) if row else None

    def delete_document(self, owner_id: int, doc_id: int) -> bool:
        with self._lock, self._conn:
            deleted = self._conn.execute("DELETE FROM documents WHERE id = ? AND owner_id = ?", (doc_id, owner_id)).rowcount
        self._cache.pop(doc_id, None)
        return deleted > 0

    # --- recuperación ---

    def _index(self, doc_id: int) -> _DocIndex:
        if doc_id not in self._cache:
            with self._lock:
                title = self._conn.execute("SELECT title FROM documents WHERE id = ?", (doc_id,)).fetchone()["title"]
                rows = self._conn.execute("SELECT location, text, embedding FROM chunks WHERE document_id = ? ORDER BY position",
                                          (doc_id,)).fetchall()
            tokens = [tokenize(r["text"]) for r in rows]
            vectors = None
            if self.embedder and rows and all(r["embedding"] for r in rows):
                vectors = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            self._cache[doc_id] = _DocIndex(title, [r["location"] for r in rows], [r["text"] for r in rows],
                                            [Counter(t) for t in tokens], [len(t) for t in tokens], vectors)
        return self._cache[doc_id]

    def search(self, query: str, doc_ids: list[int], k: int = 5, mode: str = "hybrid") -> list[Hit]:
        """Los k fragmentos más relevantes de esos documentos.

        mode: "hybrid" (BM25 + vectores con RRF), "bm25" o "vector" (estos dos, para evaluar cada parte).
        """
        entries: list[tuple[int, int]] = []  # (documento, posición del fragmento)
        term_freqs, lengths, vectors = [], [], []
        for doc_id in doc_ids:
            index = self._index(doc_id)
            entries += [(doc_id, i) for i in range(len(index.texts))]
            term_freqs += index.term_freqs
            lengths += index.lengths
            vectors.append(index.vectors)
        if not entries:
            return []

        rankings = []
        if mode in ("hybrid", "bm25"):
            keyword = bm25_scores(tokenize(query), term_freqs, lengths)
            rankings.append([int(i) for i in np.argsort(-keyword) if keyword[i] > 0][:50])
        if mode in ("hybrid", "vector") and self.embedder and all(v is not None for v in vectors):
            semantic = np.vstack(vectors) @ self.embedder.embed([query])[0]
            rankings.append([int(i) for i in np.argsort(-semantic)][:50])
        fused = sorted(rrf(rankings).items(), key=lambda x: -x[1])[:k]

        hits = []
        for idx, score in fused:
            doc_id, pos = entries[idx]
            index = self._index(doc_id)
            hits.append(Hit(doc_id, index.title, index.locations[pos], index.texts[pos], round(score, 4)))
        return hits
