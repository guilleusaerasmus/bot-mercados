"""Fuentes de documentos para el RAG: PDFs de los usuarios e informes anuales (10-K) de la SEC.

Los informes de la SEC (EDGAR) son públicos y gratuitos. La SEC pide identificarse con un
User-Agent que incluya un contacto (SEC_USER_AGENT en el .env) y no superar 10 peticiones por segundo.
"""

import io
import json
import re
import threading
import time
import urllib.request
from html.parser import HTMLParser

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_PAGES = 400
MAX_SEC_BYTES = 20 * 1024 * 1024
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class DocumentError(Exception):
    """Documento que no se puede procesar; el mensaje se muestra al usuario."""


# --- PDF ----------------------------------------------------------------------------------

def pdf_sections(data: bytes) -> tuple[list[tuple[str, str]], int]:
    """Texto de cada página como [(«pág. N», texto)] y número de páginas."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("El PDF está protegido con contraseña.")
        pages = reader.pages
        if len(pages) > MAX_PDF_PAGES:
            raise DocumentError(f"El PDF tiene {len(pages)} páginas; el máximo es {MAX_PDF_PAGES}.")
        sections = [(f"pág. {i}", page.extract_text() or "") for i, page in enumerate(pages, 1)]
    except PdfReadError as exc:
        raise DocumentError("No he podido leer el PDF (¿está dañado?).") from exc
    if not any(text.strip() for _, text in sections):
        raise DocumentError("El PDF no tiene texto seleccionable (parece escaneado).")
    return sections, len(pages)


# --- HTML a texto ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Texto visible de un HTML: sin scripts, estilos ni bloques ocultos (las cabeceras XBRL de la SEC lo están)."""
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}
    VOID = {"br", "img", "hr", "meta", "input", "link", "col", "area", "base", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._stack: list[bool] = []  # True si la etiqueta oculta su contenido

    def _hidden(self) -> bool:
        return any(self._stack)

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag in self.VOID:
            return
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        self._stack.append(tag in ("script", "style", "head", "ix:header") or "display:none" in style)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self._stack:
            self._stack.pop()
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._hidden():
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text).strip()


# Títulos de apartado: «Item 5. Market for Registrant’s Common Equity, Related Stockholder Matters...» puede
# superar los 150 caracteres, así que se aceptan largos y se recortan al guardarlos.
_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-–]\s*(.{0,220})$", re.IGNORECASE)
MAX_LOCATION_CHARS = 70


def sec_sections(text: str) -> list[tuple[str | None, str]]:
    """Divide un 10-K por apartados («Item 1A. Risk Factors») para poder citar de dónde sale cada dato."""
    sections: list[tuple[str | None, str]] = []
    location, buffer = None, []
    for line in text.split("\n"):
        match = _ITEM.match(line)
        if match:
            if buffer:
                sections.append((location, "\n".join(buffer)))
            title = match.group(2).strip()
            if len(title) > MAX_LOCATION_CHARS:
                title = title[:MAX_LOCATION_CHARS].rsplit(" ", 1)[0] + "…"
            location = f"Item {match.group(1).upper()}. {title}".strip(". ")
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((location, "\n".join(buffer)))
    return [(loc, body) for loc, body in sections if body.strip()]


# --- SEC EDGAR ------------------------------------------------------------------------------

class SecClient:
    MIN_INTERVAL = 0.15  # < 10 peticiones por segundo, como pide la SEC

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._tickers: dict[str, int] | None = None
        self._lock = threading.Lock()
        self._last = 0.0

    def _get(self, url: str, max_bytes: int = MAX_SEC_BYTES) -> bytes:
        with self._lock:
            wait = self.MIN_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=60) as resp:
            data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise DocumentError("El informe es demasiado grande.")
        return data

    def cik(self, ticker: str) -> int:
        if self._tickers is None:
            raw = json.loads(self._get(TICKERS_URL))
            self._tickers = {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}
        cik = self._tickers.get(ticker.upper().replace(".", "-"))
        if cik is None:
            raise DocumentError(f"{ticker} no está registrada en la SEC (solo hay informes de empresas de EE. UU.).")
        return cik

    def latest_annual_report(self, ticker: str) -> dict:
        """Último 10-K (o 20-F/40-F de empresas extranjeras que cotizan en EE. UU.): metadatos y URL."""
        cik = self.cik(ticker)
        data = json.loads(self._get(SUBMISSIONS_URL.format(cik=cik)))
        recent = data.get("filings", {}).get("recent", {})
        for i, form in enumerate(recent.get("form", [])):
            if form in ("10-K", "20-F", "40-F"):
                accession = recent["accessionNumber"][i]
                return {
                    "company": data.get("name") or ticker,
                    "form": form,
                    "filed": recent["filingDate"][i],
                    "period": recent.get("reportDate", [None] * (i + 1))[i],
                    "url": ARCHIVE_URL.format(cik=cik, accession=accession.replace("-", ""),
                                              document=recent["primaryDocument"][i]),
                }
        raise DocumentError(f"No he encontrado un informe anual reciente de {ticker} en la SEC.")

    def annual_report_sections(self, ticker: str) -> tuple[dict, list[tuple[str | None, str]]]:
        meta = self.latest_annual_report(ticker)
        html = self._get(meta["url"]).decode("utf-8", errors="replace")
        return meta, sec_sections(html_to_text(html))
