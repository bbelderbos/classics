from contextlib import asynccontextmanager
import json
import html
import logging
import re
import time
from functools import cache
from html import escape
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from mdweaver import weave_pdf
from pydantic import BaseModel

from db import QuoteEvent, SearchEvent, init_db, record
from main import (
    Passage,
    humanize_author,
    load_library,
    reflow,
    search_passages,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    init_db()
    yield


app = FastAPI(title="classics", lifespan=lifespan)
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@cache
def library() -> tuple[list[Passage], np.ndarray]:
    return load_library()


class Match(BaseModel):
    rank: int
    score: float
    cite: str
    author: str
    title: str
    label: str
    text: str


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/ask")
def ask(q: str, k: int = 5, per_book: int = 2, floor: float = 0.6) -> list[Match]:
    passages, vectors = library()
    if not passages:
        return []
    t0 = time.perf_counter()
    ranked = search_passages(q, passages, vectors, k, per_book, floor)
    shown = [
        {
            "id": passages[i].book_id,
            "label": passages[i].label,
            "offset": passages[i].offset,
            "score": round(score, 3),
        }
        for i, score in ranked
    ]
    record(SearchEvent(query=q, results=json.dumps(shown)))
    t1 = time.perf_counter()
    logger.info(
        "ask q=%r results=%d search=%.2fs",
        q,
        len(ranked),
        t1 - t0,
    )
    return [
        Match(
            rank=rank,
            score=score,
            cite=passages[i].cite(),
            author=passages[i].author,
            title=passages[i].title,
            label=passages[i].label,
            text=reflow(passages[i].text),
        )
        for rank, (i, score) in enumerate(ranked, 1)
    ]


class QuoteIn(BaseModel):
    query: str = ""
    author: str = ""
    title: str = ""
    text: str


@app.post("/api/quote")
def quote(body: QuoteIn) -> dict[str, bool]:
    record(
        QuoteEvent(
            query=body.query,
            author=body.author,
            title=body.title,
            text=body.text,
        )
    )
    return {"ok": True}


# mirrors static/index.html's light palette so the PDF matches the share images
PDF_CSS = """
@page {
    size: A4;
    margin: 2.4cm 2cm 2cm;
    @top-left { content: "CLASSICS"; font-family: Georgia, serif; font-size: 8pt; letter-spacing: 3px; color: #b9ad97; }
    @top-right { content: "belderbos.dev/classics"; font-family: Georgia, serif; font-size: 8pt; color: #b9ad97; }
    @bottom-center { content: counter(page); font-family: Georgia, serif; font-size: 9pt; color: #b9ad97; }
}
html { background: #fffdf8; }
body {
    font-family: Georgia, "Gelasio", "Times New Roman", serif;
    color: #2b2622;
    font-size: 11.5pt;
    line-height: 1.75;
}
.doc-head { text-align: center; margin: 0 0 2.6rem; }
.doc-head .eyebrow {
    font-style: italic; color: #8a7f6f; font-size: 9.5pt;
    letter-spacing: 1px; text-transform: uppercase; margin: 0 0 0.5rem;
}
.doc-head h1 { font-size: 22pt; font-weight: normal; margin: 0; line-height: 1.25; }
.passage { margin: 0 0 2.4rem; }
.passage + .passage { border-top: 1px solid #e2d9c8; padding-top: 2.4rem; }
.passage .where { font-size: 13pt; font-weight: bold; color: #8a5a2b; margin: 0 0 0.15rem; break-after: avoid; }
.passage .label {
    font-size: 8.5pt; letter-spacing: 1px; text-transform: uppercase;
    color: #8a7f6f; margin: 0 0 1.1rem; break-after: avoid;
}
.passage p.text { margin: 0 0 1rem; text-align: justify; hyphens: auto; }
u { text-decoration-color: rgba(138, 90, 43, 0.5); text-underline-offset: 2px; }
"""

_WORD = re.compile(r"\S+")


def _norm(word: str) -> str:
    return re.sub(r"[\W_]+", "", word.lower())


class PdfItem(BaseModel):
    author: str = ""
    title: str = ""
    label: str = ""
    text: str


class PdfIn(BaseModel):
    query: str = ""
    items: list[PdfItem]


def _pdf_document(query: str, items: list[PdfItem]) -> str:
    parts = ['<div class="doc-head">']
    if query:
        parts.append('<p class="eyebrow">You asked</p>')
        parts.append(f"<h1>{escape(query)}</h1>")
    parts.append("</div>")
    for item in items:
        where = " · ".join(p for p in (humanize_author(item.author), item.title) if p)
        parts.append('<div class="passage">')
        parts.append(f'<p class="where">{escape(where)}</p>')
        if item.label:
            parts.append(f'<p class="label">{escape(item.label)}</p>')
        for para in (p.strip() for p in item.text.split("\n\n")):
            if para:
                parts.append(f'<p class="text">{html.escape(para)}</p>')
        parts.append("</div>")
    return "\n".join(parts)


def _slug(query: str) -> str:
    base = re.sub(r"[^\w]+", "-", query.lower()).strip("-")
    return f"classics-{base}.pdf" if base else "classics.pdf"


@app.post("/api/pdf")
def pdf(body: PdfIn) -> Response:
    document = _pdf_document(body.query, body.items)
    data = weave_pdf(document, PDF_CSS)
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_slug(body.query)}"'},
    )
