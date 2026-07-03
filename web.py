from contextlib import asynccontextmanager
import json
import logging
import os
import re
import secrets
import time
from functools import cache
from html import escape
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from mdweaver import weave_pdf
from pydantic import BaseModel

from db import (
    PdfEvent,
    QuoteEvent,
    ReadEvent,
    SearchEvent,
    init_db,
    record,
    stats,
)
from main import (
    embed,
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

    logger.info("Pre-warming vector library and loading transformer models into RAM...")
    _ = library()  # Pre-loads the book arrays
    _ = embed(["warmup"])  # Forces PyTorch to wake up and allocate memory
    logger.info("System pre-warmed successfully. Ready for traffic.")

    yield


app = FastAPI(title="classics", lifespan=lifespan)
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
STATS_HTML = Path(__file__).parent / "static" / "stats.html"
FAVICON = Path(__file__).parent / "static" / "favicon.svg"

# staging area for social quote cards; Buffer fetches these by URL when a post is
# queued, then re-hosts its own copy — so they need not live here permanently
CARDS_DIR = Path(__file__).parent / "cards"
CARDS_DIR.mkdir(exist_ok=True)
app.mount("/cards", StaticFiles(directory=CARDS_DIR), name="cards")

security = HTTPBasic()


def require_stats_auth(creds: HTTPBasicCredentials = Depends(security)) -> None:
    password = os.environ.get("STATS_PASSWORD")
    if not password:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "stats dashboard not configured"
        )
    user = os.environ.get("STATS_USER", "admin")
    ok = secrets.compare_digest(creds.username, user) & secrets.compare_digest(
        creds.password, password
    )
    if not ok:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


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
    summary: str = ""
    book_id: int = 0


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(FAVICON, media_type="image/svg+xml")


@app.get("/stats")
def stats_page(_: None = Depends(require_stats_auth)) -> FileResponse:
    return FileResponse(STATS_HTML)


@app.get("/api/stats")
def api_stats(_: None = Depends(require_stats_auth)) -> dict:
    return stats()


@app.get("/api/ask")
def ask(q: str, k: int = 5, per_book: int = 2, floor: float = 0.6) -> list[Match]:
    # should load from cache at this point, because it's pre-warmed in the
    # lifespan context manager
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
            summary=passages[i].summary,
            book_id=passages[i].book_id,
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


class ReadIn(BaseModel):
    query: str = ""
    author: str = ""
    title: str = ""
    ms: int


@app.post("/api/read")
def read(body: ReadIn) -> dict[str, bool]:
    record(
        ReadEvent(
            query=body.query,
            author=body.author,
            title=body.title,
            ms=body.ms,
        )
    )
    return {"ok": True}


# mirrors static/index.html's light palette so the PDF matches the share images
PDF_CSS = """
@page {
    size: A4;
    margin: 2.4cm 2cm 2cm;
    @top-left { content: "CLASSICS"; font-family: Georgia, serif; font-size: 8pt; letter-spacing: 3px; color: #b9ad97; }
    @top-right { content: "askthecanon.com"; font-family: Georgia, serif; font-size: 8pt; color: #b9ad97; }
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
                parts.append(f'<p class="text">{escape(para)}</p>')
        parts.append("</div>")
    return "\n".join(parts)


def _slug(query: str) -> str:
    base = re.sub(r"[^\w]+", "-", query.lower()).strip("-")
    return f"classics-{base}.pdf" if base else "classics.pdf"


@app.post("/api/pdf")
def pdf(body: PdfIn) -> Response:
    document = _pdf_document(body.query, body.items)
    data = weave_pdf(document, PDF_CSS)
    record(PdfEvent(query=body.query, passages=len(body.items)))
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_slug(body.query)}"'},
    )
