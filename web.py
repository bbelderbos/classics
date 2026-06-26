import json
import logging
from functools import cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from db import QuoteEvent, SearchEvent, init_db, record
from main import (
    Passage,
    best_excerpt,
    load_library,
    reflow,
    search_passages,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="classics")
INDEX_HTML = Path(__file__).parent / "static" / "index.html"

init_db()


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
    highlight: str


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/ask")
def ask(q: str, k: int = 5, per_book: int = 2, floor: float = 0.6) -> list[Match]:
    passages, vectors = library()
    if not passages:
        return []
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
    return [
        Match(
            rank=rank,
            score=score,
            cite=passages[i].cite(),
            author=passages[i].author,
            title=passages[i].title,
            label=passages[i].label,
            text=reflow(passages[i].text),
            highlight=best_excerpt(passages[i].text, q),
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
