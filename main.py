import argparse
import json
import os
import re
import sys
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import requests

# use the locally cached model and skip the hub round-trip (and its rate-limit warning)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SEARCH_URL = "https://gutendex.com/books/?search="
BOOK_URL = "https://gutendex.com/books/"
BOOKS_DIR = Path("books")
LIBRARY_FILE = Path("library.txt")
EMBED_MODEL = "all-MiniLM-L6-v2"


class Book(NamedTuple):
    id: int
    title: str
    authors: list[str]
    summary: str
    languages: list[str]
    download_count: int


def search_book(title: str) -> list[Book]:
    response = requests.get(SEARCH_URL + title).json()
    return [
        Book(
            id=int(book["id"]),
            title=book["title"],
            authors=[entry["name"] for entry in book["authors"]],
            summary=book["summaries"][0] if book["summaries"] else "",
            languages=book["languages"],
            download_count=int(book["download_count"]),
        )
        for book in response["results"]
    ]


def get_book_text(book_id: int) -> str:
    response = requests.get(f"{BOOK_URL}{book_id}")
    response.raise_for_status()
    formats = response.json().get("formats", {})
    text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
    if not text_url:
        raise ValueError(f"No plain text format found for book ID {book_id}")
    text_response = requests.get(text_url)
    text_response.raise_for_status()
    return text_response.text


def save_book(book_id: int) -> Path:
    text = get_book_text(book_id)
    BOOKS_DIR.mkdir(exist_ok=True)
    path = BOOKS_DIR / f"{book_id}.txt"
    path.write_text(text, encoding="utf-8")
    return path


class Chunk(NamedTuple):
    label: str  # e.g. "BOOK XI — Chapter IX", best-effort from Gutenberg headings
    text: str


HEADING_RE = re.compile(
    r"^(chapter|book|part|volume|canto|letter|epilogue|prologue)\b", re.IGNORECASE
)
SECTION_RE = re.compile(r"^(book|part|volume)\b", re.IGNORECASE)


def _heading(paragraph: str) -> str | None:
    line = " ".join(paragraph.split())
    return line if len(line) <= 60 and HEADING_RE.match(line) else None


def chunk_text(text: str, target_words: int = 600, overlap: int = 1) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    words = 0
    section = chapter = label = chunk_label = ""

    for para in paragraphs:
        heading = _heading(para)
        if heading:
            if SECTION_RE.match(heading):
                section, chapter = heading, ""
            else:
                chapter = heading
            label = " — ".join(part for part in (section, chapter) if part)
            continue
        if not current:
            chunk_label = label
        current.append(para)
        words += len(para.split())
        if words >= target_words:
            chunks.append(Chunk(chunk_label, "\n\n".join(current)))
            current = current[-overlap:] if overlap else []
            words = sum(len(p.split()) for p in current)
            chunk_label = label  # next chunk starts in whatever chapter is current
    if current:
        chunks.append(Chunk(chunk_label, "\n\n".join(current)))
    return chunks


@cache
def _model():
    import sentence_transformers as st  # lazy so the offline env vars take effect first

    return st.SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    return _model().encode(texts, normalize_embeddings=True)


def retrieve(query: str, vectors: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    scores = vectors @ embed([query])[0]
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top]


def build_index(book_id: int, text: str) -> tuple[list[Chunk], np.ndarray]:
    chunks = chunk_text(text)
    vectors = embed([c.text for c in chunks])
    BOOKS_DIR.mkdir(exist_ok=True)
    (BOOKS_DIR / f"{book_id}.chunks.json").write_text(
        json.dumps([c._asdict() for c in chunks])
    )
    np.save(BOOKS_DIR / f"{book_id}.npy", vectors)
    return chunks, vectors


def preview(chunk: str, width: int = 90) -> str:
    line = " ".join(chunk.split())
    return line[:width] + ("…" if len(line) > width else "")


class Passage(NamedTuple):
    title: str
    author: str
    label: str
    text: str

    def cite(self) -> str:
        where = " · ".join(part for part in (self.author, self.title) if part)
        return f"{where} — {self.label}" if self.label else where


def book_metadata(book_id: int) -> tuple[str, str]:
    data = requests.get(f"{BOOK_URL}{book_id}").json()
    author = ", ".join(a["name"] for a in data.get("authors", []))
    return data.get("title", str(book_id)), author


def read_library(path: Path = LIBRARY_FILE) -> list[int]:
    if not path.exists():
        return []
    ids = []
    for line in path.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            ids.append(int(token))
    return ids


def add_to_library(book_ids: list[int], path: Path = LIBRARY_FILE) -> None:
    new = [b for b in book_ids if b not in set(read_library(path))]
    if new:
        with path.open("a") as f:
            f.writelines(f"{b}\n" for b in new)


def index_books(book_ids: list[int]) -> None:
    if not book_ids:
        print(
            "Nothing to index. Add ids to library.txt or pass them: main.py index 1342"
        )
        return
    for book_id in book_ids:
        meta_path = BOOKS_DIR / f"{book_id}.meta.json"
        built = (BOOKS_DIR / f"{book_id}.npy").exists()
        if built and meta_path.exists():
            print(f"  = {book_id} already indexed")
            continue
        try:
            text_path = BOOKS_DIR / f"{book_id}.txt"
            if not text_path.exists():
                save_book(book_id)
            title, author = book_metadata(book_id)
            BOOKS_DIR.mkdir(exist_ok=True)
            meta_path.write_text(json.dumps({"title": title, "author": author}))
            if built:
                print(f"  ~ {book_id} {title} (metadata backfilled)")
            else:
                chunks, _ = build_index(book_id, text_path.read_text(encoding="utf-8"))
                print(f"  + {book_id} {title} — {len(chunks)} passages")
        except Exception as e:
            print(f"  x {book_id}: {e}")
    print("done.")


def load_library(book_ids: list[int] | None = None) -> tuple[list[Passage], np.ndarray]:
    passages: list[Passage] = []
    matrices: list[np.ndarray] = []
    for vectors_path in sorted(BOOKS_DIR.glob("*.npy")):
        book_id = int(vectors_path.stem)
        chunks_path = BOOKS_DIR / f"{book_id}.chunks.json"
        if (book_ids and book_id not in book_ids) or not chunks_path.exists():
            continue
        meta_path = BOOKS_DIR / f"{book_id}.meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        title, author = meta.get("title", str(book_id)), meta.get("author", "")
        for c in json.loads(chunks_path.read_text()):
            passages.append(Passage(title, author, c["label"], c["text"]))
        matrices.append(np.load(vectors_path))
    return passages, np.vstack(matrices) if matrices else np.empty((0, 0))


def ask(query: str, book_ids: list[int] | None = None, k: int = 5) -> None:
    passages, vectors = load_library(book_ids)
    if not passages:
        print("No indexed books found. Run: uv run main.py index")
        return

    results = retrieve(query, vectors, k)
    print(f'\nPassages for "{query}":\n')
    for rank, (i, score) in enumerate(results, 1):
        passage = passages[i]
        print(f"  {rank}  [{score:.2f}]  {passage.cite()}")
        print(f"        {preview(passage.text)}\n")

    choice = input("pick a number to deep read (enter to skip) > ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(results):
        passage = passages[results[int(choice) - 1][0]]
        print("\n" + "=" * 70)
        print(f"{passage.cite()}\n")
        print(passage.text)
        print("=" * 70)


def run_search(term: str) -> None:
    books = search_book(term)
    if not books:
        print("No results found.")
        return
    for book in books:
        print(f"Title: {book.title}")
        print(f"Authors: {', '.join(book.authors)}")
        print(
            f"id: {book.id}, download_count: {book.download_count}, "
            f"languages: {', '.join(book.languages)}"
        )
        print("-" * 40)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over a curated literary canon."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="find a book's id on Gutenberg")
    p_search.add_argument("terms", nargs="+")

    p_fetch = sub.add_parser("fetch", help="download one book to books/")
    p_fetch.add_argument("book_id", type=int)

    p_index = sub.add_parser("index", help="chunk + embed books into the library")
    p_index.add_argument(
        "book_ids",
        nargs="*",
        type=int,
        help="ids to add; omit to index all of library.txt",
    )

    p_ask = sub.add_parser(
        "ask", help="find passages across the library for a question"
    )
    p_ask.add_argument("query")
    p_ask.add_argument("--book", type=int, help="limit the search to one book id")
    p_ask.add_argument("-k", type=int, default=5, help="how many passages to return")

    args = parser.parse_args(argv)

    if args.command == "search":
        run_search(" ".join(args.terms))
    elif args.command == "fetch":
        print(f"Saved to {save_book(args.book_id)}")
    elif args.command == "index":
        if args.book_ids:
            add_to_library(args.book_ids)
        index_books(args.book_ids or read_library())
    elif args.command == "ask":
        ask(args.query, [args.book] if args.book else None, args.k)


if __name__ == "__main__":
    main(sys.argv[1:])
