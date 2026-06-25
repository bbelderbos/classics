import argparse
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import requests
import sentence_transformers as st

SEARCH_URL = "https://gutendex.com/books/?search="
BOOK_URL = "https://gutendex.com/books/"
BOOKS_DIR = Path("books")
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


def chunk_text(text: str, target_words: int = 600, overlap: int = 1) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for para in paragraphs:
        current.append(para)
        words += len(para.split())
        if words >= target_words:
            chunks.append("\n\n".join(current))
            current = current[-overlap:] if overlap else []
            words = sum(len(p.split()) for p in current)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


@cache
def _model():
    return st.SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    return _model().encode(texts, normalize_embeddings=True)


def retrieve(
    query: str, chunks: list[str], vectors: np.ndarray, k: int = 5
) -> list[tuple[int, float]]:
    scores = vectors @ embed([query])[0]
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top]


def load_index(book_id: int) -> tuple[list[str], np.ndarray] | None:
    chunks_path = BOOKS_DIR / f"{book_id}.chunks.json"
    vectors_path = BOOKS_DIR / f"{book_id}.npy"
    if chunks_path.exists() and vectors_path.exists():
        return json.loads(chunks_path.read_text()), np.load(vectors_path)
    return None


def build_index(book_id: int, text: str) -> tuple[list[str], np.ndarray]:
    chunks = chunk_text(text)
    vectors = embed(chunks)
    BOOKS_DIR.mkdir(exist_ok=True)
    (BOOKS_DIR / f"{book_id}.chunks.json").write_text(json.dumps(chunks))
    np.save(BOOKS_DIR / f"{book_id}.npy", vectors)
    return chunks, vectors


def preview(chunk: str, width: int = 90) -> str:
    line = " ".join(chunk.split())
    return line[:width] + ("…" if len(line) > width else "")


def ask_book(book_id: int, query: str) -> None:
    text_path = BOOKS_DIR / f"{book_id}.txt"
    if not text_path.exists():
        print(f"Fetching book {book_id}...")
        save_book(book_id)

    index = load_index(book_id)
    if index is None:
        print("Building index (one-time)...")
        chunks, vectors = build_index(book_id, text_path.read_text(encoding="utf-8"))
    else:
        chunks, vectors = index

    results = retrieve(query, chunks, vectors)
    print(f'\nPassages matching "{query}":\n')
    for rank, (i, score) in enumerate(results, 1):
        print(f"  {rank}  [{score:.2f}]  {preview(chunks[i])}")

    choice = input("\npick a number to deep read (enter to skip) > ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(results):
        idx = results[int(choice) - 1][0]
        print("\n" + "=" * 70)
        print(chunks[idx])
        print("=" * 70)


def run_search(term: str) -> None:
    for book in search_book(term):
        print(f"Title: {book.title}")
        print(f"Authors: {', '.join(book.authors)}")
        print(
            f"id: {book.id}, download_count: {book.download_count}, "
            f"languages: {', '.join(book.languages)}"
        )
        print("-" * 40)
    else:
        print("No results found.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Search Gutenberg, fetch and query books."
    )
    parser.add_argument("query", nargs="+", help="search term, or a book ID to fetch")
    parser.add_argument(
        "--ask", metavar="QUERY", help="find passages in a book (needs a book ID)"
    )
    args = parser.parse_args(argv)

    term = " ".join(args.query)
    if term.isdigit():
        book_id = int(term)
        if args.ask:
            ask_book(book_id, args.ask)
        else:
            saved_book_path = save_book(book_id)
            print(f"Saved book {book_id} to {saved_book_path}")
    else:
        if args.ask:
            parser.error("--ask needs a book ID, not a search term")
        run_search(term)


if __name__ == "__main__":
    main(sys.argv[1:])
