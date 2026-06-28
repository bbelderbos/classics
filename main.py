import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyperclip
import requests
from rich.console import Console
from rich.panel import Panel

console = Console()

# use the locally cached model and skip the hub round-trip (and its rate-limit warning)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")  # silence the model's "Loading weights" bar

SEARCH_URL = "https://gutendex.com/books/"
BOOK_URL = "https://gutendex.com/books/"
BOOKS_DIR = Path("books")
LIBRARY_FILE = Path("library.txt")
EMBED_MODEL = "all-mpnet-base-v2"
HTTP_TIMEOUT = 30  # seconds — never let a stalled Gutendex hang the CLI forever
# below this the best match is noise — nonsense queries top out around 0.28
MIN_SCORE = 0.35


class Book(NamedTuple):
    id: int
    title: str
    authors: list[str]
    languages: list[str]
    download_count: int


def search_book(title: str) -> list[Book]:
    response = requests.get(SEARCH_URL, params={"search": title}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return [
        Book(
            id=int(book["id"]),
            title=book["title"],
            authors=[entry["name"] for entry in book["authors"]],
            languages=book["languages"],
            download_count=int(book["download_count"]),
        )
        for book in response.json().get("results", [])
    ]


def get_book_text(book_id: int) -> str:
    response = requests.get(f"{BOOK_URL}{book_id}", timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    formats = response.json().get("formats", {})
    text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
    if not text_url:
        raise ValueError(f"No plain text format found for book ID {book_id}")
    text_response = requests.get(text_url, timeout=HTTP_TIMEOUT)
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


def diversify(
    ranked: list[tuple[int, float]], key_of: Callable[[int], str], k: int, per_book: int
) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    seen: Counter[str] = Counter()
    for idx, score in ranked:
        key = key_of(idx)
        if per_book > 0 and seen[key] >= per_book:
            continue
        selected.append((idx, score))
        seen[key] += 1
        if len(selected) == k:
            break
    return selected


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


def reflow(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(" ".join(p.split()) for p in paragraphs)


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    flat = " ".join(reflow(text).split())
    return [s.strip() for s in SENTENCE_RE.split(flat) if s.strip()]


def _grow_window(sentences: list[str], scores: np.ndarray, max_words: int) -> str:
    lo = hi = int(np.argmax(scores))
    words = len(sentences[lo].split())
    while hi - lo + 1 < 3:  # grow toward the better neighbour within the word budget
        left = scores[lo - 1] if lo > 0 else -np.inf
        right = scores[hi + 1] if hi < len(sentences) - 1 else -np.inf
        if left == right == -np.inf:
            break
        nxt = lo - 1 if left >= right else hi + 1
        if words + len(sentences[nxt].split()) > max_words:
            break
        lo, hi = min(lo, nxt), max(hi, nxt)
        words += len(sentences[nxt].split())
    return " ".join(sentences[lo : hi + 1])


def best_excerpt(text: str, query: str, max_words: int = 60) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return reflow(text)
    scores = embed(sentences) @ embed([query])[0]
    return _grow_window(sentences, scores, max_words)


def humanize_author(author: str) -> str:
    if author.count(",") == 1:
        last, first = (part.strip() for part in author.split(","))
        return f"{first} {last}"
    return author


class Passage(NamedTuple):
    title: str
    author: str
    label: str
    text: str
    book_id: int = 0
    offset: int = 0  # chunk index within the book
    summary: str = ""  # book-level blurb, repeated across the book's passages

    def cite(self) -> str:
        where = " · ".join(part for part in (self.author, self.title) if part)
        return f"{where} — {self.label}" if self.label else where

    def share(self, excerpt: str | None = None) -> str:
        body = excerpt if excerpt is not None else reflow(self.text)
        author = humanize_author(self.author)
        where = ", ".join(part for part in (author, self.title) if part)
        if self.label:
            where += f" ({self.label})"
        return f"“{body}”\n\n— {where}"


SUMMARY_SUFFIX = "(This is an automatically generated summary.)"


def clean_summary(summary: str) -> str:
    return summary.replace(SUMMARY_SUFFIX, "").strip()


class BookMeta(NamedTuple):
    title: str
    author: str
    summary: str


def book_metadata(book_id: int) -> BookMeta:
    response = requests.get(f"{BOOK_URL}{book_id}", timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    author = ", ".join(a["name"] for a in data.get("authors", []))
    summaries = data.get("summaries") or []
    summary = clean_summary(summaries[0]) if summaries else ""
    return BookMeta(data.get("title", str(book_id)), author, summary)


def read_library(path: Path = LIBRARY_FILE) -> list[int]:
    if not path.exists():
        return []
    ids = []
    for line in path.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            ids.append(int(token))
    return ids


def indexed_books() -> set[int]:
    return {int(path.stem) for path in BOOKS_DIR.glob("*.npy")}


def index_books(book_ids: list[int]) -> None:
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
            if not meta_path.exists():
                meta = book_metadata(book_id)
                if not meta.author:
                    print(
                        f"  ! {book_id} has no author on Gutenberg — set it in {meta_path.name}"
                    )
                BOOKS_DIR.mkdir(exist_ok=True)
                meta_path.write_text(json.dumps(meta._asdict()))
            title = json.loads(meta_path.read_text())["title"]
            if built:
                print(f"  ~ {book_id} {title} (metadata backfilled)")
            else:
                chunks, _ = build_index(book_id, text_path.read_text(encoding="utf-8"))
                print(f"  + {book_id} {title} — {len(chunks)} passages")
        except Exception as e:
            print(f"  x {book_id}: {e}")
    print("done.")


def sync_library() -> None:
    desired = read_library()
    if not desired:
        print("library.txt has no ids — refusing to delete books/. Add ids, then sync.")
        return
    for book_id in sorted(indexed_books() - set(desired)):
        files = list(BOOKS_DIR.glob(f"{book_id}.*"))
        for path in files:
            path.unlink()
        print(f"  - {book_id} removed ({len(files)} files)")
    index_books(desired)


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
        summary = meta.get("summary", "")
        for i, c in enumerate(json.loads(chunks_path.read_text())):
            passages.append(
                Passage(title, author, c["label"], c["text"], book_id, i, summary)
            )
        matrices.append(np.load(vectors_path))
    return passages, np.vstack(matrices) if matrices else np.empty((0, 0))


def search_passages(
    query: str,
    passages: list[Passage],
    vectors: np.ndarray,
    k: int = 5,
    per_book: int = 2,
    floor: float = 0.6,
    min_score: float = MIN_SCORE,
) -> list[tuple[int, float]]:
    pool = retrieve(query, vectors, k=min(len(passages), 200))
    if not pool or pool[0][1] < min_score:
        return []  # best match too weak — off-domain question
    if floor > 0:
        cutoff = (
            floor * pool[0][1]
        )  # relative to the best match, so it scales per query
        pool = [(i, s) for i, s in pool if s >= cutoff]
    return diversify(
        pool, lambda i: passages[i].author or passages[i].title, k, per_book
    )


def ask(
    query: str,
    book_ids: list[int] | None = None,
    k: int = 5,
    per_book: int = 2,
    floor: float = 0.6,
) -> None:
    passages, vectors = load_library(book_ids)
    if not passages:
        print("No indexed books found. Run: uv run main.py sync")
        return

    results = search_passages(query, passages, vectors, k, per_book, floor)
    if not results:
        print(f'\nNothing strong enough for "{query}".')
        return
    console.print(f"\n[bold]Passages for[/bold] [italic]“{query}”[/italic]\n")
    for rank, (i, score) in enumerate(results, 1):
        passage = passages[i]
        console.print(
            f"  [bold cyan]{rank}[/]  [dim]{score:.2f}[/]  {passage.cite()}",
            no_wrap=True,
            overflow="ellipsis",
        )
        console.print(
            f"      [dim italic]{preview(passage.text)}[/]",
            no_wrap=True,
            overflow="ellipsis",
        )
        console.print()

    choice = input("pick a number to deep read (enter to skip) > ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(results):
        passage = passages[results[int(choice) - 1][0]]
        console.print()
        console.print(
            Panel(
                reflow(passage.text),
                title=passage.cite(),
                title_align="left",
                border_style="dim",
                padding=(1, 2),
            )
        )
        quote = passage.share(best_excerpt(passage.text, query))
        console.print("\n[bold]Shareable quote:[/]\n")
        console.print(quote)
        if input("\ncopy to clipboard? [y/N] > ").strip().lower() == "y":
            try:
                pyperclip.copy(quote)
                console.print("[green]copied[/]")
            except pyperclip.PyperclipException:
                console.print("[yellow]no clipboard available — copy it from above[/]")


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

    sub.add_parser("sync", help="reconcile books/ embeddings with library.txt")

    p_ask = sub.add_parser(
        "ask", help="find passages across the library for a question"
    )
    p_ask.add_argument("query")
    p_ask.add_argument("--book", type=int, help="limit the search to one book id")
    p_ask.add_argument("-k", type=int, default=5, help="how many passages to return")
    p_ask.add_argument(
        "--per-book", type=int, default=2, help="max passages per book (0 = no cap)"
    )
    p_ask.add_argument(
        "--floor",
        type=float,
        default=0.6,
        help="drop matches below this fraction of the top score (0 = keep all)",
    )

    args = parser.parse_args(argv)

    if args.command == "search":
        run_search(" ".join(args.terms))
    elif args.command == "fetch":
        print(f"Saved to {save_book(args.book_id)}")
    elif args.command == "sync":
        sync_library()
    elif args.command == "ask":
        ask(
            args.query,
            [args.book] if args.book else None,
            args.k,
            args.per_book,
            args.floor,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
