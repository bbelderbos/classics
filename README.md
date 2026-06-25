# classics

Semantic search over a curated library of public-domain literature.

Instead of grepping for words, you bring a *question* — "how should I deal with people who
wrong me?" — and get the passages that mean that, ranked across every book in your library and
cited by author, title, and chapter. Matching is by meaning, not keywords, so a passage scores
high even when it shares no words with your query.

Everything runs locally. Books come from [Project Gutenberg](https://gutenberg.org) via the
[Gutendex](https://gutendex.com) API; passages are embedded once with
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (no API key,
no network at query time) and matched with plain NumPy cosine similarity.

## How it works

1. **Index (once per book):** each book is split into ~600-word passages, tagged with their
   `Book`/`Chapter` heading, embedded, and cached to `books/<id>.{txt,chunks.json,npy,meta.json}`.
2. **Ask (per query):** only your question is embedded — one small vector — then matched against
   the cached library matrix. The corpus is never re-embedded.

The library is just `library.txt`: a list of Gutenberg IDs you grow over time.

## Setup

```bash
uv sync
```

## Usage

```bash
# find a book's Gutenberg id
uv run main.py search tolstoy war and peace

# download one book's text to books/
uv run main.py fetch 2600

# index books into the library (chunk + embed + cache)
uv run main.py index                 # index everything in library.txt (skips already-built)
uv run main.py index 1342 2680       # add these ids to library.txt and index them

# ask a question across the whole library
uv run main.py ask "how do I face death without fear"
uv run main.py ask "the meaning of suffering" -k 8        # return 8 passages
uv run main.py ask "ivan meets the devil" --book 28054    # limit to one book
```

`ask` prints the top passages with citations, then lets you pick one to read in full.

### Growing the library

Add a Gutenberg id (use `search` to find it) to `library.txt`, then run `uv run main.py index`.
The indexer skips anything already built, backfills missing metadata without re-embedding, and
reports any id it can't fetch — so re-running is always safe.

## Development

```bash
uv run ruff format .
uv run ruff check --fix .
uv run ty check .
uv run pytest -q
```
