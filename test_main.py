import numpy as np
import pytest
import requests

import main
from main import (
    Passage,
    best_excerpt,
    chunk_text,
    diversify,
    humanize_author,
    preview,
    read_library,
    reflow,
    retrieve,
    search_passages,
)


def test_diversify_caps_passages_per_book():
    # one book monopolises the ranking; cap should let others through
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
    books = {0: "A", 1: "A", 2: "A", 3: "B", 4: "C"}
    out = diversify(ranked, lambda i: books[i], k=3, per_book=2)
    assert [i for i, _ in out] == [0, 1, 3]  # A capped at 2, then B


def test_diversify_no_cap_when_per_book_zero():
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]
    out = diversify(ranked, lambda _: "A", k=3, per_book=0)
    assert [i for i, _ in out] == [0, 1, 2]


def test_search_passages_caps_by_author_across_titles(monkeypatch):
    # one author spread over several titles must still respect the per-book cap
    passages = [
        Passage("World as Will (Vol. 1)", "Schopenhauer, Arthur", "", "a"),
        Passage("World as Will (Vol. 2)", "Schopenhauer, Arthur", "", "b"),
        Passage("Essays of Schopenhauer", "Schopenhauer, Arthur", "", "c"),
        Passage("War and Peace", "Tolstoy, Leo", "", "d"),
    ]
    pool = [(0, 0.9), (1, 0.85), (2, 0.8), (3, 0.75)]
    monkeypatch.setattr("main.retrieve", lambda q, v, k: pool)
    out = search_passages("q", passages, np.empty((4, 1)), k=5, per_book=2, floor=0)
    assert [i for i, _ in out] == [0, 1, 3]  # Schopenhauer capped at 2, then Tolstoy


def test_search_passages_drops_adjacent_overlapping_chunks(monkeypatch):
    # consecutive chunks from one book share a paragraph (overlap=1) and embed alike;
    # only the higher-scoring of an adjacent pair should surface
    passages = [
        Passage("Poe Vol. 1", "Poe, Edgar Allan", "", "a", book_id=1, offset=253),
        Passage("Poe Vol. 1", "Poe, Edgar Allan", "", "b", book_id=1, offset=254),
        Passage("War and Peace", "Tolstoy, Leo", "", "c", book_id=2, offset=10),
    ]
    pool = [(0, 0.9), (1, 0.88), (2, 0.7)]
    monkeypatch.setattr("main.retrieve", lambda q, v, k: pool)
    out = search_passages("q", passages, np.empty((3, 1)), k=5, per_book=2, floor=0)
    assert [i for i, _ in out] == [0, 2]  # chunk 254 dropped as adjacent to 253


def test_search_passages_returns_empty_when_best_match_below_min_score(monkeypatch):
    passages = [Passage("Walden", "Thoreau", "", "a")]
    monkeypatch.setattr("main.retrieve", lambda q, v, k: [(0, 0.28)])
    assert search_passages("buy bitcoin", passages, np.empty((1, 1))) == []


def test_read_library_parses_ids_and_ignores_comments(tmp_path):
    f = tmp_path / "lib.txt"
    f.write_text("# header\n2600  # War and Peace\n\n28054\n")
    assert read_library(f) == [2600, 28054]


def test_sync_deletes_stale_and_builds_desired(monkeypatch, tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    for book_id in (1, 2):  # both present on disk, with their full fileset
        for ext in ("txt", "npy", "chunks.json", "meta.json"):
            (books / f"{book_id}.{ext}").write_text("x")
    monkeypatch.setattr(main, "BOOKS_DIR", books)
    monkeypatch.setattr(main, "read_library", lambda: [1, 3])  # 2 dropped, 3 added
    built: list[int] = []
    monkeypatch.setattr(main, "index_books", built.extend)

    main.sync_library()

    assert not list(books.glob("2.*"))  # every stale file deleted, not just the .npy
    assert len(list(books.glob("1.*"))) == 4  # desired book's fileset untouched
    assert built == [1, 3]  # full desired list handed to the indexer


def test_sync_refuses_to_wipe_books_when_library_empty(monkeypatch, tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    (books / "1.npy").write_text("x")
    monkeypatch.setattr(main, "BOOKS_DIR", books)
    monkeypatch.setattr(main, "read_library", lambda: [])
    monkeypatch.setattr(main, "index_books", lambda ids: pytest.fail("must not index"))

    main.sync_library()

    assert (books / "1.npy").exists()  # nothing deleted


def test_passage_cite_combines_author_title_and_label():
    p = Passage("War and Peace", "Tolstoy", "BOOK XI — CHAPTER IX", "...")
    assert p.cite() == "Tolstoy · War and Peace — BOOK XI — CHAPTER IX"
    bare = Passage("Walden", "", "", "...")
    assert bare.cite() == "Walden"


def test_chunk_text_packs_to_word_budget():
    text = "\n\n".join("word " * 100 for _ in range(10))
    chunks = chunk_text(text, target_words=300, overlap=0)
    assert len(chunks) == 4  # 10 paragraphs of 100 words, 300/chunk
    assert all(c.text.strip() for c in chunks)


def test_chunk_text_overlap_repeats_previous_paragraph():
    paras = [f"paragraph number {i} " * 80 for i in range(4)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, target_words=80, overlap=1)
    # each chunk after the first should start with the prior chunk's last paragraph
    assert "number 0" in chunks[1].text


def test_chunk_text_keeps_trailing_remainder():
    text = "short one\n\nshort two"
    chunks = chunk_text(text, target_words=1000)
    assert [c.text for c in chunks] == ["short one\n\nshort two"]


def test_chunk_text_tags_section_and_chapter():
    text = (
        "BOOK ONE\n\nCHAPTER I\n\n"
        + ("alpha " * 700)
        + "\n\nCHAPTER II\n\n"
        + ("beta " * 50)
    )
    chunks = chunk_text(text, target_words=600, overlap=0)
    assert chunks[0].label == "BOOK ONE — CHAPTER I"
    assert chunks[-1].label == "BOOK ONE — CHAPTER II"


def test_chunk_text_relabels_after_emit_with_overlap():
    # default overlap=1 must not freeze the label on the first chunk
    body = "\n\n".join("alpha " * 200 for _ in range(4))
    text = f"CHAPTER I\n\n{body}\n\nCHAPTER II\n\n{body}"
    chunks = chunk_text(text, target_words=600)
    assert chunks[0].label == "CHAPTER I"
    assert any(c.label == "CHAPTER II" for c in chunks)


def test_chunk_text_ignores_prose_starting_with_keyword():
    text = "Part of the long road home wound " * 20
    chunks = chunk_text(text, target_words=50)
    assert chunks[0].label == ""  # too long to be a heading


def test_retrieve_ranks_by_cosine_similarity(monkeypatch):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    monkeypatch.setattr("main.embed", lambda texts: np.array([[1.0, 0.0]]))

    results = retrieve("q", vectors, k=2)

    assert [i for i, _ in results] == [0, 2]
    assert results[0][1] > results[1][1]


def test_preview_truncates_and_collapses_whitespace():
    assert preview("hello    world\n\nagain", width=11) == "hello world…"


def test_reflow_unwraps_lines_but_keeps_paragraphs():
    text = "one two\nthree\n\nfour\nfive"
    assert reflow(text) == "one two three\n\nfour five"


def test_humanize_author_flips_single_last_first():
    assert humanize_author("Schopenhauer, Arthur") == "Arthur Schopenhauer"
    assert humanize_author("Homer") == "Homer"
    assert humanize_author("Beaumont, Francis, Fletcher, John") == (
        "Beaumont, Francis, Fletcher, John"
    )


def test_passage_share_quotes_text_and_attributes():
    p = Passage("The Republic", "Plato", "BOOK II", "the just\nman")
    assert p.share() == "“the just man”\n\n— Plato, The Republic (BOOK II)"
    bare = Passage("Walden", "", "", "wild\nthings")
    assert bare.share() == "“wild things”\n\n— Walden"


def test_best_excerpt_picks_query_relevant_sentence(monkeypatch):
    text = "Alpha beta gamma. The cat sat on the mat. Delta epsilon zeta."
    vectors = {
        "Alpha beta gamma.": [1.0, 0.0],
        "The cat sat on the mat.": [0.0, 1.0],
        "Delta epsilon zeta.": [1.0, 0.0],
        "cats": [0.0, 1.0],
    }
    monkeypatch.setattr(
        "main.embed", lambda texts: np.array([vectors[t] for t in texts])
    )
    # the word budget keeps it to the single best-matching sentence
    assert best_excerpt(text, "cats", max_words=6) == "The cat sat on the mat."


class FakeResponse:
    def __init__(self, json_data=None, http_error=None):
        self._json = json_data if json_data is not None else {}
        self._http_error = http_error

    def json(self):
        return self._json

    def raise_for_status(self):
        if self._http_error:
            raise self._http_error


def test_search_book_raises_on_http_error(monkeypatch):
    # a 5xx/HTML error page must surface as HTTPError, not a bare KeyError on "results"
    err = requests.HTTPError("500")
    monkeypatch.setattr(
        main.requests, "get", lambda *a, **k: FakeResponse(http_error=err)
    )
    with pytest.raises(requests.HTTPError):
        main.search_book("anything")


def test_search_book_returns_empty_when_results_missing(monkeypatch):
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: FakeResponse({}))
    assert main.search_book("nothing here") == []


def test_search_book_passes_term_as_param_with_timeout(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse({"results": []})

    monkeypatch.setattr(main.requests, "get", fake_get)
    main.search_book("war & peace")  # the & must not corrupt the query string
    assert captured["params"] == {"search": "war & peace"}
    assert captured["timeout"]  # set, non-zero


def test_book_metadata_raises_on_http_error(monkeypatch):
    err = requests.HTTPError("404")
    monkeypatch.setattr(
        main.requests, "get", lambda *a, **k: FakeResponse(http_error=err)
    )
    with pytest.raises(requests.HTTPError):
        main.book_metadata(123)


def test_book_metadata_uses_timeout(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse({"title": "Walden", "authors": [{"name": "Thoreau"}]})

    monkeypatch.setattr(main.requests, "get", fake_get)
    assert main.book_metadata(123) == ("Walden", "Thoreau", "")
    assert captured["timeout"]


def test_book_metadata_strips_autogenerated_summary_suffix(monkeypatch):
    def fake_get(url, **kwargs):
        return FakeResponse(
            {
                "title": "Walden",
                "authors": [{"name": "Thoreau"}],
                "summaries": [f"A life in the woods. {main.SUMMARY_SUFFIX}"],
            }
        )

    monkeypatch.setattr(main.requests, "get", fake_get)
    assert main.book_metadata(123) == ("Walden", "Thoreau", "A life in the woods.")


def test_clean_summary_handles_empty():
    assert main.clean_summary("") == ""
    assert main.clean_summary(main.SUMMARY_SUFFIX) == ""
