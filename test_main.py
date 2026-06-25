import numpy as np

from main import Passage, chunk_text, preview, read_library, retrieve


def test_read_library_parses_ids_and_ignores_comments(tmp_path):
    f = tmp_path / "lib.txt"
    f.write_text("# header\n2600  # War and Peace\n\n28054\n")
    assert read_library(f) == [2600, 28054]


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
