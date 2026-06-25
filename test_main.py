import numpy as np

from main import chunk_text, preview, retrieve


def test_chunk_text_packs_to_word_budget():
    text = "\n\n".join("word " * 100 for _ in range(10))
    chunks = chunk_text(text, target_words=300, overlap=0)
    assert len(chunks) == 4  # 10 paragraphs of 100 words, 300/chunk
    assert all(c.strip() for c in chunks)


def test_chunk_text_overlap_repeats_previous_paragraph():
    paras = [f"paragraph number {i} " * 80 for i in range(4)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, target_words=80, overlap=1)
    # each chunk after the first should start with the prior chunk's last paragraph
    assert "number 0" in chunks[1]


def test_chunk_text_keeps_trailing_remainder():
    text = "short one\n\nshort two"
    assert chunk_text(text, target_words=1000) == ["short one\n\nshort two"]


def test_retrieve_ranks_by_cosine_similarity(monkeypatch):
    chunks = ["a", "b", "c"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    monkeypatch.setattr("main.embed", lambda texts: np.array([[1.0, 0.0]]))

    results = retrieve("q", chunks, vectors, k=2)

    assert [i for i, _ in results] == [0, 2]
    assert results[0][1] > results[1][1]


def test_preview_truncates_and_collapses_whitespace():
    assert preview("hello    world\n\nagain", width=11) == "hello world…"
