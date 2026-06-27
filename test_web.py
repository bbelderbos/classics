from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from sqlmodel import create_engine

import db
import web
from db import PdfEvent, QuoteEvent, ReadEvent, SearchEvent


@pytest.fixture
def store(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "engine", engine)
    db.init_db()
    return db


def test_stats_empty_db_returns_zeros(store):
    s = store.stats()
    assert s["week"] == {
        "queries": 0,
        "reads": 0,
        "shares": 0,
        "pdfs": 0,
        "reading_minutes": 0,
        "avg_read_seconds": 0,
    }
    assert s["top_searches"] == []
    assert s["top_reads"] == []


def test_stats_counts_each_event_type(store):
    store.record(SearchEvent(query="q", results="[]"))
    store.record(QuoteEvent(query="q", author="a", title="t", text="x"))
    store.record(PdfEvent(query="q", passages=2))
    store.record(ReadEvent(query="q", author="a", title="t", ms=1000))
    week = store.stats()["week"]
    assert (week["queries"], week["shares"], week["pdfs"], week["reads"]) == (
        1,
        1,
        1,
        1,
    )


def test_stats_reading_time_converts_ms(store):
    store.record(ReadEvent(query="q", author="a", title="t", ms=30000))
    store.record(ReadEvent(query="q", author="a", title="t", ms=90000))
    week = store.stats()["week"]
    assert week["reading_minutes"] == 2.0  # (30s + 90s) / 60
    assert week["avg_read_seconds"] == 60.0


def test_stats_top_searches_ranked_by_count(store):
    for _ in range(3):
        store.record(SearchEvent(query="on death", results="[]"))
    store.record(SearchEvent(query="on hope", results="[]"))
    top = store.stats()["top_searches"]
    assert top[0] == {"query": "on death", "n": 3}
    assert {"query": "on hope", "n": 1} in top


def test_stats_top_reads_needs_three_reads_and_sorts_by_dwell(store):
    for ms in (10000, 20000, 30000):  # avg 20s, qualifies
        store.record(ReadEvent(query="q", author="Plato", title="Republic", ms=ms))
    for ms in (99000, 99000):  # only two reads, excluded despite long dwell
        store.record(ReadEvent(query="q", author="Homer", title="Iliad", ms=ms))
    top = store.stats()["top_reads"]
    assert [r["title"] for r in top] == ["Republic"]
    assert top[0]["reads"] == 3
    assert top[0]["avg_seconds"] == 20.0


def test_stats_window_separates_week_from_all_time(store):
    old = datetime.now(UTC) - timedelta(days=10)
    store.record(SearchEvent(query="old", results="[]", created_at=old))
    store.record(SearchEvent(query="new", results="[]"))
    s = store.stats()
    assert s["week"]["queries"] == 1
    assert s["all_time"]["queries"] == 2


def _auth(username="admin", password="secret"):
    return HTTPBasicCredentials(username=username, password=password)


def test_stats_auth_unconfigured_returns_503(monkeypatch):
    monkeypatch.delenv("STATS_PASSWORD", raising=False)
    with pytest.raises(HTTPException) as exc:
        web.require_stats_auth(_auth())
    assert exc.value.status_code == 503


def test_stats_auth_rejects_wrong_credentials(monkeypatch):
    monkeypatch.setenv("STATS_USER", "admin")
    monkeypatch.setenv("STATS_PASSWORD", "secret")
    with pytest.raises(HTTPException) as exc:
        web.require_stats_auth(_auth(password="wrong"))
    assert exc.value.status_code == 401


def test_stats_auth_accepts_correct_credentials(monkeypatch):
    monkeypatch.setenv("STATS_USER", "admin")
    monkeypatch.setenv("STATS_PASSWORD", "secret")
    assert web.require_stats_auth(_auth()) is None


def test_read_endpoint_records_dwell(store):
    web.read(web.ReadIn(query="q", author="a", title="t", ms=5000))
    assert store.stats()["week"]["reads"] == 1


def test_pdf_endpoint_records_only_after_successful_generation(store, monkeypatch):
    monkeypatch.setattr(web, "weave_pdf", lambda doc, css: b"%PDF-")
    web.pdf(web.PdfIn(query="q", items=[web.PdfItem(text="x"), web.PdfItem(text="y")]))
    assert store.stats()["week"]["pdfs"] == 1


def test_pdf_endpoint_does_not_record_on_failure(store, monkeypatch):
    def boom(doc, css):
        raise RuntimeError("no weasyprint")

    monkeypatch.setattr(web, "weave_pdf", boom)
    with pytest.raises(RuntimeError):
        web.pdf(web.PdfIn(query="q", items=[web.PdfItem(text="x")]))
    assert store.stats()["week"]["pdfs"] == 0
