import logging
from datetime import UTC, datetime
from pathlib import Path

from typing import Any

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "classics.db"
engine = create_engine(f"sqlite:///{DB_PATH}")


def _now() -> datetime:
    return datetime.now(UTC)


class SearchEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query: str
    # JSON: [{"id", "label", "offset", "score"}, ...] of the shown passages
    results: str
    created_at: datetime = Field(default_factory=_now)


class QuoteEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query: str
    author: str
    title: str
    text: str
    created_at: datetime = Field(default_factory=_now)


class PdfEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query: str
    passages: int
    created_at: datetime = Field(default_factory=_now)


class ReadEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query: str
    author: str
    title: str
    # dwell time on an expanded passage, in milliseconds
    ms: int
    created_at: datetime = Field(default_factory=_now)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def record(event: SQLModel) -> None:
    try:
        with Session(engine) as session:
            session.add(event)
            session.commit()
    except Exception:
        logger.exception("failed to record %s", type(event).__name__)


WEEK = "datetime('now', '-7 days')"


def stats() -> dict[str, Any]:
    with engine.connect() as conn:

        def one(sql: str) -> int | float:
            return conn.execute(text(sql)).scalar() or 0

        def rows(sql: str) -> list[dict[str, Any]]:
            return [dict(r) for r in conn.execute(text(sql)).mappings()]

        def counts(window: str) -> dict[str, Any]:
            return {
                "queries": one(f"SELECT count(*) FROM searchevent {window}"),
                "reads": one(f"SELECT count(*) FROM readevent {window}"),
                "shares": one(f"SELECT count(*) FROM quoteevent {window}"),
                "pdfs": one(f"SELECT count(*) FROM pdfevent {window}"),
                "reading_minutes": round(
                    one(f"SELECT coalesce(sum(ms), 0) FROM readevent {window}")
                    / 60000.0,
                    1,
                ),
            }

        week = counts(f"WHERE created_at >= {WEEK}")
        week["avg_read_seconds"] = round(
            one(
                f"SELECT coalesce(avg(ms), 0) FROM readevent WHERE created_at >= {WEEK}"
            )
            / 1000.0,
            1,
        )
        return {
            "week": week,
            "all_time": counts(""),
            "top_searches": rows(
                f"SELECT query, count(*) AS n FROM searchevent "
                f"WHERE created_at >= {WEEK} GROUP BY query ORDER BY n DESC, query LIMIT 15"
            ),
            "top_reads": rows(
                f"SELECT author, title, count(*) AS reads, "
                f"round(avg(ms) / 1000.0, 1) AS avg_seconds FROM readevent "
                f"WHERE created_at >= {WEEK} GROUP BY author, title "
                f"HAVING reads >= 3 ORDER BY avg_seconds DESC LIMIT 10"
            ),
        }
