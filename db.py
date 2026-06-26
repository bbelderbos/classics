import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, text

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


def init_db() -> None:
    with engine.begin() as conn:
        cols = conn.exec_driver_sql("PRAGMA table_info(searchevent)").fetchall()
        if any(c[1] == "results" and c[2] == "INTEGER" for c in cols):
            conn.execute(
                text("DROP TABLE searchevent")
            )  # old count schema, rebuild below
    SQLModel.metadata.create_all(engine)


def record(event: SQLModel) -> None:
    try:
        with Session(engine) as session:
            session.add(event)
            session.commit()
    except Exception:
        logger.exception("failed to record %s", type(event).__name__)
