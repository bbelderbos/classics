import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "classics.db"
engine = create_engine(f"sqlite:///{DB_PATH}")


def _now() -> datetime:
    return datetime.now(UTC)


class SearchEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query: str
    results: int
    created_at: datetime = Field(default_factory=_now)


class QuoteEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    query: str
    author: str
    title: str
    text: str
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
