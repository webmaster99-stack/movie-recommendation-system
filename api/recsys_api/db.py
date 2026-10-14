"""Database tables and the engine.

The schema itself is owned by `supabase/migrations/`: plain SQL, because row-level security,
the foreign keys to Supabase's `auth.users` and the profile trigger live there. The classes
below describe the same tables so the API can query them, and `tests/test_db.py` fails if a
table or column exists on one side only.

Types are chosen to also run on SQLite, which the tests and a local run without Postgres use.
The API connects as the database owner, which bypasses row-level security; the policies
protect the tables from direct access with the public Supabase key.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    create_engine,
    event,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

_Timestamp = DateTime(timezone=True)
_BigIntKey = BigInteger().with_variant(Integer, "sqlite")  # SQLite autoincrements INTEGER only
_IntArray = ARRAY(Integer).with_variant(JSON, "sqlite")


def now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """SQLite returns timestamps without a time zone; they were stored as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (Index("ratings_user_id_updated_at_idx", "user_id", "updated_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rating: Mapped[float] = mapped_column(Numeric(2, 1, asdecimal=False))
    created_at: Mapped[datetime] = mapped_column(_Timestamp, default=now)
    updated_at: Mapped[datetime] = mapped_column(_Timestamp, default=now)


class MovieMetadata(Base):
    """What TMDB says about a movie. A cache: rows are refreshed, and deleted when too old."""

    __tablename__ = "movie_metadata"
    __table_args__ = (Index("movie_metadata_fetched_at_idx", "fetched_at"),)

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tmdb_id: Mapped[int] = mapped_column(Integer)
    found: Mapped[bool]  # False: TMDB has no such movie, so don't ask again until refresh
    title: Mapped[str | None] = mapped_column(Text)
    overview: Mapped[str | None] = mapped_column(Text)
    poster_path: Mapped[str | None] = mapped_column(Text)
    backdrop_path: Mapped[str | None] = mapped_column(Text)
    release_date: Mapped[str | None] = mapped_column(String(10))
    runtime: Mapped[int | None] = mapped_column(Integer)
    vote_average: Mapped[float | None] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(_Timestamp)


class RecRequest(Base):
    """One answered `/recommendations` call: what was shown, by which model, how fast."""

    __tablename__ = "rec_requests"
    __table_args__ = (Index("rec_requests_created_at_idx", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    model: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(Text)
    strategy: Mapped[str] = mapped_column(Text)  # personalised | popular
    n_liked: Mapped[int] = mapped_column(Integer)
    movie_ids: Mapped[list[int]] = mapped_column(_IntArray)  # in the order shown
    model_ms: Mapped[float] = mapped_column(Float)  # scoring only
    latency_ms: Mapped[float] = mapped_column(Float)  # whole request, incl. database and TMDB
    created_at: Mapped[datetime] = mapped_column(_Timestamp, default=now)
    # Where the shown movies sit in the popularity order on average, from 0 (the most popular
    # movie) to 1 (the least), and the size of the catalog they were picked from. Both are
    # null on rows written before monitoring existed.
    popularity_percentile: Mapped[float | None] = mapped_column(Float)
    catalog_size: Mapped[int | None] = mapped_column(Integer)


class Event(Base):
    """What a user did with a recommendation: clicked it, or rated it afterwards."""

    __tablename__ = "events"
    __table_args__ = (Index("events_created_at_idx", "created_at"),)

    id: Mapped[int] = mapped_column(_BigIntKey, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    request_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    event_type: Mapped[str] = mapped_column(Text)  # click | rating
    movie_id: Mapped[int] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)  # 0-based place in the list shown
    rating: Mapped[float | None] = mapped_column(Numeric(2, 1, asdecimal=False))
    created_at: Mapped[datetime] = mapped_column(_Timestamp, default=now)


class RequestLog(Base):
    """One answered request: which endpoint, what status, how long. No user and no ids."""

    __tablename__ = "request_log"
    __table_args__ = (Index("request_log_created_at_idx", "created_at"),)

    id: Mapped[int] = mapped_column(_BigIntKey, primary_key=True, autoincrement=True)
    method: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(Text)  # the template, e.g. /movies/{movie_id}
    status: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(_Timestamp, default=now)


def upsert(session: Session, table: type[Base], values: dict[str, Any], update: list[str]) -> None:
    """Insert a row, or on a primary-key clash overwrite the `update` columns. One statement,
    so two requests for the same row can't both try to insert it."""
    dialect = session.get_bind().dialect.name
    insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
    statement = insert(table).values(**values)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[column.name for column in table.__table__.primary_key],
            set_={name: statement.excluded[name] for name in update},
        )
    )


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite"):
        in_memory = database_url in ("sqlite://", "sqlite:///:memory:")
        engine = create_engine(
            database_url,
            # SQLite allows one writer at a time; wait for it instead of failing.
            connect_args={"check_same_thread": False, "timeout": 30},
            # An in-memory database exists per connection, so every thread must share one.
            **({"poolclass": StaticPool} if in_memory else {}),
        )
        if not in_memory:
            # Write-ahead log: readers no longer block the writer, nor the writer them.
            event.listen(
                engine, "connect", lambda dbapi, _: dbapi.execute("PRAGMA journal_mode=WAL")
            )
        Base.metadata.create_all(engine)
        return engine
    # Supabase hands out `postgresql://` URLs; SQLAlchemy needs the driver named.
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            database_url = "postgresql+psycopg://" + database_url.removeprefix(prefix)
    return create_engine(
        database_url,
        pool_size=3,
        max_overflow=2,
        pool_pre_ping=True,  # the free host sleeps, and its connections die meanwhile
        # Supabase's pooler in transaction mode can't keep prepared statements.
        connect_args={"prepare_threshold": None},
    )
