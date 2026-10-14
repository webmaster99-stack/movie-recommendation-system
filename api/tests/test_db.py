"""The tables are described twice: in the SQL migrations, which create them in Supabase, and
in `recsys_api.db`, which queries them. These tests keep the two descriptions in step."""

import re
import uuid

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from recsys.utils.config import PROJECT_ROOT
from recsys_api.db import Base, Rating, make_engine, upsert

MIGRATIONS = PROJECT_ROOT / "supabase" / "migrations"
_CREATE_TABLE = re.compile(r"create table public\.(\w+) \((.*?)\n\);", re.DOTALL)
_ALTER_TABLE = re.compile(r"alter table public\.(\w+)(.*?);", re.DOTALL)
_ADD_COLUMN = re.compile(r"add column (\w+)")
_CONSTRAINTS = ("primary", "constraint", "check", "foreign", "unique")
_NOT_USED_BY_THE_API = {"profiles"}  # maintained by a trigger, read by the frontend


def migration_tables() -> dict[str, set[str]]:
    sql = "\n".join(path.read_text(encoding="utf-8") for path in sorted(MIGRATIONS.glob("*.sql")))
    tables = {}
    for name, body in _CREATE_TABLE.findall(sql):
        words = [line.split()[0] for line in body.splitlines() if line.strip()]
        tables[name] = {word for word in words if word not in _CONSTRAINTS}
    for name, body in _ALTER_TABLE.findall(sql):  # columns added by a later migration
        tables[name] |= set(_ADD_COLUMN.findall(body))
    return tables


def test_migrations_and_python_describe_the_same_tables() -> None:
    in_sql = migration_tables()
    in_python = {
        table.name: {c.name for c in table.columns} for table in Base.metadata.tables.values()
    }
    assert set(in_sql) - _NOT_USED_BY_THE_API == set(in_python)
    for name, columns in in_python.items():
        assert in_sql[name] == columns, name


def test_every_table_has_row_level_security() -> None:
    sql = "\n".join(path.read_text(encoding="utf-8") for path in sorted(MIGRATIONS.glob("*.sql")))
    for table in migration_tables():
        assert f"alter table public.{table} enable row level security;" in sql, table


def test_upsert_inserts_then_overwrites_only_the_named_columns(engine: Engine) -> None:
    user = uuid.uuid4()
    values = {"user_id": user, "movie_id": 1, "rating": 4.0}
    with Session(engine) as session:
        upsert(session, Rating, values, update=["rating"])
        session.commit()
        created = session.scalars(select(Rating)).one().created_at
        upsert(session, Rating, values | {"rating": 2.5}, update=["rating"])
        session.commit()
        session.expire_all()
        row = session.scalars(select(Rating)).one()
        assert (row.rating, row.created_at) == (2.5, created)


@pytest.mark.parametrize("url", ["postgresql://u:p@host:6543/db", "postgres://u:p@host/db"])
def test_supabase_urls_get_the_psycopg_driver(url: str) -> None:
    engine = make_engine(url)  # connects lazily, so no database is needed
    assert engine.url.drivername == "postgresql+psycopg"
    assert (engine.url.username, engine.url.host) == ("u", "host")
