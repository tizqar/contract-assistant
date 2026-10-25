"""The schema: the migrations build the tables the models describe, on Postgres and SQLite."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from contract_assistant.db.models import Base, Event, User
from tests.support.db import add_user


@pytest.mark.pg
def test_the_migrations_build_exactly_the_tables_of_the_models(pg_database):
    with pg_database.engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


@pytest.mark.pg
def test_postgres_keeps_analyses_as_jsonb_and_files_as_bytea(pg_database):
    columns = {
        (table, column["name"]): str(column["type"])
        for table in ("analyses", "blobs")
        for column in inspect(pg_database.engine).get_columns(table)
    }
    assert columns[("analyses", "data")] == "JSONB"
    assert columns[("blobs", "data")] == "BYTEA"


def test_sqlite_gets_every_table(database):
    assert set(inspect(database.engine).get_table_names()) == set(Base.metadata.tables)


def test_times_come_back_in_utc_on_sqlite(database):
    belgrade = timezone(timedelta(hours=2))
    actor = add_user(database, "ana@example.com")
    with database.session() as session:
        session.add(Event(action="user.login", actor_id=actor.id, at=datetime(2026, 10, 25, 12, tzinfo=belgrade)))
    with database.session() as session:
        event = session.scalars(select(Event)).one()
    assert event.at == datetime(2026, 10, 25, 10, tzinfo=UTC)
    assert event.at.tzinfo is UTC


def test_a_time_without_a_zone_is_refused(database):
    actor = add_user(database, "ana@example.com")
    with pytest.raises(Exception, match="time zone"), database.session() as session:
        session.add(Event(action="user.login", actor_id=actor.id, at=datetime(2026, 10, 25)))  # noqa: DTZ001


def test_sqlite_enforces_foreign_keys_and_checks(database):
    with pytest.raises(IntegrityError), database.session() as session:
        session.add(Event(action="user.login", actor_id="nobody"))
    with pytest.raises(IntegrityError), database.session() as session:
        session.add(User(email="x@example.com", name="X", password_hash="!", role="owner"))
