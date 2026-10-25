"""Databases for the tests: a fresh SQLite file per test, or the Postgres of docker-compose."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from contract_assistant.access import Actor
from contract_assistant.db.engine import Database
from contract_assistant.db.models import ADMIN, USER, User

# The Postgres the integration tests use: a database of its own is made per test run on it.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://contract_assistant:contract_assistant@127.0.0.1:5433/postgres",
)
# A real hash of "correct horse battery" would be slow to make in every test; services that
# check passwords make their own.
UNUSABLE_HASH = "!"


def sqlite_database(directory: Path) -> Database:
    """Return a database with every table, in a new SQLite file under directory."""
    database = Database(f"sqlite:///{directory / 'app.db'}")
    database.create_all()
    return database


def add_user(
    database: Database,
    email: str,
    name: str | None = None,
    role: str = USER,
    user_id: str | None = None,
) -> Actor:
    """Add an active account straight to the table and return it as an Actor."""
    with database.session() as session:
        user = User(
            email=email.lower(),
            name=name or email.split("@")[0].title(),
            password_hash=UNUSABLE_HASH,
            role=role,
        )
        if user_id is not None:
            user.id = user_id
        session.add(user)
        session.flush()
        return Actor(id=user.id, email=user.email, name=user.name, role=user.role)


def add_admin(database: Database, email: str = "admin@example.com", **kwargs) -> Actor:
    return add_user(database, email, role=ADMIN, **kwargs)


def postgres_database() -> Database:
    """Create an empty Postgres database for this run and return it migrated to the newest schema.

    Skips the test when the server of TEST_DATABASE_URL is not reachable (`make db-up`).
    """
    server = make_url(TEST_DATABASE_URL)
    name = f"contract_assistant_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(server, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as error:  # noqa: BLE001 - any failure to reach the server skips
        admin.dispose()
        pytest.skip(f"Postgres not reachable for integration tests: {error.__class__.__name__}")
    admin.dispose()
    database = Database(server.set(database=name).render_as_string(hide_password=False))
    database.upgrade()
    database.name = name  # type: ignore[attr-defined]
    return database


def drop_postgres_database(database: Database) -> None:
    """Drop a database made by postgres_database."""
    database.dispose()
    admin = create_engine(make_url(TEST_DATABASE_URL), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{database.name}" WITH (FORCE)'))
    admin.dispose()
