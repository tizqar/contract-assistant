"""The connection to the database, its sessions and its migrations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from contract_assistant.db.models import Base

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
SQLITE = "sqlite"
SQLITE_TIMEOUT_S = 30
POOL_SIZE = 10
UNAVAILABLE = (
    "The database at {where} is not reachable. Start it with `make db-up` "
    "(docker compose up -d db), or set DATABASE_URL in .env."
)


class DatabaseUnavailableError(RuntimeError):
    """Raised when the database cannot be reached."""


class Database:
    """An engine and the sessions over it; one per app."""

    def __init__(self, url: str, echo: bool = False):
        self.url = url
        self.engine = create_engine(url, echo=echo, **_options(url))
        if self.is_sqlite:
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @property
    def is_sqlite(self) -> bool:
        return self.engine.dialect.name == SQLITE

    @property
    def where(self) -> str:
        """The database's address without its password, for messages and logs."""
        return self.engine.url.render_as_string(hide_password=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A unit of work: committed when the block ends, rolled back when it raises."""
        with self._sessions() as session:
            try:
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise

    def check(self) -> None:
        """Connect once; raise DatabaseUnavailableError with what to do when that fails."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except OperationalError as error:
            raise DatabaseUnavailableError(UNAVAILABLE.format(where=self.where)) from error

    def create_all(self) -> None:
        """Create every table from the models, for tests and throwaway databases."""
        Base.metadata.create_all(self.engine)

    def upgrade(self) -> None:
        """Bring the schema to the newest migration."""
        from alembic import command

        with self.engine.begin() as connection:
            config = alembic_config()
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    def dispose(self) -> None:
        """Close every pooled connection."""
        self.engine.dispose()


def alembic_config(url: str | None = None):
    """Return the Alembic configuration of the migrations, for a URL or a given connection."""
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url is not None:
        config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def _options(url: str) -> dict:
    if url.startswith(SQLITE):
        return {"connect_args": {"check_same_thread": False, "timeout": SQLITE_TIMEOUT_S}}
    return {"pool_pre_ping": True, "pool_size": POOL_SIZE, "max_overflow": POOL_SIZE}


def _sqlite_pragmas(connection, _record) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def connect(url: str, *, migrate: bool = True) -> Database:
    """Open the database at url and bring its schema up to date; SQLite gets the tables."""
    database = Database(url)
    database.check()
    if database.is_sqlite:
        database.create_all()
    elif migrate:
        database.upgrade()
    return database


__all__ = ["Database", "DatabaseUnavailableError", "Engine", "alembic_config", "connect"]
