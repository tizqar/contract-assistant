"""Alembic's entry point: run the migrations on the app's connection or on a URL."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from contract_assistant.db.models import Base, UTCDateTime

config = context.config
target_metadata = Base.metadata


def _render_item(kind: str, item, autogen_context):
    """Write the app's UTC time type as the plain column type it is in the database."""
    if kind == "type" and isinstance(item, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_item=_render_item,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.begin() as connection:
        _run(connection)
    engine.dispose()


run_migrations()
