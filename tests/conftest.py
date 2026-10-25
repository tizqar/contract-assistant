import json
import os
from pathlib import Path

import pytest

from tests.support.db import drop_postgres_database, postgres_database, sqlite_database

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def clause_types() -> dict:
    return json.loads((ROOT / "data" / "clause_types.json").read_text())


@pytest.fixture(scope="session")
def parity_contracts() -> dict:
    """Contract texts of the parity fixture, keyed by CUAD title."""
    data = json.loads((FIXTURES / "parity_contracts.json").read_text())["data"]
    return {a["title"]: a["paragraphs"][0]["context"] for a in data}


@pytest.fixture
def database(tmp_path):
    """An empty app database with every table, in SQLite, for one test."""
    db = sqlite_database(tmp_path)
    yield db
    db.dispose()


@pytest.fixture
def pg_database():
    """An empty Postgres database of its own, migrated with the real migrations, for one test."""
    db = postgres_database()
    yield db
    drop_postgres_database(db)
