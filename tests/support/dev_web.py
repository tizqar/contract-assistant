"""Development web app: demo caches, fake router, fake slow extractor, echoing generator.

Its database is a SQLite file in the same temporary folder; the first visit asks for the
first admin account.

Run with: PYTHONPATH=. .venv/bin/uvicorn tests.support.dev_web:app --reload --port 7861
"""

import dataclasses
import tempfile
from pathlib import Path

from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import load_settings
from contract_assistant.db.engine import connect
from contract_assistant.extract.extractor import Extractor
from contract_assistant.review import load_rules
from contract_assistant.search import PassageSearch
from contract_assistant.services import Services
from contract_assistant.store import Store
from contract_assistant.web.app import create_app
from tests.support.fakes import EchoTextClient, FakeEncoder, FakeRouter, slow_fake_qa

DEV_CACHE_DIR = Path(tempfile.gettempdir()) / "contract-assistant-dev"


def dev_services() -> Services:
    """Return services backed by fakes, with uploads kept in a fixed temporary folder."""
    settings = dataclasses.replace(load_settings(), cache_dir=DEV_CACHE_DIR)
    clause_types = app_clause_types(settings)
    extra = load_extra_clause_types(settings.extra_clause_types_path)
    extractor = Extractor(slow_fake_qa(), clause_types, settings, device="fake")
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    unvalidated = frozenset(clause_type.name for clause_type in extra)
    return Services(
        store,
        FakeRouter(),
        extractor,
        clause_types,
        settings,
        PassageSearch(FakeEncoder()),
        unvalidated,
        gen=EchoTextClient(),
        rules=load_rules(known_types=[c.name for c in clause_types]),
    )


app = create_app(dev_services(), connect(f"sqlite:///{DEV_CACHE_DIR / 'dev.db'}"))
