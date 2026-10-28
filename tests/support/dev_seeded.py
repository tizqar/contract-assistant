"""Development web app with accounts, folders and a share, over fake models and a SQLite file.

Run with:
    PYTHONPATH=. DEV_DIR=/some/folder .venv/bin/uvicorn tests.support.dev_seeded:app --port 7881

DEV_DATABASE_URL puts it on another database, such as a database of its own in the
Postgres of docker-compose; by default it is a SQLite file in DEV_DIR.

Accounts (password "correct horse battery" for each): ana@example.com (administrator),
marko@example.com and jelena@example.com (users). Marko owns the three demo contracts: the
short one in "Supply deals", which is shared with Jelena to edit, the medium one in
"Sponsorship", and the long one at the top of his space. DEV_DIR keeps the database between
runs; a new folder starts again from these.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

from contract_assistant import accounts, files
from contract_assistant.access import Actor
from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import load_settings
from contract_assistant.db.engine import connect
from contract_assistant.db.models import ADMIN, EDIT, USER
from contract_assistant.db.stores import AnalysisStore
from contract_assistant.extract.extractor import Extractor
from contract_assistant.review import load_rules
from contract_assistant.search import PassageSearch
from contract_assistant.services import Services
from contract_assistant.store import Store
from contract_assistant.web.app import create_app
from tests.support.fakes import EchoTextClient, FakeEncoder, FakeRouter, slow_fake_qa

DEV_DIR = Path(os.environ.get("DEV_DIR") or Path(tempfile.gettempdir()) / "contract-seeded")
PASSWORD = "correct horse battery"
SHORT = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
MEDIUM = "sha256:3d51fb71a4d06f8185ebfdb7c3feae4fa2935d9fc75951cdfc4aaff4b5733b07"
LONG = "sha256:5ace7a03650252f3f2a08d44b3bf4c689d6b9e1bc62a90a455a23bcb4e484b02"


def _services(settings) -> Services:
    clause_types = app_clause_types(settings)
    extra = load_extra_clause_types(settings.extra_clause_types_path)
    return Services(
        Store(settings.cache_dir, settings.demo_cache_dir),
        FakeRouter(),
        Extractor(slow_fake_qa(), clause_types, settings, device="fake"),
        clause_types,
        settings,
        PassageSearch(FakeEncoder()),
        frozenset(clause_type.name for clause_type in extra),
        gen=EchoTextClient(),
        rules=load_rules(known_types=[c.name for c in clause_types]),
    )


def _user(session, email: str, name: str, role: str) -> Actor:
    user, _ = accounts.create_user(session, None, email, name, role, PASSWORD)
    return accounts.actor_of(user)


def seed(database, settings) -> None:
    """Fill an empty database with the accounts, folders, contracts and share above."""
    with database.session() as session:
        if accounts.count_users(session):
            return
        _user(session, "ana@example.com", "Ana Admin", ADMIN)
        marko = _user(session, "marko@example.com", "Marko Markovic", USER)
        _user(session, "jelena@example.com", "Jelena Jovanovic", USER)
    analyses = AnalysisStore(database, settings, settings.demo_cache_dir)
    names = {doc_id: analyses.load(doc_id).source.filename for doc_id in (SHORT, MEDIUM, LONG)}
    with database.session() as session:
        supply = files.create_folder(session, marko, None, "Supply deals")
        sponsorship = files.create_folder(session, marko, None, "Sponsorship")
        for folder, doc_id in ((supply.id, SHORT), (sponsorship.id, MEDIUM), (None, LONG)):
            files.add_contract(session, marko, folder, names[doc_id], doc_id, "txt")
        files.share_folder(session, marko, supply.id, "jelena@example.com", EDIT)


settings = dataclasses.replace(load_settings(), cache_dir=DEV_DIR / "cache")
DEV_DIR.mkdir(parents=True, exist_ok=True)
database = connect(os.environ.get("DEV_DATABASE_URL") or f"sqlite:///{DEV_DIR / 'dev.db'}")
seed(database, settings)
app = create_app(_services(settings), database)
