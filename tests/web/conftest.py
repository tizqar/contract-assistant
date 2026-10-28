"""Fixtures of the web app tests: fake services, a database with the demo contracts, a client.

Each test gets its own SQLite database. The client is signed in as a user who owns the three
demo contracts, under contract ids equal to their analyses' doc ids, so the tests can name
them by SHORT_DOC_ID and the others.
"""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from contract_assistant import accounts, files
from contract_assistant.access import Actor
from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import Settings, load_settings
from contract_assistant.db.models import Contract, User
from contract_assistant.db.stores import AnalysisStore
from contract_assistant.extract.extractor import Extractor
from contract_assistant.search import PassageSearch
from contract_assistant.services import Services
from contract_assistant.store import Store
from contract_assistant.web.app import create_app
from contract_assistant.web.auth import COOKIE
from tests.support.db import add_admin, add_user
from tests.support.fakes import FakeEncoder, FakeRouter

SHORT_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
LONG_DOC_ID = "sha256:5ace7a03650252f3f2a08d44b3bf4c689d6b9e1bc62a90a455a23bcb4e484b02"
MEDIUM_DOC_ID = "sha256:3d51fb71a4d06f8185ebfdb7c3feae4fa2935d9fc75951cdfc4aaff4b5733b07"
DEMO_DOC_IDS = (LONG_DOC_ID, MEDIUM_DOC_ID, SHORT_DOC_ID)
READER_EMAIL = "reader@example.com"
ADMIN_EMAIL = "admin@example.com"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(load_settings(), cache_dir=tmp_path / "cache")


@pytest.fixture
def clause_types(settings):
    return app_clause_types(settings)


@pytest.fixture
def unvalidated(settings) -> frozenset[str]:
    extra = load_extra_clause_types(settings.extra_clause_types_path)
    return frozenset(clause_type.name for clause_type in extra)


@pytest.fixture
def make_services(settings, clause_types, unvalidated):
    def make(
        extractor: Extractor | None = None, router=None, gen=None, rules=(), **extra
    ) -> Services:
        store = Store(settings.cache_dir, settings.demo_cache_dir)
        return Services(
            store,
            router or FakeRouter(),
            extractor,
            clause_types,
            settings,
            PassageSearch(FakeEncoder()),
            unvalidated,
            gen=gen,
            rules=rules,
            **extra,
        )

    return make


@pytest.fixture
def reader(database) -> Actor:
    """The user the client signs in as: an ordinary user."""
    return add_user(database, READER_EMAIL, name="Reader", user_id="reader")


@pytest.fixture
def admin(database) -> Actor:
    """An administrator, signed in only by the tests that ask for it."""
    return add_admin(database, ADMIN_EMAIL, name="Admin", user_id="admin")


def seed_demo(database, settings: Settings, owner: Actor) -> None:
    """Give owner the three demo contracts, at the top of their space, unless they have them."""
    analyses = AnalysisStore(database, settings, settings.demo_cache_dir)
    for doc_id in DEMO_DOC_IDS:
        cache = analyses.load(doc_id)
        with database.session() as session:
            if session.get(Contract, doc_id) is not None:
                continue
            files.add_contract(
                session,
                owner,
                None,
                cache.source.filename,
                doc_id,
                cache.source.kind,
                contract_id=doc_id,
            )


class SignedInWorkspace:
    """The app's workspace as a test's signed-in user sees it, made fresh on every use.

    A request's workspace remembers what it read; a test that reads, posts and reads again
    must not see the first read, so each attribute comes from a new one.
    """

    def __init__(self, app, actor: Actor):
        self.app = app
        self.actor = actor

    def __getattr__(self, name: str):
        return getattr(self.app.state.workspace.for_user(self.actor), name)


def sign_in(client: TestClient, database, actor: Actor) -> TestClient:
    """Start a session for actor, give its cookie to the client, and its workspace as `ws`."""
    with database.session() as session:
        token = accounts.start_session(session, session.get(User, actor.id), days=1)
    client.cookies.set(COOKIE, token)
    client.ws = SignedInWorkspace(client.app, actor)
    return client


@pytest.fixture
def make_client(make_services, database, settings, reader):
    """Build a test client over fake services, signed in as the reader.

    seed=False leaves the reader without contracts; as_actor signs in someone else; the other
    keyword arguments go to make_services.
    """

    def make(seed: bool = True, as_actor: Actor | None = None, **kwargs) -> TestClient:
        if seed:
            seed_demo(database, settings, reader)
        client = TestClient(create_app(make_services(**kwargs), database))
        return sign_in(client, database, as_actor or reader)

    return make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def frames(client: TestClient, method: str, url: str, **kwargs) -> list[dict]:
    """Send a request to a streaming route and return its frames in order."""
    with client.stream(method, url, **kwargs) as response:
        assert response.status_code == 200, response.read()
        return [json.loads(line) for line in response.iter_lines() if line.strip()]
