"""Fixtures of the web app tests: fake services over the demo caches, a test client."""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import Settings, load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.search import PassageSearch
from contract_assistant.services import Services
from contract_assistant.store import Store
from contract_assistant.web.app import create_app
from tests.support.fakes import FakeEncoder, FakeRouter

SHORT_DOC_ID = "sha256:6d43065ff242642866da6be6b83f9849ad898e6eea9c1e71b0c2853dd3822c5a"
LONG_DOC_ID = "sha256:5ace7a03650252f3f2a08d44b3bf4c689d6b9e1bc62a90a455a23bcb4e484b02"
MEDIUM_DOC_ID = "sha256:3d51fb71a4d06f8185ebfdb7c3feae4fa2935d9fc75951cdfc4aaff4b5733b07"


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
def make_client(make_services):
    """Build a test client over fake services; keyword arguments go to make_services."""

    def make(**kwargs) -> TestClient:
        return TestClient(create_app(make_services(**kwargs)))

    return make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def frames(client: TestClient, method: str, url: str, **kwargs) -> list[dict]:
    """Send a request to a streaming route and return its frames in order."""
    with client.stream(method, url, **kwargs) as response:
        assert response.status_code == 200, response.read()
        return [json.loads(line) for line in response.iter_lines() if line.strip()]
