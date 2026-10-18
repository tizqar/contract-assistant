"""Development app: demo caches, fake router and a fake slow extractor."""

import dataclasses
import tempfile
from pathlib import Path

from contract_assistant.clause_types import app_clause_types, load_extra_clause_types
from contract_assistant.config import load_settings
from contract_assistant.extract.extractor import Extractor
from contract_assistant.search import PassageSearch
from contract_assistant.store import Store
from contract_assistant.ui.app import Services, build_app, launch_app
from tests.support.fakes import FakeEncoder, FakeRouter, slow_fake_qa

DEV_CACHE_DIR = Path(tempfile.gettempdir()) / "contract-assistant-dev"


def dev_services() -> Services:
    """Return services backed by fakes, with uploads kept in a fixed temporary folder."""
    settings = dataclasses.replace(load_settings(), cache_dir=DEV_CACHE_DIR)
    clause_types = app_clause_types(settings)
    extra = load_extra_clause_types(settings.extra_clause_types_path)
    extractor = Extractor(slow_fake_qa(), clause_types, settings, device="fake")
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    unvalidated = frozenset(clause_type.name for clause_type in extra)
    search = PassageSearch(FakeEncoder())
    return Services(store, FakeRouter(), extractor, clause_types, settings, search, unvalidated)


services = dev_services()
demo = build_app(services)

if __name__ == "__main__":
    launch_app(demo, services)
