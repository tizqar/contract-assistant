import json
from pathlib import Path

import pytest

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
