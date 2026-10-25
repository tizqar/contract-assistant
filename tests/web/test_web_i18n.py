"""The two interface languages: the catalogue covers every string, the cookie picks it."""

import importlib
import pkgutil
import re
from pathlib import Path

from contract_assistant import access, audit, web
from contract_assistant.web.i18n import SERBIAN, copy_for, copy_strings, missing, translate
from tests.web.conftest import SHORT_DOC_ID

TEMPLATES = Path("contract_assistant/web/templates")
# _("...") and _('...') in templates; the string has no quote of its own kind inside.
CALLS = re.compile(r"""_\(\s*(?:"([^"]+)"|'([^']+)')\s*\)""")


def template_strings() -> list[str]:
    found = []
    for path in TEMPLATES.rglob("*.html"):
        for double, single in CALLS.findall(path.read_text()):
            found.append(double or single)
    return found


def registering_modules() -> list:
    """Every web module, and the policy modules whose messages the pages show."""
    names = [info.name for info in pkgutil.iter_modules(web.__path__) if not info.ispkg]
    found = [importlib.import_module(f"{web.__name__}.{name}") for name in names]
    return [*found, access, audit]


def module_strings() -> list[str]:
    """The user-facing constants each area registers for translation."""
    found = []
    for module in registering_modules():
        found.extend(getattr(module, "TRANSLATED", ()))
    return found


def test_the_areas_that_register_strings_are_all_checked():
    names = {module.__name__.rsplit(".", 1)[-1] for module in registering_modules()}
    assert {"chat", "insights", "library", "access", "audit"} <= names


def test_every_word_of_the_vocabulary_has_a_serbian_version():
    assert missing(SERBIAN, copy_strings()) == []


def test_every_translated_template_string_has_a_serbian_version():
    assert missing(SERBIAN, template_strings()) == []


def test_every_registered_module_string_has_a_serbian_version():
    assert missing(SERBIAN, module_strings()) == []


def test_translate_falls_back_to_the_english_text():
    assert translate("Found", "sr") == "Pronađeno"
    assert translate("Found", "en") == "Found"
    assert translate("no such text", "sr") == "no such text"


def test_the_vocabulary_is_translated_as_a_whole():
    serbian = copy_for("sr")
    assert serbian.STATUS["found"] == "Pronađeno"
    assert serbian.SECTIONS[0][1] == "Sažetak"
    assert copy_for("en").STATUS["found"] == "Found"


def test_the_cookie_switches_the_interface(client):
    english = client.get(f"/documents/{SHORT_DOC_ID}").text
    assert '<html lang="en">' in english and ">Overview<" in english
    client.cookies.set("lang", "sr")
    serbian = client.get(f"/documents/{SHORT_DOC_ID}").text
    assert '<html lang="sr">' in serbian and ">Pregled<" in serbian
    assert "Pitaj" in serbian


def test_the_language_form_sets_the_cookie_and_goes_back(client):
    response = client.post(
        "/language", data={"lang": "sr", "next": f"/documents/{SHORT_DOC_ID}?low=0.1"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/documents/{SHORT_DOC_ID}?low=0.1"
    assert response.cookies.get("lang") == "sr"
    elsewhere = client.post(
        "/language", data={"lang": "sr", "next": "https://example.com/x"}, follow_redirects=False
    )
    assert elsewhere.headers["location"] == "/"
    unknown = client.post("/language", data={"lang": "xx", "next": "/"}, follow_redirects=False)
    assert "lang" not in unknown.cookies


def test_each_serbian_entry_lives_in_one_catalogue():
    import importlib
    import pkgutil

    from contract_assistant.web import translations

    homes: dict[str, list[str]] = {}
    for module in pkgutil.iter_modules(translations.__path__):
        if module.name.startswith("sr_"):
            loaded = importlib.import_module(f"{translations.__name__}.{module.name}")
            for key in loaded.CATALOGUE:
                homes.setdefault(key, []).append(module.name)
    assert {key: where for key, where in homes.items() if len(where) > 1} == {}
