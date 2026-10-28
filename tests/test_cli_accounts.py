"""The database, account and import commands of the command line, on SQLite and Postgres."""

import io
import shutil

import pytest
from sqlalchemy import select

from contract_assistant import accounts, cli
from contract_assistant.db.engine import Database
from contract_assistant.db.models import ADMIN, USER, Contract, User
from tests.support.db import add_admin, add_user

# Nothing listens on the discard port, so a connection there fails at once.
UNREACHABLE = "postgresql+psycopg://nobody:nothing@127.0.0.1:9/nowhere"
PASSWORD = "correct horse battery"
SET_ROLE = ("user", "set-role", "--email")


@pytest.fixture
def url(monkeypatch, tmp_path) -> str:
    """The SQLite database the commands use, with every table."""
    url = f"sqlite:///{tmp_path / 'app.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    return url


@pytest.fixture
def database(url):
    database = Database(url)
    database.create_all()
    yield database
    database.dispose()


def user(database, email: str) -> User:
    with database.session() as session:
        return accounts.find_by_email(session, email)


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# db


def test_db_upgrade_makes_the_tables_of_a_sqlite_database(url, capsys):
    code, out, _ = run(capsys, "db", "upgrade")
    assert code == 0
    assert "SQLite takes its tables from the models" in out
    with Database(url).session() as session:
        assert session.scalars(select(User)).all() == []


def test_db_status_says_where_and_counts_accounts_contracts_and_events(
    database, capsys
):
    add_admin(database)
    run(capsys, "user", "create", "--email", "ana@example.com", "--name", "Ana")
    code, out, _ = run(capsys, "db", "status")
    assert code == 0
    assert f"database: {database.where}" in out
    assert "newest migration 0001" in out
    assert "accounts: 2 (2 active, 1 of them admins)" in out
    assert "contracts: 0" in out
    assert "events: 1" in out


def test_db_status_of_a_database_without_tables_says_to_upgrade(url, capsys):
    code, out, _ = run(capsys, "db", "status")
    assert code == 0
    assert cli.NO_TABLES in out


@pytest.mark.pg
def test_db_upgrade_and_status_on_postgres(pg_database, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", pg_database.engine.url.render_as_string(False))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    code, out, _ = run(capsys, "db", "upgrade")
    assert code == 0
    assert "at the newest revision, 0001" in out
    code, out, _ = run(capsys, "db", "status")
    assert "revision: 0001; newest 0001" in out
    assert "***" in out and "accounts: 0" in out
    assert cli.BEHIND not in out


# user


def test_user_create_prints_a_temporary_password_to_change(database, capsys):
    add_admin(database)
    code, out, _ = run(
        capsys, "user", "create", "--email", "Ana@Example.com", "--name", "Ana"
    )
    assert code == 0
    assert "created the account of Ana <ana@example.com> (user)" in out
    password = out.split("temporary password: ")[1].split()[0]
    created = user(database, "ana@example.com")
    assert accounts.verify_password(password, created.password_hash)
    assert created.must_change_password and created.role == USER
    assert cli.MUST_CHANGE in out


def test_user_create_reads_the_password_from_stdin(database, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))
    code, out, _ = run(
        capsys,
        "user", "create", "--email", "ana@example.com", "--name", "Ana", "--role", "admin",
        "--password-stdin",
    )
    assert code == 0
    created = user(database, "ana@example.com")
    assert accounts.verify_password(PASSWORD, created.password_hash)
    assert created.role == ADMIN and not created.must_change_password
    assert "temporary password" not in out


def test_the_first_account_made_on_the_command_line_can_be_the_admin(database, capsys):
    _, out, _ = run(capsys, "user", "create", "--email", "ana@example.com", "--name", "Ana")
    assert cli.NO_ADMIN in out
    _, out, _ = run(
        capsys, "user", "create", "--email", "boss@example.com", "--name", "Boss",
        "--role", "admin",
    )
    assert cli.NO_ADMIN not in out


def test_user_create_with_a_taken_email_prints_the_refusal(database, capsys):
    add_user(database, "ana@example.com")
    code, out, err = run(capsys, "user", "create", "--email", "ana@example.com", "--name", "A")
    assert code == cli.EXIT_ERROR
    assert err.strip() == f"error: {accounts.EMAIL_TAKEN}"
    assert out == ""


def test_user_list_shows_every_account_with_its_state(database, capsys):
    add_admin(database, name="Boss")
    add_user(database, "ana@example.com", name="Ana")
    run(capsys, "user", "deactivate", "--email", "ana@example.com")
    code, out, _ = run(capsys, "user", "list")
    lines = out.splitlines()
    assert code == 0
    assert lines[0].split() == ["email", "name", "role", "state", "last", "sign-in"]
    assert lines[1].split()[:4] == ["ana@example.com", "Ana", "user", "deactivated"]
    assert lines[2].split()[:4] == ["admin@example.com", "Boss", "admin", "active"]


def test_user_list_without_accounts(database, capsys):
    assert run(capsys, "user", "list")[1].strip() == cli.NO_ACCOUNTS


def test_reset_password_lets_the_only_admin_back_in(database, capsys):
    add_admin(database)
    code, out, _ = run(capsys, "user", "reset-password", "--email", "admin@example.com")
    password = out.split("temporary password: ")[1].split()[0]
    admin = user(database, "admin@example.com")
    assert code == 0
    assert accounts.verify_password(password, admin.password_hash)
    assert admin.must_change_password


def test_set_role_and_the_last_admin_stays_an_admin(database, capsys):
    add_admin(database)
    add_user(database, "ana@example.com")
    code, _, err = run(capsys, *SET_ROLE, "admin@example.com", "--role", USER)
    assert code == cli.EXIT_ERROR and accounts.LAST_ADMIN in err
    code, out, _ = run(capsys, *SET_ROLE, "ana@example.com", "--role", ADMIN)
    assert code == 0 and out.strip() == "ana@example.com is now an admin"
    assert user(database, "ana@example.com").role == ADMIN


def test_deactivate_and_activate(database, capsys):
    add_admin(database)
    add_user(database, "ana@example.com")
    assert run(capsys, "user", "deactivate", "--email", "ana@example.com")[0] == 0
    assert not user(database, "ana@example.com").active
    assert run(capsys, "user", "activate", "--email", "ana@example.com")[0] == 0
    assert user(database, "ana@example.com").active


def test_an_unknown_email_prints_the_refusal(database, capsys):
    code, _, err = run(capsys, "user", "deactivate", "--email", "nobody@example.com")
    assert code == cli.EXIT_ERROR
    assert accounts.NO_SUCH_USER in err


# import-cache


def test_import_cache_prints_the_report(database, tmp_path, capsys):
    add_user(database, "ana@example.com")
    cache = tmp_path / "old-cache"
    cache.mkdir()
    shutil.copy(cli.ROOT / "demo" / "cache" / "short_supply_agreement.json", cache)
    (cache / "broken.json").write_text("{")
    argv = ("import-cache", "--owner", "ana@example.com", "--cache-dir", str(cache))
    code, out, _ = run(capsys, *argv, "--folder", "Old")
    assert code == 0
    assert 'folder: "Old" (new)' in out
    assert "contracts: 1 imported, 0 already there, 1 skipped" in out
    assert "  - broken.json: not a valid cache" in out
    code, out, _ = run(capsys, *argv, "--folder", "Old")
    assert "contracts: 0 imported, 1 already there, 1 skipped" in out
    with database.session() as session:
        assert len(session.scalars(select(Contract)).all()) == 1


def test_import_cache_for_an_unknown_owner_fails(database, tmp_path, capsys):
    argv = ("import-cache", "--owner", "nobody@example.com", "--cache-dir", str(tmp_path))
    code, _, err = run(capsys, *argv)
    assert code == cli.EXIT_ERROR and accounts.NO_SUCH_USER in err


def test_import_cache_without_the_folder_fails(database, tmp_path, capsys):
    add_user(database, "ana@example.com")
    argv = ("import-cache", "--owner", "ana@example.com", "--cache-dir", str(tmp_path / "no"))
    code, _, err = run(capsys, *argv)
    assert code == cli.EXIT_ERROR and "no cache folder at" in err


# An unreachable database


@pytest.mark.parametrize(
    "argv",
    [
        ("db", "upgrade"),
        ("db", "status"),
        ("user", "list"),
        ("user", "create", "--email", "ana@example.com", "--name", "Ana"),
        ("import-cache", "--owner", "ana@example.com"),
    ],
)
def test_an_unreachable_database_says_how_to_start_it(monkeypatch, capsys, argv):
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    code, _, err = run(capsys, *argv)
    assert code == cli.EXIT_ERROR
    assert "make db-up" in err and "nowhere" in err
