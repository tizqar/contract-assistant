"""The stores of the web app in the database: analyses, files, generated texts, feedback, chats.

Every method opens its own session, so the stores can be called from the app's worker threads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import pickle
import re
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ValidationError
from sqlalchemy import delete, func, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from contract_assistant.config import Settings
from contract_assistant.db.engine import Database
from contract_assistant.db.models import Analysis, Blob, ChatMessage, Generated, utcnow
from contract_assistant.db.models import Feedback as FeedbackRow
from contract_assistant.feedback import Feedback
from contract_assistant.store import CacheIntegrityError, DocumentCache, check_cache, read_cache

logger = logging.getLogger(__name__)

POSTGRESQL = "postgresql"
# JSONB keeps the keys of an object in an order of its own, so the clause types' order is kept.
CLAUSE_ORDER = "clause_order"
# JSONB refuses the NUL character, which a PDF's text can hold; such an analysis is kept as
# one JSON string under this key instead, where the character is written as an escape.
ENCODED = "encoded_json"
NUL = "\x00"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
SUFFIX = re.compile(r"(\.[A-Za-z0-9]{1,16})?")
PICKLE_PROTOCOL = 5
PACKAGE = "contract_assistant"
# What a chat message may name besides the data classes of the package: plain values whose
# constructors build the value and nothing else. An aware datetime needs timezone, which needs
# timedelta. The chat's own classes need none of these today.
SAFE_GLOBALS = frozenset(
    {
        ("datetime", "date"),
        ("datetime", "datetime"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
    }
)


def _insert(session: Session, model):
    """Return an INSERT in the session's dialect, which can say what to do on a conflict."""
    dialect = postgresql if session.get_bind().dialect.name == POSTGRESQL else sqlite
    return dialect.insert(model)


class AnalysisStore:
    """The analyses of the texts, read from the database first and then from the demo caches.

    A demo cache is copied into the database the first time it is loaded. The analyses loaded
    last are kept in memory, as a page makes several requests that each load one; treat them
    as read-only and change copies (`model_copy`).
    """

    def __init__(
        self,
        database: Database,
        settings: Settings,
        demo_dir: Path | None = None,
        memory: int = 8,
    ):
        self.database = database
        self.settings = settings
        self.demo_dir = demo_dir
        self.memory = memory
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, DocumentCache] = OrderedDict()
        # Counts the saves, so that a load which overlapped a save does not keep what it read.
        self._saves = 0
        self._demo: dict[str, Path] | None = None

    def save(self, cache: DocumentCache) -> None:
        """Keep the analysis of a text, replacing the one saved before."""
        now = utcnow()
        with self.database.session() as session:
            row = {**_analysis_row(cache), "created_at": now, "updated_at": now}
            statement = _insert(session, Analysis).values(**row)
            statement = statement.on_conflict_do_update(
                index_elements=[Analysis.doc_id],
                set_={
                    "data": statement.excluded.data,
                    "extractor": statement.excluded.extractor,
                    "updated_at": now,
                },
            )
            session.execute(statement)
        with self._lock:
            self._saves += 1
            self._recent.pop(cache.doc_id, None)

    def load(self, doc_id: str) -> DocumentCache | None:
        """Return the analysis of doc_id, or None; raise as `store.read_cache` does."""
        with self._lock:
            cache = self._recent.get(doc_id)
            if cache is not None:
                self._recent.move_to_end(doc_id)
                return cache
            saves = self._saves
        cache = self._read(doc_id)
        if cache is None:
            cache = self._copy_demo(doc_id)
        if cache is not None:
            self._remember(cache, saves)
        return cache

    def exists(self, doc_id: str) -> bool:
        """Tell whether load finds an analysis of doc_id, in the database or the demo caches."""
        with self._lock:
            if doc_id in self._recent:
                return True
        return self._in_database(doc_id) or doc_id in self._demo_paths()

    def ensure(self, doc_id: str) -> bool:
        """Copy the demo analysis of doc_id into the database if needed; tell if it is there."""
        return self._in_database(doc_id) or self._copy_demo(doc_id) is not None

    def _in_database(self, doc_id: str) -> bool:
        with self.database.session() as session:
            found = session.scalar(select(Analysis.doc_id).where(Analysis.doc_id == doc_id))
        return found is not None

    def _read(self, doc_id: str) -> DocumentCache | None:
        with self.database.session() as session:
            data = session.scalar(select(Analysis.data).where(Analysis.doc_id == doc_id))
        if data is None:
            return None
        order = data.pop(CLAUSE_ORDER, None)
        try:
            if ENCODED in data:
                cache = DocumentCache.model_validate_json(data[ENCODED])
            else:
                cache = DocumentCache.model_validate(data)
        except ValidationError as error:
            raise CacheIntegrityError(f"not a valid analysis: {doc_id}") from error
        if order is not None:
            clauses = {name: cache.clauses[name] for name in order if name in cache.clauses}
            cache.clauses = {**clauses, **cache.clauses}
        return self._checked(cache, doc_id)

    def _copy_demo(self, doc_id: str) -> DocumentCache | None:
        path = self._demo_paths().get(doc_id)
        if path is None:
            return None
        cache = self._checked(read_cache(path, self.settings), doc_id)
        with self.database.session() as session:
            statement = _insert(session, Analysis).values(**_analysis_row(cache))
            session.execute(statement.on_conflict_do_nothing(index_elements=[Analysis.doc_id]))
        return cache

    def _checked(self, cache: DocumentCache, doc_id: str) -> DocumentCache:
        check_cache(cache, self.settings)
        if cache.doc_id != doc_id:
            raise CacheIntegrityError(f"the analysis of {doc_id} belongs to {cache.doc_id}")
        return cache

    def _remember(self, cache: DocumentCache, saves: int) -> None:
        with self._lock:
            if self.memory <= 0 or saves != self._saves:
                return
            self._recent[cache.doc_id] = cache
            self._recent.move_to_end(cache.doc_id)
            while len(self._recent) > self.memory:
                self._recent.popitem(last=False)

    def _demo_paths(self) -> dict[str, Path]:
        with self._lock:
            if self._demo is not None:
                return self._demo
        paths = _demo_index(self.demo_dir) if self.demo_dir is not None else {}
        with self._lock:
            self._demo = paths
        return paths


def _analysis_row(cache: DocumentCache) -> dict:
    if NUL in cache.text or NUL in cache.source.filename:
        data = {ENCODED: cache.model_dump_json()}
    else:
        data = cache.model_dump(mode="json")
        data[CLAUSE_ORDER] = list(cache.clauses)
    return {"doc_id": cache.doc_id, "data": data, "extractor": cache.model.name}


def _demo_index(directory: Path) -> dict[str, Path]:
    """Return the demo cache file of each doc id, the first one when two have the same."""
    paths: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            cache = DocumentCache.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            logger.warning("skipping invalid demo cache file: %s", path)
            continue
        paths.setdefault(cache.doc_id, path)
    return paths


class BlobStore:
    """The bytes of uploaded files, kept once per content, with local copies for the readers."""

    def __init__(self, database: Database, local_dir: Path):
        self.database = database
        self.local_dir = local_dir

    def put(self, data: bytes, media_type: str) -> str:
        """Keep the bytes unless they are kept already; return their sha256 in hex."""
        sha256 = hashlib.sha256(data).hexdigest()
        values = {"sha256": sha256, "media_type": media_type, "size": len(data), "data": data}
        with self.database.session() as session:
            statement = _insert(session, Blob).values(**values, created_at=utcnow())
            session.execute(statement.on_conflict_do_nothing(index_elements=[Blob.sha256]))
        return sha256

    def get(self, sha256: str) -> bytes | None:
        """Return the bytes of a sha256, or None."""
        if not SHA256_HEX.fullmatch(sha256):
            return None
        with self.database.session() as session:
            data = session.scalar(select(Blob.data).where(Blob.sha256 == sha256))
        return None if data is None else bytes(data)

    def local_path(self, sha256: str, suffix: str = "") -> Path | None:
        """Return a local file with the bytes of a sha256, written once; None when unknown."""
        if not SUFFIX.fullmatch(suffix):
            raise ValueError(f"not a file suffix: {suffix!r}")
        if not SHA256_HEX.fullmatch(sha256):
            return None
        path = self.local_dir / f"{sha256}{suffix}"
        if path.is_file():
            return path
        data = self.get(sha256)
        if data is None:
            return None
        _write_bytes(path, data)
        return path


def _write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class DbRecords:
    """The records of the generated stores in the `generated` table, each as JSON in `data`."""

    def __init__(self, database: Database):
        self.database = database

    def read(self, kind: str, key: str) -> str | None:
        """Return the record saved under kind and key, or None."""
        with self.database.session() as session:
            data = session.scalar(
                select(Generated.data).where(Generated.kind == kind, Generated.key == key)
            )
        return None if data is None else json.dumps(data)

    def write(self, kind: str, key: str, doc_id: str, payload: str) -> None:
        """Save a record of doc_id under kind and key, replacing an earlier one."""
        values = {"kind": kind, "key": key, "doc_id": doc_id, "data": json.loads(payload)}
        with self.database.session() as session:
            statement = _insert(session, Generated).values(**values, created_at=utcnow())
            statement = statement.on_conflict_do_update(
                index_elements=[Generated.kind, Generated.key],
                set_={
                    "doc_id": statement.excluded.doc_id,
                    "data": statement.excluded.data,
                    "created_at": statement.excluded.created_at,
                },
            )
            session.execute(statement)

    def delete_doc(self, doc_id: str) -> int:
        """Delete every record of doc_id, of every kind; return how many there were."""
        with self.database.session() as session:
            result = session.execute(delete(Generated).where(Generated.doc_id == doc_id))
        return result.rowcount


class FeedbackLog:
    """The thumbs up and down on chat answers, with the contract and the user they came from."""

    def __init__(self, database: Database):
        self.database = database

    def record(self, contract_id: str | None, user_id: str | None, verdict: Feedback) -> None:
        """Keep one verdict."""
        row = FeedbackRow(
            contract_id=contract_id,
            user_id=user_id,
            question=verdict.question,
            route_kind=verdict.route_kind,
            clause_types=list(verdict.clause_types),
            liked=verdict.liked,
            comment=verdict.comment,
            at=_parse_time(verdict.at),
        )
        with self.database.session() as session:
            session.add(row)

    def read(self, contract_id: str | None = None) -> list[Feedback]:
        """Return the verdicts on one contract, or on all, oldest first.

        The doc id of a verdict is its contract id, empty once the contract is purged.
        """
        query = select(FeedbackRow).order_by(FeedbackRow.at, FeedbackRow.id)
        if contract_id is not None:
            query = query.where(FeedbackRow.contract_id == contract_id)
        with self.database.session() as session:
            rows = session.scalars(query).all()
        return [
            Feedback(
                doc_id=row.contract_id or "",
                question=row.question,
                route_kind=row.route_kind,
                clause_types=list(row.clause_types),
                liked=row.liked,
                comment=row.comment,
                at=row.at.isoformat(),
            )
            for row in rows
        ]


def _parse_time(text: str) -> datetime:
    """Read an ISO time; one without a time zone is taken as UTC."""
    at = datetime.fromisoformat(text)
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


class ChatLog:
    """Each user's chat about each contract: the chat's message objects, kept in order.

    A message is kept as a pickle and read back by `load_message`, which builds only the data
    classes of this package and a few plain values. A message that no longer loads, because
    the code changed or the payload names something else, is skipped with a warning.
    """

    def __init__(self, database: Database):
        self.database = database

    def load(self, contract_id: str, user_id: str) -> list[tuple[str, object]]:
        """Return the chat's messages, oldest first, as (role, message) pairs."""
        with self.database.session() as session:
            rows = self._rows(session, contract_id, user_id)
        return [(row.role, message) for row, message in _loaded(rows)]

    def append(self, contract_id: str, user_id: str, role: str, message: object) -> None:
        """Add a message at the end of the chat."""
        payload = dump_message(message)
        with self.database.session() as session:
            last = session.scalar(
                select(func.max(ChatMessage.position)).where(
                    ChatMessage.contract_id == contract_id, ChatMessage.user_id == user_id
                )
            )
            position = 0 if last is None else last + 1
            session.add(
                ChatMessage(
                    contract_id=contract_id,
                    user_id=user_id,
                    position=position,
                    role=role,
                    payload=payload,
                )
            )

    def replace(self, contract_id: str, user_id: str, index: int, message: object) -> bool:
        """Put message in place of the index-th message load returns; tell if there is one."""
        payload = dump_message(message)
        with self.database.session() as session:
            loaded = list(_loaded(self._rows(session, contract_id, user_id)))
            if not 0 <= index < len(loaded):
                return False
            loaded[index][0].payload = payload
        return True

    def clear(self, contract_id: str, user_id: str) -> int:
        """Delete the chat; return how many messages it had."""
        with self.database.session() as session:
            result = session.execute(
                delete(ChatMessage).where(
                    ChatMessage.contract_id == contract_id, ChatMessage.user_id == user_id
                )
            )
        return result.rowcount

    @staticmethod
    def _rows(session: Session, contract_id: str, user_id: str) -> list[ChatMessage]:
        query = (
            select(ChatMessage)
            .where(ChatMessage.contract_id == contract_id, ChatMessage.user_id == user_id)
            .order_by(ChatMessage.position, ChatMessage.id)
        )
        return list(session.scalars(query))


def _loaded(rows: list[ChatMessage]) -> Iterator[tuple[ChatMessage, object]]:
    """Yield each row with its message, skipping with a warning a payload that does not load."""
    for row in rows:
        try:
            message = load_message(bytes(row.payload))
        except Exception as error:  # noqa: BLE001 - any payload that fails to load is skipped
            logger.warning(
                "skipping chat message %s of contract %s: %s: %s",
                row.id,
                row.contract_id,
                error.__class__.__name__,
                error,
            )
            continue
        yield row, message


class RefusedClassError(pickle.UnpicklingError):
    """Raised when a chat payload names something the chat log does not build."""


class _MessageUnpickler(pickle.Unpickler):
    """An unpickler that builds only the package's data classes and the safe plain values."""

    def find_class(self, module: str, name: str):
        if (module, name) in SAFE_GLOBALS:
            return super().find_class(module, name)
        if not _in_package(module) or "." in name:
            raise RefusedClassError(f"{module}.{name} is not allowed in a chat message")
        # A payload never makes the app import code: the module must be loaded already.
        loaded = sys.modules.get(module)
        found = getattr(loaded, name, None) if loaded is not None else None
        if found is None:
            raise RefusedClassError(f"{module}.{name} no longer exists")
        if not _is_data_class(found) or found.__module__ != module:
            raise RefusedClassError(f"{module}.{name} is not allowed in a chat message")
        return found


def _in_package(module: str) -> bool:
    return module == PACKAGE or module.startswith(PACKAGE + ".")


def _is_data_class(found: object) -> bool:
    """Tell whether found is a class that only holds data: a dataclass, a model or an enum."""
    if not isinstance(found, type):
        return False
    return dataclasses.is_dataclass(found) or issubclass(found, BaseModel | Enum)


def load_message(payload: bytes) -> object:
    """Build a chat message from its payload; raise when it names something not allowed."""
    return _MessageUnpickler(io.BytesIO(payload)).load()


def dump_message(message: object) -> bytes:
    """Return the payload of a chat message; refuse one that load_message could not build."""
    payload = pickle.dumps(message, protocol=PICKLE_PROTOCOL)
    load_message(payload)
    return payload
