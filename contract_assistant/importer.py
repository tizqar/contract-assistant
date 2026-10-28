"""Bring the cache folder of the command line into the web app's database, for one account.

Before the web app had accounts it kept everything under the cache folder: an analysis per
text (`<sha>.json`), a copy of each PDF (`documents/`), the reader's decisions
(`decisions/`), the saved summaries, key terms, obligations and overviews, and the thumbs up
and down (`feedback.jsonl`). `import_cache` gives one account a contract for each valid
analysis there, in one folder at the top of their space, with all of that attached.

Running it again adds nothing that is there already: a contract of the same text in that
folder is reused, and only the saved texts and feedback it lacks are added to it. Its
decisions are brought in with a new contract only, so a decision undone in the app stays
undone. The cache folder is only read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select

from contract_assistant import files
from contract_assistant.access import AccessError, Actor
from contract_assistant.accounts import (
    NO_SUCH_USER,
    NOT_FOUND,
    AccountError,
    actor_of,
    find_by_email,
)
from contract_assistant.config import Settings
from contract_assistant.db.engine import Database
from contract_assistant.db.models import PDF, TXT, Contract
from contract_assistant.db.models import Decision as DecisionRow
from contract_assistant.db.stores import AnalysisStore, BlobStore, DbRecords, FeedbackLog
from contract_assistant.decisions import DECISIONS_DIR, Decision
from contract_assistant.explain import LANGUAGES
from contract_assistant.feedback import FEEDBACK_FILE, Feedback
from contract_assistant.fulltext import FULLTEXT_DIR, FullTextStore
from contract_assistant.generation import GeneratedRecord, GeneratedStore
from contract_assistant.obligations import OBLIGATIONS_DIR, ObligationStore
from contract_assistant.pdf_view import DOCUMENTS_DIR, copy_path
from contract_assistant.store import DOC_ID_PREFIX, DocumentCache, read_cache
from contract_assistant.summary import SUMMARY_DIR, SummaryStore
from contract_assistant.term_sheet import TERM_SHEET_DIR, TermSheetStore

DEFAULT_FOLDER = "Imported from the cache"
MEDIA_TYPES = {PDF: "application/pdf", TXT: "text/plain; charset=utf-8"}
# Where the web app writes local copies of the files; the import itself never writes there.
LOCAL_FILES_DIR = "files"
GENERATED_DIRS = (SUMMARY_DIR, TERM_SHEET_DIR, OBLIGATIONS_DIR, FULLTEXT_DIR)
DECISION_LIST = TypeAdapter(list[Decision])

NO_CACHE_FOLDER = "no cache folder at {path}"
SAME_TEXT = "the same text as {name}"
NO_ANALYSIS = "no usable analysis of its text in the cache folder"
NOT_DECISIONS = "not a valid list of decisions"
BAD_SPAN = "the {clause} decision is not about a passage of the text"
BAD_TIME = "its time cannot be read: {at!r}"
NOT_A_RECORD = "not a valid saved text"
OLD_RECORD = "saved by an older version of the app, which the app no longer reads"
NOT_FEEDBACK = "not a valid verdict"
UNREADABLE = "cannot be read: {error}"


@dataclass(frozen=True)
class Skipped:
    """A file, or a part of one, that was not imported, and why."""

    path: str
    reason: str


@dataclass
class ImportReport:
    """What an import brought in, found there already, and left out, with the reasons.

    `skipped` names the analyses that gave no contract; `left_out` the other files and lines.
    """

    cache_dir: Path
    owner: str
    folder: str
    folder_created: bool = False
    imported: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    pdfs: int = 0
    decisions: int = 0
    generated: Counter[str] = field(default_factory=Counter)
    feedback: int = 0
    left_out: list[Skipped] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Tell whether the import added anything to the database."""
        return bool(self.imported or sum(self.generated.values()) or self.feedback)


@dataclass(frozen=True)
class _Saved:
    """A saved text of the cache folder and the store of the database it belongs in."""

    path: str
    store: GeneratedStore
    record: GeneratedRecord


def import_cache(
    database: Database,
    settings: Settings,
    owner_email: str,
    folder_name: str = DEFAULT_FOLDER,
    cache_dir: Path | None = None,
) -> ImportReport:
    """Give an account a contract for each valid analysis of the cache folder; return a report.

    The analyses are read against the settings: one written with other settings, or damaged,
    is skipped and named in the report. The cache folder is never changed.
    """
    cache_dir = cache_dir or settings.cache_dir
    if not cache_dir.is_dir():
        raise FileNotFoundError(NO_CACHE_FOLDER.format(path=cache_dir))
    with database.session() as session:
        user = find_by_email(session, owner_email)
        if user is None:
            raise AccountError(NO_SUCH_USER, NOT_FOUND)
        owner = actor_of(user)
    report = ImportReport(cache_dir=cache_dir, owner=owner.email, folder=folder_name)
    run = _Import(database, settings, cache_dir, owner, report)
    caches = run.analyses()
    saved = run.saved_texts()
    verdicts = run.verdicts()
    run.leave_out_others(caches, saved, verdicts)
    for doc_id, (path, cache) in caches.items():
        contract_id = run.contract(path, cache)
        if contract_id is None:
            continue
        run.move_saved(contract_id, saved.get(doc_id, []))
        run.add_feedback(contract_id, verdicts.get(doc_id, []))
    return report


class _Import:
    """One run of the import: the stores it writes to and the folder it fills."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        cache_dir: Path,
        owner: Actor,
        report: ImportReport,
    ):
        self.database = database
        self.settings = settings
        self.cache_dir = cache_dir
        self.owner = owner
        self.report = report
        self.analyses_store = AnalysisStore(database, settings, memory=0)
        self.blobs = BlobStore(database, settings.cache_dir / LOCAL_FILES_DIR)
        records = DbRecords(database)
        self.stores: dict[str, list[GeneratedStore]] = {
            SUMMARY_DIR: [SummaryStore(records, language) for language in LANGUAGES],
            TERM_SHEET_DIR: [TermSheetStore(records)],
            OBLIGATIONS_DIR: [ObligationStore(records)],
            FULLTEXT_DIR: [FullTextStore(records)],
        }
        self.feedback_log = FeedbackLog(database)
        self.folder_id = self._find_folder()

    # Reading the cache folder

    def analyses(self) -> dict[str, tuple[Path, DocumentCache]]:
        """Return the valid analyses by doc id; skip the others with their reason."""
        found: dict[str, tuple[Path, DocumentCache]] = {}
        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                cache = read_cache(path, self.settings)
            except (ValueError, OSError) as error:
                self.report.skipped.append(Skipped(path.name, str(error)))
                continue
            if cache.doc_id in found:
                same = SAME_TEXT.format(name=found[cache.doc_id][0].name)
                self.report.skipped.append(Skipped(path.name, same))
                continue
            found[cache.doc_id] = (path, cache)
        return found

    def saved_texts(self) -> dict[str, list[_Saved]]:
        """Return the saved summaries, key terms, obligations and overviews by doc id."""
        found: dict[str, list[_Saved]] = {}
        for directory in GENERATED_DIRS:
            stores = self.stores[directory]
            for path in sorted((self.cache_dir / directory).glob("*.json")):
                name = f"{directory}/{path.name}"
                try:
                    record = stores[0].record_type.model_validate_json(path.read_bytes())
                except (ValidationError, OSError):
                    self.report.left_out.append(Skipped(name, NOT_A_RECORD))
                    continue
                store = next((store for store in stores if _keyed(store, record, path)), None)
                if store is None:
                    self.report.left_out.append(Skipped(name, OLD_RECORD))
                    continue
                found.setdefault(record.doc_id, []).append(_Saved(name, store, record))
        return found

    def verdicts(self) -> dict[str, list[tuple[str, Feedback]]]:
        """Return the lines of the feedback file by doc id, each with where it was."""
        path = self.cache_dir / FEEDBACK_FILE
        if not path.is_file():
            return {}
        found: dict[str, list[tuple[str, Feedback]]] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{FEEDBACK_FILE}, line {number}"
            try:
                verdict = Feedback.model_validate_json(line)
            except ValidationError:
                self.report.left_out.append(Skipped(where, NOT_FEEDBACK))
                continue
            found.setdefault(verdict.doc_id, []).append((where, verdict))
        return found

    def leave_out_others(
        self,
        caches: dict[str, tuple[Path, DocumentCache]],
        saved: dict[str, list[_Saved]],
        verdicts: dict[str, list[tuple[str, Feedback]]],
    ) -> None:
        """Name every decision, PDF, saved text and verdict of a text without an analysis."""
        others: list[str] = []
        for directory, suffix in ((DECISIONS_DIR, ".json"), (DOCUMENTS_DIR, ".pdf")):
            for path in sorted((self.cache_dir / directory).glob(f"*{suffix}")):
                if DOC_ID_PREFIX + path.stem not in caches:
                    others.append(f"{directory}/{path.name}")
        for doc_id, items in saved.items():
            if doc_id not in caches:
                others.extend(item.path for item in items)
        for doc_id, lines in verdicts.items():
            if doc_id not in caches:
                others.extend(where for where, _ in lines)
        self.report.left_out.extend(Skipped(path, NO_ANALYSIS) for path in others)

    # Writing to the database

    def contract(self, path: Path, cache: DocumentCache) -> str | None:
        """Return the owner's contract of the analysis in the folder, added when missing."""
        existing = self._existing(cache.doc_id)
        if existing is not None:
            self.report.reused.append(existing[1])
            return existing[0]
        decisions = self._decisions(cache)
        if not self.analyses_store.exists(cache.doc_id):
            self.analyses_store.save(cache)
        kind, sha256 = self._file(cache)
        name = cache.source.filename.strip() or f"{path.stem}.{kind}"
        folder_id = self._folder()
        try:
            with self.database.session() as session:
                view = files.add_contract(
                    session,
                    self.owner,
                    folder_id,
                    name,
                    cache.doc_id,
                    kind,
                    sha256,
                    system=True,
                    details={"file": path.name},
                )
                for decision, at in decisions:
                    session.add(
                        DecisionRow(
                            contract_id=view.id,
                            clause_type=decision.clause_type,
                            verdict=decision.verdict,
                            start=decision.start,
                            end=decision.end,
                            decided_by=None,
                            decided_at=at,
                        )
                    )
        except AccessError as error:
            self.report.skipped.append(Skipped(path.name, error.message))
            return None
        self.report.imported.append(view.name)
        self.report.pdfs += kind == PDF
        self.report.decisions += len(decisions)
        return view.id

    def move_saved(self, contract_id: str, saved: list[_Saved]) -> None:
        """Save the texts of an analysis under the contract id, where none is saved yet."""
        for item in saved:
            record = item.record
            keys = (record.model, record.tau_low, record.tau_high)
            key = item.store.key_of(contract_id, *keys)
            if item.store.records.read(item.store.kind, key) is not None:
                continue
            item.store.save(record.model_copy(update={"doc_id": contract_id}))
            self.report.generated[item.store.kind] += 1

    def add_feedback(self, contract_id: str, lines: list[tuple[str, Feedback]]) -> None:
        """Keep the verdicts on a contract that are not kept already."""
        if not lines:
            return
        kept = {_verdict_key(verdict) for verdict in self.feedback_log.read(contract_id)}
        for where, verdict in lines:
            try:
                key = _verdict_key(verdict)
            except ValueError:
                self.report.left_out.append(Skipped(where, BAD_TIME.format(at=verdict.at)))
                continue
            if key in kept:
                continue
            self.feedback_log.record(contract_id, None, verdict)
            kept.add(key)
            self.report.feedback += 1

    def _existing(self, doc_id: str) -> tuple[str, str] | None:
        """Return the id and name of the owner's contract of a text in the folder, or None."""
        if self.folder_id is None:
            return None
        query = (
            select(Contract.id, Contract.name)
            .where(
                Contract.owner_id == self.owner.id,
                Contract.folder_id == self.folder_id,
                Contract.doc_id == doc_id,
            )
            .order_by(Contract.created_at, Contract.id)
            .limit(1)
        )
        with self.database.session() as session:
            row = session.execute(query).first()
        return None if row is None else (row[0], row[1])

    def _decisions(self, cache: DocumentCache) -> list[tuple[Decision, datetime]]:
        """Return the decisions of the cache folder on a text that can be kept, with times."""
        stem = cache.doc_id.removeprefix(DOC_ID_PREFIX)
        path = self.cache_dir / DECISIONS_DIR / f"{stem}.json"
        if not path.is_file():
            return []
        name = f"{DECISIONS_DIR}/{path.name}"
        try:
            entries = DECISION_LIST.validate_json(path.read_bytes())
        except (ValidationError, OSError):
            self.report.left_out.append(Skipped(name, NOT_DECISIONS))
            return []
        kept = []
        for decision in entries:
            if not 0 <= decision.start < decision.end <= len(cache.text):
                reason = BAD_SPAN.format(clause=decision.clause_type)
                self.report.left_out.append(Skipped(name, reason))
                continue
            try:
                kept.append((decision, _utc(decision.at)))
            except ValueError:
                self.report.left_out.append(Skipped(name, BAD_TIME.format(at=decision.at)))
        return kept

    def _file(self, cache: DocumentCache) -> tuple[str, str]:
        """Keep the file of a contract: its PDF copy when there is one, else its text."""
        copy = copy_path(self.cache_dir / DOCUMENTS_DIR, cache.doc_id)
        if copy.is_file():
            return PDF, self.blobs.put(copy.read_bytes(), MEDIA_TYPES[PDF])
        return TXT, self.blobs.put(cache.text.encode("utf-8"), MEDIA_TYPES[TXT])

    def _find_folder(self) -> str | None:
        """Return the id of the folder at the top of the owner's space, or None."""
        with self.database.session() as session:
            top = files.listing(session, self.owner)
        wanted = " ".join(self.report.folder.split()).casefold()
        return next(
            (folder.id for folder in top.folders if folder.name.casefold() == wanted), None
        )

    def _folder(self) -> str:
        """Return the id of the folder, made at the top of the owner's space when missing."""
        if self.folder_id is None:
            with self.database.session() as session:
                folder = files.create_folder(
                    session, self.owner, None, self.report.folder, system=True
                )
            self.folder_id = folder.id
            self.report.folder = folder.name
            self.report.folder_created = True
        return self.folder_id


def _keyed(store: GeneratedStore, record: GeneratedRecord, path: Path) -> bool:
    """Tell whether a saved text's file is the one the store would read for its keys."""
    key = store.key_of(record.doc_id, record.model, record.tau_low, record.tau_high)
    return key == path.stem and record.prompt_version == store.version


def _utc(text: str) -> datetime:
    """Read an ISO time; one without a time zone is taken as UTC."""
    at = datetime.fromisoformat(text)
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def _verdict_key(verdict: Feedback) -> tuple[str, bool, datetime]:
    return verdict.question, verdict.liked, _utc(verdict.at)


def render(report: ImportReport) -> str:
    """Render a report for the command line: the counts, then what was skipped and why."""
    folder = f'"{report.folder}"' + (" (new)" if report.folder_created else "")
    generated = ", ".join(f"{kind} {count}" for kind, count in sorted(report.generated.items()))
    counts = (
        f"contracts: {len(report.imported)} imported, {len(report.reused)} already there, "
        f"{len(report.skipped)} skipped"
    )
    lines = [
        f"cache folder: {report.cache_dir}",
        f"owner: {report.owner}, folder: {folder}",
        counts,
        *(f"  + {name}" for name in report.imported),
        *(f"  = {name}" for name in report.reused),
        f"files: {report.pdfs} PDF, {len(report.imported) - report.pdfs} text",
        f"decisions: {report.decisions}",
        f"saved texts: {sum(report.generated.values())}"
        + (f" ({generated})" if generated else ""),
        f"feedback: {report.feedback}",
    ]
    lines.extend(_skipped_lines("skipped", report.skipped))
    lines.extend(_skipped_lines("left out", report.left_out))
    if not report.changed:
        lines.append("nothing new to import")
    return "\n".join(lines)


def _skipped_lines(title: str, items: list[Skipped]) -> Iterator[str]:
    if not items:
        return
    yield f"{title}: {len(items)}"
    for item in items:
        yield f"  - {item.path}: {item.reason}"
