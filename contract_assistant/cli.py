"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv
from pydantic import BaseModel

from contract_assistant.answer import (
    Answer,
    AnswerItem,
    Definition,
    Passage,
    answer,
    blockquote,
    compose,
    render_markdown,
    restrict_to_cache,
    verify_quotes,
)
from contract_assistant.background import precompute_all, run_in_background
from contract_assistant.clause_types import (
    ClauseType,
    app_clause_types,
    load_clause_types,
    load_extra_clause_types,
)
from contract_assistant.compare import compare_checklists, compare_terms
from contract_assistant.compare import render_markdown as render_comparison
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.definitions import find_definitions, lookup
from contract_assistant.evaluation.e2e import (
    ORACLE,
    E2EReport,
    GoldError,
    OracleRouter,
    evaluate_e2e,
    load_cuad_gold,
)
from contract_assistant.evaluation.e2e import markdown_table as e2e_table
from contract_assistant.evaluation.passage_eval import (
    PassageAnswerReport,
    evaluate_passage_answers,
)
from contract_assistant.evaluation.phrase_eval import PhrasingReport, evaluate_phrasing
from contract_assistant.evaluation.questions import EvalQuestion, Split, load_questions
from contract_assistant.evaluation.router_eval import (
    THRESHOLD_GRID,
    RouterRun,
    evaluate_router,
    threshold_sweep,
    tune_threshold,
)
from contract_assistant.evaluation.router_eval import markdown_table as router_table
from contract_assistant.evaluation.summary_eval import SummaryReport, evaluate_summaries
from contract_assistant.evaluation.term_sheet_eval import (
    TermSheetReport,
    evaluate_term_sheets,
    load_demo_gold,
)
from contract_assistant.explain import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    Explanation,
    request_explanation,
)
from contract_assistant.extract.extractor import Extractor, ModelNotFoundError
from contract_assistant.extract.windowing import assert_sliding_windows
from contract_assistant.passage_answer import APP_ATTEMPTS as PASSAGE_ATTEMPTS
from contract_assistant.passage_answer import request_passage_answer
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.report import build_report, write_report
from contract_assistant.review import RuleFileError, load_rules, review
from contract_assistant.review import render_markdown as render_flags
from contract_assistant.router.base import Router, RouteResult
from contract_assistant.router.clients import LLMClient, LLMError
from contract_assistant.router.factory import (
    BACKEND_NONE,
    UnknownBackendError,
    build_embedding_router,
    build_gen_client,
    build_llm_client,
    build_router,
    load_encoder,
)
from contract_assistant.search import PassageSearch
from contract_assistant.store import (
    DOC_ID_PREFIX,
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    HarnessImportError,
    Store,
    doc_id_of,
    import_harness_preds,
    read_cache,
    write_cache,
)
from contract_assistant.summary import (
    APP_ATTEMPTS as SUMMARY_ATTEMPTS,
)
from contract_assistant.summary import (
    SUMMARY_DIR,
    Summary,
    SummaryStore,
    request_summary,
    summary_items,
)
from contract_assistant.summary import (
    render_markdown as render_summary,
)
from contract_assistant.term_sheet import (
    APP_ATTEMPTS as TERM_SHEET_ATTEMPTS,
)
from contract_assistant.term_sheet import (
    TERM_SHEET_DIR,
    TermSheet,
    TermSheetStore,
    request_term_sheet,
    term_sheet_items,
)
from contract_assistant.term_sheet import (
    render_markdown as render_term_sheet,
)

DEMO_DOCUMENTS = ROOT / "demo" / "demo_documents.json"
CHECK_DOCUMENT = ROOT / "demo" / "short_supply_agreement.txt"
CHECK_CLAUSE = "Governing Law"
QUOTE_WIDTH = 80
DEFINITION_WIDTH = 120
NO_DEFINITIONS = "no defined terms found"
DEFINITION_HEADING = "**Definition: {term}** · p. {page}"
DEFINITIONS_HEADER = ("| Term | Defined as | Page |", "| --- | --- | --- |")
ELLIPSIS = "…"
PRECOMPUTE_NOTICE = "precomputing summaries and key terms in the background"
EXIT_ERROR = 1
EXIT_NO_MODEL = 2
EXIT_USAGE = 2
JSON_INDENT = 2
MS_PER_S = 1000
ROUTER_CHOICES = ("embedding", "llm")
E2E_ROUTER_CHOICES = (ORACLE, *ROUTER_CHOICES)
SPLITS = get_args(Split)
MODEL_FIELDS = {"embedding": "embedding_model_id", "llm": "llm_model", "gen": "gen_model"}
RESULTS_DIR = ROOT / "eval" / "results"
TUNING_RESULT = "embedding_threshold.json"
TUNING_SPLIT = "dev"
ROUTER_RESULTS = "router_*.json"
ROUTER_TABLE = "router.md"
E2E_RESULTS = "e2e_*.json"
E2E_TABLE = "e2e.md"
ORACLE_RESULT = f"e2e_{ORACLE}.json"
NOTHING_TO_SUMMARIZE = "nothing to summarize: no clause was found"
LLM_FAILED = "error: the language model did not answer: {error}"
NOT_FOUND = "error: {name} was not found in this contract"
NOT_IN_CACHE = "error: {name} is not in the analysis of this contract"
EXPLANATION_WITHHELD = "Explanation withheld: {reason}."
NO_KEY_TERMS = "error: no key-term clause was found in {name}"
NO_TERM_COMPARISON = "key terms need a generation model ({settings})"
NO_SAVED_SUMMARY = "no saved summary; run summarize first"
NO_SAVED_TERMS = "no saved key terms; run term-sheet first"
REPORTS_DIR = "reports"
NO_BACKEND = "none"
PASSAGES_MESSAGE = "Passages of the contract closest to the question:"
GENERATED_MESSAGE = (
    "Answer written by the language model from the passages below "
    "(quoted text verified, not a clause match):"
)
GENERATED_WITHHELD = "The language model's answer was withheld: {reason}."
GENERATED_FAILED = "warning: the language model did not answer: {error}"
WARM_UP_QUESTION = "Is there an arbitration clause?"
CLEAR_LINE = "\r\033[K"
CACHE_ERRORS = (CacheMismatchError, CacheIntegrityError)
DOCUMENT_ERRORS = (UnsupportedFileError, EmptyDocumentError, FileNotFoundError)
THESIS_ERRORS = (HarnessImportError, GoldError, FileNotFoundError)


def _thesis_paths(settings: Settings) -> tuple[Path, Path]:
    preds_path = settings.diplomski_dir / "results" / f"{settings.extractor_name}_preds.jsonl"
    return preds_path, settings.diplomski_dir / "data" / "raw" / "test.json"


def _cmd_import_demo(args: argparse.Namespace) -> int:
    settings = load_settings()
    preds_path, cuad_json = _thesis_paths(settings)
    demo = json.loads(DEMO_DOCUMENTS.read_text(encoding="utf-8"))
    slugs = {entry["cuad_title"]: entry["slug"] for entry in demo}
    try:
        caches = import_harness_preds(preds_path, cuad_json, settings, titles=list(slugs))
    except (HarnessImportError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    for title, cache in caches.items():
        path = settings.demo_cache_dir / f"{slugs[title]}.json"
        write_cache(cache, path)
        print(f"{slugs[title]}: {len(cache.clauses)} clause types -> {_relative(path)}")
    return 0


def _model_missing(error: ModelNotFoundError) -> int:
    print(f"error: {error}", file=sys.stderr)
    print("download the weights as described in IMPLEMENTATION.md, Step A", file=sys.stderr)
    return EXIT_NO_MODEL


def _progress(done: int, total: int, name: str) -> None:
    print(f"{CLEAR_LINE}  {done}/{total} {name}", end="", file=sys.stderr, flush=True)


def _existing_cache(
    path: Path, doc_id: str, args: argparse.Namespace, settings: Settings, store: Store
) -> tuple[DocumentCache, Path] | None:
    found = settings.demo_cache_dir / f"{path.stem}.json" if args.demo else store.find(doc_id)
    if found is None or not found.is_file():
        return None
    try:
        cache = store.load_path(found, settings)
    except CacheMismatchError:
        return None
    return (cache, found) if cache.doc_id == doc_id else None


def _missing_types(cache: DocumentCache, settings: Settings) -> list[str]:
    return [c.name for c in app_clause_types(settings) if c.name not in cache.clauses]


def _save(
    cache: DocumentCache, demo_target: Path, args: argparse.Namespace, store: Store
) -> Path:
    if not args.demo:
        return store.save(cache)
    write_cache(cache, demo_target)
    return demo_target


def _extract_file(
    path: Path,
    args: argparse.Namespace,
    settings: Settings,
    store: Store,
    extractor: Extractor | None,
) -> tuple[str, Extractor | None]:
    doc = load_document(path)
    doc_id = doc_id_of(doc.text)
    existing = _existing_cache(path, doc_id, args, settings, store)
    if existing and not _missing_types(existing[0], settings):
        return f"{path.name}: cache hit -> {_relative(existing[1])}", extractor
    extractor = extractor or Extractor.from_settings(settings)
    if existing:
        return _extend_file(path, existing, args, extractor, store), extractor
    cache = extractor.extract(doc, _progress)
    print(CLEAR_LINE, end="", file=sys.stderr)
    target = _save(cache, settings.demo_cache_dir / f"{path.stem}.json", args, store)
    seconds = cache.timing["extract_s"]
    return (
        f"{path.name}: {cache.source.n_chars} chars, {seconds} s -> {_relative(target)}",
        extractor,
    )


def _extend_file(
    path: Path,
    existing: tuple[DocumentCache, Path],
    args: argparse.Namespace,
    extractor: Extractor,
    store: Store,
) -> str:
    cache, found = existing
    started = time.perf_counter()
    extended = extractor.extend(cache, _progress)
    seconds = time.perf_counter() - started
    print(CLEAR_LINE, end="", file=sys.stderr)
    added = len(extended.clauses) - len(cache.clauses)
    target = _save(extended, found, args, store)
    return f"{path.name}: extended with {added} clause types, {seconds:.1f} s -> {_relative(target)}"


def _cmd_extract(args: argparse.Namespace) -> int:
    settings = load_settings()
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    extractor = None
    for path in args.files:
        try:
            line, extractor = _extract_file(path, args, settings, store, extractor)
        except ModelNotFoundError as error:
            return _model_missing(error)
        except (*DOCUMENT_ERRORS, *CACHE_ERRORS) as error:
            print(f"error: {path}: {error}", file=sys.stderr)
            return EXIT_ERROR
        print(line)
    return 0


def _resolve_cache(target: str, settings: Settings, store: Store) -> DocumentCache | None:
    if target.startswith(DOC_ID_PREFIX):
        return store.load(target, settings)
    path = Path(target)
    if path.suffix == ".json":
        return store.load_path(path, settings)
    return store.load(doc_id_of(load_document(path).text), settings)


def _load_cache(target: str, settings: Settings) -> DocumentCache | None:
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    try:
        cache = _resolve_cache(target, settings, store)
    except (*DOCUMENT_ERRORS, *CACHE_ERRORS) as error:
        print(f"error: {target}: {error}", file=sys.stderr)
        return None
    if cache is None:
        print(f"no cache for {target}; run extract first", file=sys.stderr)
    return cache


def _quote(text: str, width: int = QUOTE_WIDTH) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    return text[: width - len(ELLIPSIS)] + ELLIPSIS


def _row(name: str, width: int, cache: DocumentCache) -> str:
    if name not in cache.clauses:
        return f"{name:<{width}}  missing"
    candidates = cache.clauses[name]
    if not candidates:
        return f"{name:<{width}}  {'-':>5}  {'-':>4}  "
    best = candidates[0]
    page = "n/a" if best.page is None else best.page
    return f"{name:<{width}}  {best.conf:>5.3f}  {page:>4}  {_quote(best.text)}"


def _cmd_show(args: argparse.Namespace) -> int:
    settings = load_settings()
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    print(
        f"{cache.source.filename}  {cache.source.n_chars} chars  "
        f"model {cache.model.name}  origin {cache.model.origin}"
    )
    names = _names(app_clause_types(settings))
    width = max(len(name) for name in names)
    for name in names:
        print(_row(name, width, cache))
    return 0


def _names(clause_types: list[ClauseType]) -> list[str]:
    return [clause_type.name for clause_type in clause_types]


def _extra_names(settings: Settings) -> frozenset[str]:
    return frozenset(_names(load_extra_clause_types(settings.extra_clause_types_path)))


def _split_names(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(",")]


def _valid_ask_arguments(args: argparse.Namespace, names: list[str]) -> bool:
    if args.types is None:
        if args.question.strip():
            return True
        print("error: a question is required unless --types is given", file=sys.stderr)
        return False
    unknown = [name for name in _split_names(args.types) if name not in names]
    if not unknown:
        return True
    print(f"error: unknown clause types: {', '.join(unknown)}", file=sys.stderr)
    print("\n".join(names), file=sys.stderr)
    return False


def _llm_settings(settings: Settings) -> str:
    return f"LLM_BACKEND={settings.llm_backend}, LLM_MODEL={settings.llm_model}"


def _gen_settings(settings: Settings) -> str:
    return f"LLM_BACKEND={settings.llm_backend}, GEN_MODEL={settings.gen_model}"


def _gen_client(command: str, settings: Settings) -> LLMClient | None:
    client = build_gen_client(settings)
    if client is None:
        print(
            f"error: {command} needs a generation model ({_gen_settings(settings)})",
            file=sys.stderr,
        )
    return client


def _asked_router(
    choice: str | None, settings: Settings, clause_types: list[ClauseType]
) -> Router | None:
    if choice == "embedding":
        settings = replace(settings, llm_backend=BACKEND_NONE)
    router = build_router(settings, clause_types)
    if choice is not None and router.kind != choice:
        print(
            f"error: the {choice} router is not available ({_llm_settings(settings)})",
            file=sys.stderr,
        )
        return None
    return router


def _route(
    args: argparse.Namespace, settings: Settings, clause_types: list[ClauseType]
) -> RouteResult | None:
    if args.types is not None:
        types = _split_names(args.types)
        return RouteResult(clause_types=types, kind="explicit", latency_ms=0.0)
    router = _asked_router(args.router, settings, clause_types)
    return None if router is None else router.route(args.question)


def _definition_row(definition: Definition) -> str:
    page = "n/a" if definition.page is None else definition.page
    quote = _quote(definition.quote, DEFINITION_WIDTH).replace("|", "\\|")
    return f"| {definition.term} | {quote} | {page} |"


def _cmd_definitions(args: argparse.Namespace) -> int:
    cache = _load_cache(args.target, load_settings())
    if cache is None:
        return EXIT_ERROR
    definitions = find_definitions(cache.text, cache.page_starts)
    if not definitions:
        print(NO_DEFINITIONS)
    elif args.json:
        print(json.dumps([d.model_dump(mode="json") for d in definitions], indent=JSON_INDENT))
    else:
        print("\n".join([*DEFINITIONS_HEADER, *map(_definition_row, definitions)]))
    return 0


def _definition_answer(
    args: argparse.Namespace, cache: DocumentCache, settings: Settings
) -> Answer | None:
    definition = lookup(find_definitions(cache.text, cache.page_starts), args.question)
    if definition is None:
        return None
    route = RouteResult(clause_types=[], kind="definition", latency_ms=0.0)
    return Answer(
        question=args.question,
        route=route,
        tau_low=settings.tau_low,
        tau_high=settings.tau_high,
        items=[],
        dropped_quotes=0,
        definition=definition,
    )


def _render_definition(definition: Definition) -> str:
    page = "n/a" if definition.page is None else definition.page
    heading = DEFINITION_HEADING.format(term=definition.term, page=page)
    return f"{heading}\n{blockquote(definition.quote)}"


def _render(result: Answer, names: list[str]) -> str:
    if result.definition is not None:
        return _render_definition(result.definition)
    text = render_markdown(result, names)
    if not result.passages:
        return text
    passages = [_render_passage(passage) for passage in result.passages]
    return "\n\n".join([text, *_render_generated(result), PASSAGES_MESSAGE, *passages])


def _render_generated(result: Answer) -> list[str]:
    generated = result.generated
    if generated is None:
        return []
    if generated.text is None:
        return [GENERATED_WITHHELD.format(reason=generated.reason)]
    return [f"{GENERATED_MESSAGE}\n\n{generated.text}"]


def _render_passage(passage: Passage) -> str:
    quote = blockquote(passage.quote)
    return quote if passage.page is None else f"p. {passage.page}\n{quote}"


def _with_passages(result: Answer, cache: DocumentCache, settings: Settings) -> Answer:
    search = PassageSearch(load_encoder(settings.embedding_model_id))
    return result.model_copy(update={"passages": search.search(cache, result.question)})


def _with_generated(result: Answer, settings: Settings) -> Answer:
    client = build_gen_client(settings)
    if client is None or not result.passages:
        return result
    try:
        generated = request_passage_answer(
            result.question, result.passages, client, attempts=PASSAGE_ATTEMPTS
        )
    except LLMError as error:
        print(GENERATED_FAILED.format(error=error), file=sys.stderr)
        return result
    return result.model_copy(update={"generated": generated})


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    clause_types = app_clause_types(settings)
    names = _names(clause_types)
    if not _valid_ask_arguments(args, names):
        return EXIT_USAGE
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    result = None if args.types is not None else _definition_answer(args, cache, settings)
    if result is not None:
        return _print_answer(result, names, args)
    route = _route(args, settings, clause_types)
    if route is None:
        return EXIT_ERROR
    route = restrict_to_cache(route, cache)
    result = answer(
        args.question,
        route,
        cache,
        settings.tau_low,
        settings.tau_high,
        unvalidated=_extra_names(settings),
    )
    if args.types is None and not route.clause_types:
        result = _with_generated(_with_passages(result, cache, settings), settings)
    return _print_answer(result, names, args)


def _print_answer(result: Answer, names: list[str], args: argparse.Namespace) -> int:
    if args.json:
        print(result.model_dump_json(indent=JSON_INDENT))
        return 0
    print(_render(result, names))
    if args.types is None:
        route = result.route
        print(f"\nrouter: {route.kind} · {route.latency_ms:.0f} ms")
    return 0


def _summary(
    cache: DocumentCache,
    items: list[AnswerItem],
    client: LLMClient,
    fresh: bool,
    settings: Settings,
) -> Summary:
    store = SummaryStore(settings.cache_dir / SUMMARY_DIR)
    key = (cache.doc_id, client.model, settings.tau_low, settings.tau_high)
    saved = None if fresh else store.load(*key)
    if saved is not None:
        return saved
    summary = request_summary(
        cache.doc_id,
        items,
        client,
        settings.tau_low,
        settings.tau_high,
        attempts=SUMMARY_ATTEMPTS,
    )
    # A withheld summary is not saved, so the next run asks the model again.
    if summary.text is not None:
        store.save(summary)
    return summary


def _cmd_summarize(args: argparse.Namespace) -> int:
    settings = load_settings()
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    client = _gen_client("summarize", settings)
    if client is None:
        return EXIT_ERROR
    names = _names(app_clause_types(settings))
    items = summary_items(
        cache, names, settings.tau_low, settings.tau_high, _extra_names(settings)
    )
    if not items:
        print(NOTHING_TO_SUMMARIZE, file=sys.stderr)
        return EXIT_ERROR
    try:
        summary = _summary(cache, items, client, args.fresh, settings)
    except LLMError as error:
        print(LLM_FAILED.format(error=error), file=sys.stderr)
        return EXIT_ERROR
    print(summary.model_dump_json(indent=JSON_INDENT) if args.json else render_summary(summary))
    return 0


def _term_sheet(
    cache: DocumentCache,
    items: list[AnswerItem],
    client: LLMClient,
    fresh: bool,
    settings: Settings,
) -> TermSheet:
    store = TermSheetStore(settings.cache_dir / TERM_SHEET_DIR)
    key = (cache.doc_id, client.model, settings.tau_low, settings.tau_high)
    saved = None if fresh else store.load(*key)
    if saved is not None:
        return saved
    sheet = request_term_sheet(
        cache.doc_id,
        items,
        client,
        settings.tau_low,
        settings.tau_high,
        attempts=TERM_SHEET_ATTEMPTS,
    )
    # A withheld term sheet is not saved, so the next run asks the model again.
    if sheet.reason is None:
        store.save(sheet)
    return sheet


def _sheets(
    args: argparse.Namespace, client: LLMClient, settings: Settings
) -> list[tuple[str, TermSheet]] | None:
    sheets = []
    for target in args.targets:
        cache = _load_cache(target, settings)
        if cache is None:
            return None
        name = Path(cache.source.filename).stem
        items = term_sheet_items(
            cache, settings.tau_low, settings.tau_high, _extra_names(settings)
        )
        if not items:
            print(NO_KEY_TERMS.format(name=name), file=sys.stderr)
            return None
        sheets.append((name, _term_sheet(cache, items, client, args.fresh, settings)))
    return sheets


def _cmd_term_sheet(args: argparse.Namespace) -> int:
    settings = load_settings()
    client = _gen_client("term-sheet", settings)
    if client is None:
        return EXIT_ERROR
    try:
        sheets = _sheets(args, client, settings)
    except LLMError as error:
        print(LLM_FAILED.format(error=error), file=sys.stderr)
        return EXIT_ERROR
    if sheets is None:
        return EXIT_ERROR
    if args.json:
        data = {name: sheet.model_dump(mode="json") for name, sheet in sheets}
        print(json.dumps(data, indent=JSON_INDENT))
    else:
        print(render_term_sheet(sheets))
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    settings = load_settings()
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    try:
        rules = load_rules(known_types=_names(app_clause_types(settings)))
    except RuleFileError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    flags = review(cache, rules, settings.tau_low, settings.tau_high, _extra_names(settings))
    if args.json:
        print(json.dumps([flag.model_dump(mode="json") for flag in flags], indent=JSON_INDENT))
    else:
        print(render_flags(flags))
    return 0


def _compared_sheet(
    cache: DocumentCache, client: LLMClient | None, fresh: bool, settings: Settings
) -> TermSheet | None:
    items = term_sheet_items(cache, settings.tau_low, settings.tau_high, _extra_names(settings))
    if not items:
        return None
    if client is not None:
        return _term_sheet(cache, items, client, fresh, settings)
    store = TermSheetStore(settings.cache_dir / TERM_SHEET_DIR)
    key = (cache.doc_id, settings.gen_model, settings.tau_low, settings.tau_high)
    return None if fresh else store.load(*key)


def _cmd_compare(args: argparse.Namespace) -> int:
    settings = load_settings()
    left = _load_cache(args.left, settings)
    right = _load_cache(args.right, settings)
    if left is None or right is None:
        return EXIT_ERROR
    checklist = compare_checklists(
        left,
        right,
        _names(app_clause_types(settings)),
        settings.tau_low,
        settings.tau_high,
        _extra_names(settings),
    )
    client = build_gen_client(settings)
    try:
        sheets = [_compared_sheet(cache, client, args.fresh, settings) for cache in (left, right)]
    except LLMError as error:
        print(LLM_FAILED.format(error=error), file=sys.stderr)
        return EXIT_ERROR
    if client is None and not all(sheets):
        print(NO_TERM_COMPARISON.format(settings=_gen_settings(settings)), file=sys.stderr)
    terms = compare_terms(*sheets) if all(sheets) else []
    documents = [Path(cache.source.filename).stem for cache in (left, right)]
    if args.json:
        data = {
            "documents": documents,
            "terms": [row.model_dump(mode="json") for row in terms],
            "clauses": [row.model_dump(mode="json") for row in checklist],
        }
        print(json.dumps(data, indent=JSON_INDENT))
    else:
        print(render_comparison((documents[0], documents[1]), checklist, terms))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    settings = load_settings()
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    names = _names(app_clause_types(settings))
    extras = _extra_names(settings)
    try:
        rules = load_rules(known_types=names)
    except RuleFileError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    key = (cache.doc_id, settings.gen_model, settings.tau_low, settings.tau_high)
    summary = SummaryStore(settings.cache_dir / SUMMARY_DIR).load(*key)
    sheet = TermSheetStore(settings.cache_dir / TERM_SHEET_DIR).load(*key)
    if summary is None:
        print(NO_SAVED_SUMMARY, file=sys.stderr)
    if sheet is None:
        print(NO_SAVED_TERMS, file=sys.stderr)
    name = Path(cache.source.filename).stem
    report = build_report(
        cache,
        name,
        settings,
        names,
        extras,
        summary=summary,
        sheet=sheet,
        flags=review(cache, rules, settings.tau_low, settings.tau_high, extras),
        definitions=find_definitions(cache.text, cache.page_starts),
        model=None if settings.llm_backend == NO_BACKEND else settings.gen_model,
    )
    out_dir = args.out or settings.cache_dir / REPORTS_DIR
    for path in write_report(report, out_dir, name):
        print(_relative(path))
    return 0


def _explainable_quote(name: str, cache: DocumentCache, settings: Settings) -> str | None:
    if name not in cache.clauses:
        print(NOT_IN_CACHE.format(name=name), file=sys.stderr)
        return None
    items = compose(cache, [name], settings.tau_low, settings.tau_high, _extra_names(settings))
    [item], _ = verify_quotes(items, cache.text)
    if item.citation is None:
        print(NOT_FOUND.format(name=name), file=sys.stderr)
        return None
    return item.citation.quote


def _render_explanation(explanation: Explanation) -> str:
    if explanation.text is None:
        return EXPLANATION_WITHHELD.format(reason=explanation.reason)
    language = LANGUAGES[explanation.language].split(",")[0]
    return f"**{explanation.clause_type}** · {language}\n{explanation.text}"


def _cmd_explain(args: argparse.Namespace) -> int:
    settings = load_settings()
    names = _names(app_clause_types(settings))
    if args.type not in names:
        print(f"error: unknown clause types: {args.type}", file=sys.stderr)
        print("\n".join(names), file=sys.stderr)
        return EXIT_USAGE
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    quote = _explainable_quote(args.type, cache, settings)
    if quote is None:
        return EXIT_ERROR
    client = _gen_client("explain", settings)
    if client is None:
        return EXIT_ERROR
    try:
        explanation = request_explanation(args.type, quote, args.language, client)
    except LLMError as error:
        print(LLM_FAILED.format(error=error), file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(explanation.model_dump_json(indent=JSON_INDENT))
    else:
        print(_render_explanation(explanation))
    return 0


def _slug(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace(":", "-")


def _model_name(settings: Settings, choice: str) -> str | None:
    name = MODEL_FIELDS.get(choice)
    return None if name is None else getattr(settings, name)


def _with_model(settings: Settings, choice: str, model: str | None) -> Settings:
    if model is None or choice not in MODEL_FIELDS:
        return settings
    return replace(settings, **{MODEL_FIELDS[choice]: model})


def _write_result(name: str, text: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _oracle_first(path: Path) -> tuple[bool, str]:
    return path.name != ORACLE_RESULT, path.name


def _read_results[T: BaseModel](pattern: str, model: type[T]) -> list[T]:
    paths = sorted(RESULTS_DIR.glob(pattern), key=_oracle_first)
    return [model.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]


def _publish(name: str, table: str) -> None:
    _write_result(name, table)
    print(table)


def _warmed_router(choice: str, settings: Settings) -> Router | None:
    router = _asked_router(choice, settings, load_clause_types(settings.clause_types_path))
    if router is None:
        return None
    # The first call loads the model, so its latency says nothing about a question.
    router.route(WARM_UP_QUESTION)
    return router


def _cmd_router_tune(args: argparse.Namespace) -> int:
    settings = load_settings()
    dev = load_questions(settings.router_eval_path, TUNING_SPLIT)
    router = build_embedding_router(settings, load_clause_types(settings.clause_types_path))
    sweep = threshold_sweep(router, dev, THRESHOLD_GRID)
    threshold = tune_threshold(router, dev, THRESHOLD_GRID)
    tuning = {
        "model": settings.embedding_model_id,
        "split": TUNING_SPLIT,
        "n_questions": len(dev),
        "threshold": threshold,
        "f1": sweep[threshold],
        "sweep": {f"{value:.2f}": f1 for value, f1 in sweep.items()},
    }
    path = _write_result(TUNING_RESULT, json.dumps(tuning, indent=JSON_INDENT))
    print(f"threshold {threshold:.2f}, dev F1 {sweep[threshold]:.3f} -> {_relative(path)}")
    print(f"set EMBEDDING_THRESHOLD={threshold:.2f} in .env and .env.example to use it")
    return 0


def _cmd_router_eval(args: argparse.Namespace) -> int:
    settings = _with_model(load_settings(), args.router, args.model)
    questions = load_questions(settings.router_eval_path, args.split)
    router = _warmed_router(args.router, settings)
    if router is None:
        return EXIT_ERROR
    run = RouterRun(
        router=router.kind,
        model=_model_name(settings, args.router),
        split=args.split,
        threshold=settings.embedding_threshold if args.router == "embedding" else None,
        report=evaluate_router(router, questions),
    )
    name = f"router_{run.router}_{_slug(run.model)}_{run.split}.json"
    _write_result(name, run.model_dump_json(indent=JSON_INDENT))
    _publish(ROUTER_TABLE, router_table(_read_results(ROUTER_RESULTS, RouterRun)))
    return 0


def _e2e_report(
    router: Router, questions: list[EvalQuestion], settings: Settings, model: str | None
) -> E2EReport:
    preds_path, cuad_json = _thesis_paths(settings)
    caches = import_harness_preds(preds_path, cuad_json, settings)
    gold = load_cuad_gold(cuad_json, load_clause_types(settings.clause_types_path))
    return evaluate_e2e(
        router, questions, caches, gold, settings.tau_low, settings.tau_high, model=model
    )


def _e2e_filename(report: E2EReport) -> str:
    if report.model is None:
        return f"e2e_{report.router}.json"
    return f"e2e_{report.router}_{_slug(report.model)}.json"


def _publish_e2e_table() -> int:
    try:
        table = e2e_table(_read_results(E2E_RESULTS, E2EReport))
    except ValueError as error:
        print(f"error: stale results in {_relative(RESULTS_DIR)}: {error}", file=sys.stderr)
        print("re-run e2e-eval for the oracle and for every other router", file=sys.stderr)
        return EXIT_ERROR
    _publish(E2E_TABLE, table)
    return 0


def _cmd_e2e_eval(args: argparse.Namespace) -> int:
    settings = _with_model(load_settings(), args.router, args.model)
    questions = load_questions(settings.router_eval_path)
    if args.router == ORACLE:
        router = OracleRouter(questions)
    else:
        router = _warmed_router(args.router, settings)
    if router is None:
        return EXIT_ERROR
    try:
        report = _e2e_report(router, questions, settings, _model_name(settings, args.router))
    except THESIS_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    _write_result(_e2e_filename(report), report.model_dump_json(indent=JSON_INDENT))
    return _publish_e2e_table()


def _demo_caches(settings: Settings) -> dict[str, DocumentCache]:
    paths = sorted(settings.demo_cache_dir.glob("*.json"))
    return {path.stem: read_cache(path, settings) for path in paths}


def _phrasing_line(report: PhrasingReport, path: Path) -> str:
    replies = report.calls - report.llm_failures
    rate = "n/a" if report.violation_rate is None else f"{report.violation_rate:.3f}"
    return (
        f"violations {report.violations} of {replies} replies (rate {rate}), "
        f"unquoted {report.unquoted_replies}, failures {report.llm_failures}, "
        f"invalid {report.invalid_replies} -> {_relative(path)}"
    )


def _cmd_phrase_eval(args: argparse.Namespace) -> int:
    settings = _with_model(load_settings(), "llm", args.model)
    client = build_llm_client(settings)
    if client is None:
        print(f"error: phrase-eval needs an LLM ({_llm_settings(settings)})", file=sys.stderr)
        return EXIT_ERROR
    questions = load_questions(settings.router_eval_path)
    caches = _demo_caches(settings)
    report = evaluate_phrasing(client, questions, caches, settings.tau_low, settings.tau_high)
    name = f"phrasing_{_slug(report.model)}.json"
    path = _write_result(name, report.model_dump_json(indent=JSON_INDENT))
    print(_phrasing_line(report, path))
    return 0


def _summary_line(report: SummaryReport, path: Path) -> str:
    mean = "n/a"
    if report.mean_latency_ms is not None:
        mean = f"{report.mean_latency_ms / MS_PER_S:.1f} s"
    return (
        f"summary-eval: {report.model} · {report.documents} documents · "
        f"{report.withheld} withheld · {report.retried} retried · "
        f"{report.llm_failures} failures · mean {mean} "
        f"→ {_relative(path)}"
    )


def _cmd_summary_eval(args: argparse.Namespace) -> int:
    settings = _with_model(load_settings(), "gen", args.model)
    client = _gen_client("summary-eval", settings)
    if client is None:
        return EXIT_ERROR
    report = evaluate_summaries(
        client,
        _demo_caches(settings),
        _names(app_clause_types(settings)),
        settings.tau_low,
        settings.tau_high,
        _extra_names(settings),
    )
    name = f"summary_{_slug(report.model)}.json"
    path = _write_result(name, report.model_dump_json(indent=JSON_INDENT))
    print(_summary_line(report, path))
    return 0


def _mean_seconds(latency_ms: float | None) -> str:
    return "n/a" if latency_ms is None else f"{latency_ms / MS_PER_S:.1f} s"


def _term_sheet_line(report: TermSheetReport, path: Path) -> str:
    return (
        f"term-sheet-eval: {report.model} · {report.documents} documents · "
        f"{report.values} values · {report.agreeing} agree with gold · "
        f"{report.rejected} rejected · {report.missing} missing · "
        f"{report.withheld} withheld · {report.llm_failures} failures · "
        f"mean {_mean_seconds(report.mean_latency_ms)} → {_relative(path)}"
    )


def _cmd_term_sheet_eval(args: argparse.Namespace) -> int:
    settings = _with_model(load_settings(), "gen", args.model)
    client = _gen_client("term-sheet-eval", settings)
    if client is None:
        return EXIT_ERROR
    report = evaluate_term_sheets(
        client,
        _demo_caches(settings),
        load_demo_gold(DEMO_DOCUMENTS),
        settings.tau_low,
        settings.tau_high,
        _extra_names(settings),
    )
    path = _write_result(
        f"term_sheet_{_slug(report.model)}.json", report.model_dump_json(indent=JSON_INDENT)
    )
    print(_term_sheet_line(report, path))
    return 0


def _passage_line(report: PassageAnswerReport, path: Path) -> str:
    return (
        f"passage-eval: {report.model} · {report.calls} calls · {report.answered} answered · "
        f"{report.withheld} withheld · {report.no_passages} without passages · "
        f"{report.llm_failures} failures · "
        f"mean {_mean_seconds(report.mean_latency_ms)} → {_relative(path)}"
    )


def _cmd_passage_eval(args: argparse.Namespace) -> int:
    settings = _with_model(load_settings(), "gen", args.model)
    client = _gen_client("passage-eval", settings)
    if client is None:
        return EXIT_ERROR
    search = PassageSearch(load_encoder(settings.embedding_model_id))
    report = evaluate_passage_answers(
        client, load_questions(settings.router_eval_path), _demo_caches(settings), search
    )
    path = _write_result(
        f"passages_{_slug(report.model)}.json", report.model_dump_json(indent=JSON_INDENT)
    )
    print(_passage_line(report, path))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        extractor = Extractor.from_settings(settings)
    except ModelNotFoundError as error:
        return _model_missing(error)
    windows = assert_sliding_windows(
        extractor.qa.tok, settings.max_seq_len, settings.doc_stride
    )
    print(f"device: {extractor.device}")
    print(f"windows: {windows}")
    question = next(c for c in extractor.clause_types if c.name == CHECK_CLAUSE).cuad_question
    text = load_document(CHECK_DOCUMENT).text
    candidates = extractor.qa(
        question,
        text,
        max_seq_len=settings.max_seq_len,
        doc_stride=settings.doc_stride,
        max_answer_len=settings.max_answer_len,
        top_k=settings.top_k,
        n_best_size=settings.n_best_size,
    )
    if not candidates:
        print(f"{CHECK_CLAUSE}: no candidates")
        return EXIT_ERROR
    print(f"{CHECK_CLAUSE}: {candidates[0].conf:.3f}  {_quote(candidates[0].text)}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    # Imported here so the other commands do not pay for loading Gradio.
    from contract_assistant.ui.app import Services, build_app, launch_app

    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    app_types = app_clause_types(settings)
    router = build_router(settings, app_types)
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    try:
        extractor = Extractor.from_settings(settings)
    except ModelNotFoundError:
        extractor = None
        print(
            f"uploads need the extractor model in {settings.extractor_path}",
            file=sys.stderr,
        )
    gen = build_gen_client(settings)
    summaries = SummaryStore(settings.cache_dir / SUMMARY_DIR)
    term_sheets = TermSheetStore(settings.cache_dir / TERM_SHEET_DIR)
    extras = _extra_names(settings)
    services = Services(
        store,
        router,
        extractor,
        app_types,
        settings,
        PassageSearch(load_encoder(settings.embedding_model_id)),
        extras,
        gen=gen,
        summaries=summaries,
        term_sheets=term_sheets,
        rules=load_rules(known_types=[clause.name for clause in app_types]),
    )
    if gen is not None and not args.no_precompute:
        print(PRECOMPUTE_NOTICE, file=sys.stderr)
        names = [clause.name for clause in app_types]
        run_in_background(
            precompute_all, store, settings, names, gen, summaries, term_sheets, extras
        )
    launch_app(build_app(services), services, share=args.share, port=args.port)
    return 0


def _relative(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _add_evaluation_commands(commands: argparse._SubParsersAction) -> None:
    router_tune = commands.add_parser(
        "router-tune", help="choose the embedding threshold on the dev questions"
    )
    router_tune.set_defaults(func=_cmd_router_tune)
    router_eval = commands.add_parser(
        "router-eval", help="score a router on the evaluation questions"
    )
    router_eval.add_argument("--router", choices=ROUTER_CHOICES, required=True)
    router_eval.add_argument(
        "--model", help="model of the router instead of the configured one"
    )
    router_eval.add_argument(
        "--split", choices=SPLITS, default="test", help="questions to score"
    )
    router_eval.set_defaults(func=_cmd_router_eval)
    e2e_eval = commands.add_parser(
        "e2e-eval", help="score a router end to end on CUAD's test contracts"
    )
    e2e_eval.add_argument(
        "--router",
        choices=E2E_ROUTER_CHOICES,
        required=True,
        help="the oracle routes with the gold clause types",
    )
    e2e_eval.add_argument("--model", help="model of the router; the oracle has none")
    e2e_eval.set_defaults(func=_cmd_e2e_eval)
    phrase_eval = commands.add_parser(
        "phrase-eval", help="check that LLM-phrased answers quote only verified text"
    )
    phrase_eval.add_argument("--model", help="LLM model instead of the configured one")
    phrase_eval.set_defaults(func=_cmd_phrase_eval)
    summary_eval = commands.add_parser(
        "summary-eval", help="check that generated summaries quote only verified text"
    )
    summary_eval.add_argument("--model", help="generation model instead of the configured one")
    summary_eval.set_defaults(func=_cmd_summary_eval)
    term_sheet_eval = commands.add_parser(
        "term-sheet-eval", help="check generated term sheets against CUAD's gold spans"
    )
    term_sheet_eval.add_argument(
        "--model", help="generation model instead of the configured one"
    )
    term_sheet_eval.set_defaults(func=_cmd_term_sheet_eval)
    passage_eval = commands.add_parser(
        "passage-eval", help="check answers written from passages for out-of-scope questions"
    )
    passage_eval.add_argument("--model", help="generation model instead of the configured one")
    passage_eval.set_defaults(func=_cmd_passage_eval)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contract-assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    import_demo = commands.add_parser(
        "import-demo", help="build the demo caches from the thesis predictions"
    )
    import_demo.set_defaults(func=_cmd_import_demo)
    extract = commands.add_parser("extract", help="extract the clause candidates of documents")
    extract.add_argument("files", nargs="+", type=Path)
    extract.add_argument("--demo", action="store_true", help="write to the demo cache folder")
    extract.set_defaults(func=_cmd_extract)
    show = commands.add_parser("show", help="show the cached result of a document")
    show.add_argument("target", help="document, cache file or sha256: document id")
    show.set_defaults(func=_cmd_show)
    ask = commands.add_parser("ask", help="answer a question about an extracted document")
    ask.add_argument("target", help="document, cache file or sha256: document id")
    ask.add_argument("question", nargs="?", default="", help="the question to answer")
    selection = ask.add_mutually_exclusive_group()
    selection.add_argument("--types", help="comma separated clause types; no router is used")
    selection.add_argument(
        "--router", choices=ROUTER_CHOICES, help="use this router; fail if it is unavailable"
    )
    ask.add_argument("--json", action="store_true", help="print the answer as JSON")
    ask.set_defaults(func=_cmd_ask)
    definitions = commands.add_parser("definitions", help="list the terms a contract defines")
    definitions.add_argument("target", help="document, cache file or sha256: document id")
    definitions.add_argument("--json", action="store_true", help="print the terms as JSON")
    definitions.set_defaults(func=_cmd_definitions)
    summarize = commands.add_parser(
        "summarize", help="write a short summary from the clauses found in a document"
    )
    summarize.add_argument("target", help="document, cache file or sha256: document id")
    summarize.add_argument("--json", action="store_true", help="print the summary as JSON")
    summarize.add_argument(
        "--fresh",
        action="store_true",
        help="ask the model again instead of using the saved summary",
    )
    summarize.set_defaults(func=_cmd_summarize)
    term_sheet = commands.add_parser(
        "term-sheet", help="fill in the key terms of one or more documents, side by side"
    )
    term_sheet.add_argument(
        "targets", nargs="+", metavar="target", help="document, cache file or sha256: id"
    )
    term_sheet.add_argument("--json", action="store_true", help="print the term sheets as JSON")
    term_sheet.add_argument(
        "--fresh", action="store_true", help="ask the model again instead of using saved sheets"
    )
    term_sheet.set_defaults(func=_cmd_term_sheet)
    review_parser = commands.add_parser(
        "review", help="flag review points from the clauses found, with no model"
    )
    review_parser.add_argument("target", help="document, cache file or sha256: document id")
    review_parser.add_argument("--json", action="store_true", help="print the flags as JSON")
    review_parser.set_defaults(func=_cmd_review)
    compare = commands.add_parser(
        "compare", help="put the key terms and clauses of two documents side by side"
    )
    compare.add_argument("left", help="document, cache file or sha256: document id")
    compare.add_argument("right", help="document, cache file or sha256: document id")
    compare.add_argument("--json", action="store_true", help="print the comparison as JSON")
    compare.add_argument(
        "--fresh", action="store_true", help="ask the model again instead of using saved sheets"
    )
    compare.set_defaults(func=_cmd_compare)
    report = commands.add_parser(
        "report", help="write a Markdown and a DOCX report from the saved analysis"
    )
    report.add_argument("target", help="document, cache file or sha256: document id")
    report.add_argument("--out", type=Path, help="folder for the report files")
    report.set_defaults(func=_cmd_report)
    explain = commands.add_parser("explain", help="restate one clause in plain language")
    explain.add_argument("target", help="document, cache file or sha256: document id")
    explain.add_argument("--type", required=True, help="clause type to restate")
    explain.add_argument("--language", choices=list(LANGUAGES), default=DEFAULT_LANGUAGE)
    explain.add_argument("--json", action="store_true", help="print the explanation as JSON")
    explain.set_defaults(func=_cmd_explain)
    _add_evaluation_commands(commands)
    check = commands.add_parser("check", help="load the model and answer one question")
    check.set_defaults(func=_cmd_check)
    serve = commands.add_parser("serve", help="start the web app")
    serve.add_argument("--share", action="store_true", help="create a public Gradio link")
    serve.add_argument("--port", type=int, help="port to listen on")
    serve.add_argument(
        "--no-precompute",
        action="store_true",
        help="do not prepare summaries and key terms in the background",
    )
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    load_dotenv()
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except UnknownBackendError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
