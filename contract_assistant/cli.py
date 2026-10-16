"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv
from pydantic import BaseModel

from contract_assistant.answer import answer, render_markdown
from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.evaluation.e2e import (
    ORACLE,
    E2EReport,
    GoldError,
    OracleRouter,
    evaluate_e2e,
    load_cuad_gold,
)
from contract_assistant.evaluation.e2e import markdown_table as e2e_table
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
from contract_assistant.extract.extractor import Extractor, ModelNotFoundError
from contract_assistant.extract.windowing import assert_sliding_windows
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.router.base import Router, RouteResult
from contract_assistant.router.factory import (
    BACKEND_NONE,
    UnknownBackendError,
    build_embedding_router,
    build_llm_client,
    build_router,
)
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

DEMO_DOCUMENTS = ROOT / "demo" / "demo_documents.json"
CHECK_DOCUMENT = ROOT / "demo" / "short_supply_agreement.txt"
CHECK_CLAUSE = "Governing Law"
QUOTE_WIDTH = 80
ELLIPSIS = "…"
EXIT_ERROR = 1
EXIT_NO_MODEL = 2
EXIT_USAGE = 2
JSON_INDENT = 2
ROUTER_CHOICES = ("embedding", "llm")
E2E_ROUTER_CHOICES = (ORACLE, *ROUTER_CHOICES)
SPLITS = get_args(Split)
MODEL_FIELDS = {"embedding": "embedding_model_id", "llm": "llm_model"}
RESULTS_DIR = ROOT / "eval" / "results"
TUNING_RESULT = "embedding_threshold.json"
TUNING_SPLIT = "dev"
ROUTER_RESULTS = "router_*.json"
ROUTER_TABLE = "router.md"
E2E_RESULTS = "e2e_*.json"
E2E_TABLE = "e2e.md"
ORACLE_RESULT = f"e2e_{ORACLE}.json"
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


def _valid_cache_path(store: Store, doc_id: str, settings: Settings) -> Path | None:
    found = store.find(doc_id)
    if found is None:
        return None
    try:
        store.load_path(found, settings)
    except CacheMismatchError:
        return None
    return found


def _extract_file(
    path: Path,
    args: argparse.Namespace,
    settings: Settings,
    store: Store,
    extractor: Extractor | None,
) -> tuple[str, Extractor | None]:
    doc = load_document(path)
    doc_id = doc_id_of(doc.text)
    found = None if args.demo else _valid_cache_path(store, doc_id, settings)
    if found:
        return f"{path.name}: cache hit -> {_relative(found)}", extractor
    extractor = extractor or Extractor.from_settings(settings)
    cache = extractor.extract(doc, _progress)
    print(CLEAR_LINE, end="", file=sys.stderr)
    if args.demo:
        target = settings.demo_cache_dir / f"{path.stem}.json"
        write_cache(cache, target)
    else:
        target = store.save(cache)
    seconds = cache.timing["extract_s"]
    return (
        f"{path.name}: {cache.source.n_chars} chars, {seconds} s -> {_relative(target)}",
        extractor,
    )


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


def _quote(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= QUOTE_WIDTH:
        return text
    return text[: QUOTE_WIDTH - len(ELLIPSIS)] + ELLIPSIS


def _row(name: str, width: int, cache: DocumentCache) -> str:
    candidates = cache.clauses.get(name, [])
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
    names = [clause_type.name for clause_type in load_clause_types(settings.clause_types_path)]
    width = max(len(name) for name in names)
    for name in names:
        print(_row(name, width, cache))
    return 0


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


def _asked_router(choice: str | None, settings: Settings) -> Router | None:
    if choice == "embedding":
        settings = replace(settings, llm_backend=BACKEND_NONE)
    router = build_router(settings)
    if choice is not None and router.kind != choice:
        print(
            f"error: the {choice} router is not available ({_llm_settings(settings)})",
            file=sys.stderr,
        )
        return None
    return router


def _route(args: argparse.Namespace, settings: Settings) -> RouteResult | None:
    if args.types is not None:
        types = _split_names(args.types)
        return RouteResult(clause_types=types, kind="explicit", latency_ms=0.0)
    router = _asked_router(args.router, settings)
    return None if router is None else router.route(args.question)


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    names = [clause_type.name for clause_type in load_clause_types(settings.clause_types_path)]
    if not _valid_ask_arguments(args, names):
        return EXIT_USAGE
    cache = _load_cache(args.target, settings)
    if cache is None:
        return EXIT_ERROR
    route = _route(args, settings)
    if route is None:
        return EXIT_ERROR
    result = answer(args.question, route, cache, settings.tau_low, settings.tau_high)
    if args.json:
        print(result.model_dump_json(indent=JSON_INDENT))
        return 0
    print(render_markdown(result, names))
    if args.types is None:
        print(f"\nrouter: {route.kind} · {route.latency_ms:.0f} ms")
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
    router = _asked_router(choice, settings)
    if router is None:
        return None
    # The first call loads the model, so its latency says nothing about a question.
    router.route(WARM_UP_QUESTION)
    return router


def _cmd_router_tune(args: argparse.Namespace) -> int:
    settings = load_settings()
    dev = load_questions(settings.router_eval_path, TUNING_SPLIT)
    router = build_embedding_router(settings)
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
    paths = sorted(settings.demo_cache_dir.glob("*.json"))
    caches = {path.stem: read_cache(path, settings) for path in paths}
    report = evaluate_phrasing(client, questions, caches, settings.tau_low, settings.tau_high)
    name = f"phrasing_{_slug(report.model)}.json"
    path = _write_result(name, report.model_dump_json(indent=JSON_INDENT))
    print(_phrasing_line(report, path))
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
    router = build_router(settings)
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    try:
        extractor = Extractor.from_settings(settings)
    except ModelNotFoundError:
        extractor = None
        print(
            f"uploads need the extractor model in {settings.extractor_path}",
            file=sys.stderr,
        )
    clause_types = load_clause_types(settings.clause_types_path)
    services = Services(store, router, extractor, clause_types, settings)
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
    _add_evaluation_commands(commands)
    check = commands.add_parser("check", help="load the model and answer one question")
    check.set_defaults(func=_cmd_check)
    serve = commands.add_parser("serve", help="start the web app")
    serve.add_argument("--share", action="store_true", help="create a public Gradio link")
    serve.add_argument("--port", type=int, help="port to listen on")
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
