"""Command line interface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from contract_assistant.clause_types import load_clause_types
from contract_assistant.config import ROOT, Settings, load_settings
from contract_assistant.extract.extractor import Extractor, ModelNotFoundError
from contract_assistant.extract.windowing import assert_sliding_windows
from contract_assistant.pdf import EmptyDocumentError, UnsupportedFileError, load_document
from contract_assistant.store import (
    DOC_ID_PREFIX,
    CacheIntegrityError,
    CacheMismatchError,
    DocumentCache,
    HarnessImportError,
    Store,
    doc_id_of,
    import_harness_preds,
    write_cache,
)

DEMO_DOCUMENTS = ROOT / "demo" / "demo_documents.json"
CHECK_DOCUMENT = ROOT / "demo" / "short_supply_agreement.txt"
CHECK_CLAUSE = "Governing Law"
QUOTE_WIDTH = 80
ELLIPSIS = "…"
EXIT_ERROR = 1
EXIT_NO_MODEL = 2
CLEAR_LINE = "\r\033[K"
CACHE_ERRORS = (CacheMismatchError, CacheIntegrityError)
DOCUMENT_ERRORS = (UnsupportedFileError, EmptyDocumentError, FileNotFoundError)


def _cmd_import_demo(args: argparse.Namespace) -> int:
    settings = load_settings()
    preds_path = settings.diplomski_dir / "results" / f"{settings.extractor_name}_preds.jsonl"
    cuad_json = settings.diplomski_dir / "data" / "raw" / "test.json"
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
    store = Store(settings.cache_dir, settings.demo_cache_dir)
    try:
        cache = _resolve_cache(args.target, settings, store)
    except (*DOCUMENT_ERRORS, *CACHE_ERRORS) as error:
        print(f"error: {args.target}: {error}", file=sys.stderr)
        return EXIT_ERROR
    if cache is None:
        print(f"no cache for {args.target}; run extract first", file=sys.stderr)
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


def _relative(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


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
    check = commands.add_parser("check", help="load the model and answer one question")
    check.set_defaults(func=_cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    load_dotenv()
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
