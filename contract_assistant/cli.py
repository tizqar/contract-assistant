"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from contract_assistant.config import ROOT, load_settings
from contract_assistant.store import HarnessImportError, import_harness_preds, write_cache

DEMO_DOCUMENTS = ROOT / "demo" / "demo_documents.json"


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
        return 1
    for title, cache in caches.items():
        path = settings.demo_cache_dir / f"{slugs[title]}.json"
        write_cache(cache, path)
        print(f"{slugs[title]}: {len(cache.clauses)} clause types -> {_relative(path)}")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    load_dotenv()
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
