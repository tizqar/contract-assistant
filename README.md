# contract-assistant

A local legal assistant for one contract at a time: upload a contract, ask questions in
plain language, get **verbatim quotes with a page number and a confidence** — or an
explicit *"not found"*.

The expensive model (a fine-tuned 150M-parameter encoder, trained in the companion
`diplomski` repo) runs **once at upload** and extracts all 41 CUAD clause types with
character offsets. Every later question is a cache lookup: a router only maps the question
to clause types and never writes legal content. The intended router is a small local LLM
(Qwen3 via Ollama); today an embedding router (bge-base-en-v1.5) does the routing. Quotes
are verified against the source text before they are shown.

**Status:** built are the inference core, the extractor and cache, three-tier answers with
the verbatim guard, the embedding router and the Gradio app. Still to do are the LLM
router, the router evaluation and the end-to-end evaluation. Build order:
[IMPLEMENTATION.md](IMPLEMENTATION.md).

## Install

Python 3.12.

```bash
uv venv && uv pip install -e ".[dev]"     # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env
```

The `api` extra (`openai`) is optional. The Makefile expects the environment in `.venv/`.

## Model weights

The fine-tuned 150M extractor is the one manual step: download it from the Kaggle run
output and unzip it to `models/ettin-150m-cuad-seed3407/` (gitignored). Step A in
[IMPLEMENTATION.md](IMPLEMENTATION.md) has the details.

Without the model the app still opens the three demo documents from their caches. Uploading
a new contract needs the model, and the app says so.

## Run

```bash
make demo      # http://127.0.0.1:7860
make ui-dev    # fake backends, no model, reloads on save
make test      # fast tests
make lint      # ruff
```

The first `make demo` loads the embedding model and downloads it if it is not cached.
`make ui-dev` is meant for the VS Code Simple Browser. The other commands are
`contract-assistant import-demo | extract | show | check | serve`.

## How answers read

Each requested clause type gets one of three tiers, from the extractor's confidence
(thresholds chosen on validation in `diplomski`: τ_high = 0.91, τ_low = 0.06):

| tier | meaning |
|---|---|
| found | confidence at or above τ_high; the quote is shown with page and confidence |
| possible, verify | between τ_low and τ_high; the quote is shown, to be checked by the reader |
| not found | below τ_low; only the best candidate score is reported |

A quote is shown only if it matches the source text exactly; otherwise it is withheld.
Three answers on the short demo contract:

```
**Governing Law** — found · p. n/a · confidence 1.00
> This agreement shall be governed by the Laws of England (English common and statutory Law).

**Exclusivity** — possible, verify · p. n/a · confidence 0.41
> The Manufacturer grants exclusive rights to the Customer for the term of ten (10) years …

**Source Code Escrow** — not found (checked; best candidate 0.00)
```

The demo documents are plain text, so there are no page numbers (`p. n/a`); a PDF gives
the page.

| | |
|---|---|
| ![A found answer](docs/screenshots/found.png) | ![A possible, verify answer](docs/screenshots/check.png) |
| A found answer with its quote and page. | A tier that asks the reader to verify. |
| ![A not found answer](docs/screenshots/not_found.png) | ![A question that maps to several clause types](docs/screenshots/compound.png) |
| An explicit not found. | One question routed to several clause types. |

## Demo documents

Three CUAD contracts with pre-extracted caches: see [demo/README.md](demo/README.md).

Results tables will appear under `eval/results/` once the evaluations are built.

## Repository layout

This is its **own repository**, a sibling of the thesis repo — not a subfolder of it:

```
~/Desktop/diplomski/            training, evaluation, thesis figures   (produces the model)
~/Desktop/contract-assistant/   this application                       (consumes it)
```

Separate git history, separate virtualenv. See [IMPLEMENTATION.md](IMPLEMENTATION.md),
"Where this code lives", for the rules and the setup from scratch.

## Relationship to `diplomski`

| from `diplomski` | into here |
|---|---|
| fine-tuned extractor (local folder) and its run tag | `EXTRACTOR_MODEL_ID`, `EXTRACTOR_NAME` |
| operating thresholds (val-selected) | `TAU_LOW`, `TAU_HIGH` |
| 41 clause types + CUAD's question wording | `data/clause_types.json` |
| inference core + sliding-window guard | vendored, with a parity test |
| CUAD's span match rule | `contract_assistant/cuad_match.py`, vendored |
| test predictions of the chosen run | demo caches, via `contract-assistant import-demo` |

Training, evaluation against published CUAD numbers, and the thesis figures stay in
`diplomski`.
