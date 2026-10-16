# Implementation plan v2 — from the trained model to the demo

**Date:** 2026-10-16
**Supersedes:** v1 of this file (2026-10-04). The design in [PLAN.md](PLAN.md) stands; this
is the order to build it in, now that the model, its thresholds and its numbers exist.

---

## Where this code lives — read first

The application is a **separate repository**, next to the thesis repo, never inside it:

```
~/Desktop/
  diplomski/            thesis: data prep, training (Kaggle), evaluation harness, figures.
                        Produces the model, the thresholds and the test predictions.
  contract-assistant/   THIS repo: the application. Own git history, own virtualenv,
                        own dependencies. Consumes what diplomski produces; never trains.
```

Rules that keep the boundary clean:

- **No application code in `diplomski/`, no training or evaluation-harness changes here.**
  Anything that needs both is written here and reads `diplomski` through `DIPLOMSKI_DIR`.
- **What crosses the boundary is explicit and listed in PLAN.md §1:** the model folder
  (`models/…`, from Kaggle), the two thresholds (`TAU_LOW`, `TAU_HIGH` in `.env`, with the
  path of the `operating_point.json` they came from), `data/clause_types.json` (generated
  once from CUAD's `test.json`), the vendored inference core (+ parity test), and — for the
  end-to-end evaluation only — the thesis test predictions, read from
  `$DIPLOMSKI_DIR/results/`.
- **Own environment.** Never install this repo's dependencies into `diplomski/.venv` or the
  other way round (Gradio, Ollama client, sentence-transformers live only here).

Setting up from nothing:

```bash
cd ~/Desktop
git clone <remote-url> contract-assistant      # or: it already exists locally
cd contract-assistant
uv venv --python 3.12 && uv pip install -e ".[dev]"
cp .env.example .env                            # DIPLOMSKI_DIR=../diplomski, model path, thresholds
make test
```

To publish it: create an empty private repository (e.g. GitHub `contract-assistant`),
then `git remote add origin <url> && git push -u origin main`. Model weights (`models/`),
caches (`cache/`) and `.env` are gitignored and never pushed.

---

## 0. Where we are

| | state |
|---|---|
| **extractor** | **Ettin-150M, seed 3407** (`ettin-encoder-150m_w1024_s256_multi_neg1_seed3407_e3v`): test AUPR 53.21, P@80R 40.32; the best *validation* AUPR of the three seeds (52.52), which is why it is the one used — chosen on val, not test. Ettin-400M (57.31 / 53.86) is a drop-in upgrade later: same tokenizer, same code, ~2.7× slower |
| **operating point** | chosen on validation, scored once on test — `diplomski/results/…seed3407_e3v_operating_point.json` |
| **M0** (skeleton, vendored inference core with offsets, parity test, `clause_types.json`, demo documents) | ✅ done, tests green, **not yet committed** |
| M1–M6 | to build — below |

### The operating point, and why the assistant has three tiers

| tier | rule | validation | test (once) |
|---|---|---|---|
| **found** | top candidate conf ≥ **τ_high = 0.91** | precision 0.754, recall 0.342 | precision **0.735**, recall 0.357 |
| **possible — check** | **τ_low = 0.06** ≤ conf < 0.91 | — | — |
| **not found (checked)** | conf < 0.06 | — | — |
| *everything ≥ τ_low* | | citation F1 0.759, recall 0.883 | citation F1 **0.739**, recall **0.867**, omission 0.023 |

Measured on validation: the errors are mostly **false citations** — a clause type the
annotators did not mark in that contract, cited anyway (266 of 325 errors at τ_low). When
a clause *is* present, the top citation is nearly always the right text (mean Jaccard 0.94;
only 59 wrong spans). So one threshold either hides most real clauses or shows many false
ones. The middle tier, with the quote and page in front of the reader, is the honest
answer — and it is a result for the application chapter, not a UI nicety. Every shown
quote is verbatim in the contract (1.000 on both splits, measured).

---

## Step A — Model weights on this machine (½ hour, one manual step)

The Kaggle CLI only serves a notebook's **latest** version, which is now the 400M run, so
the 150M weights come from the web UI:

1. kaggle.com → *Your Work* → `train_encoder_cuad96805c4c2d` → **Versions** → the version
   from **2026-10-07** (the three-seed run) → **Output**.
2. Open `work/runs/ettin-encoder-150m_w1024_s256_multi_neg1_seed3407_e3v/final/` →
   **Download** (≈ 600 MB, a zip).
3. Unzip to `contract-assistant/models/ettin-150m-cuad-seed3407/` (the folder must contain
   `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`).
   `models/` is gitignored.

Then:

```bash
cd ~/Desktop/contract-assistant
cp .env.example .env    # EXTRACTOR_MODEL_ID=models/ettin-150m-cuad-seed3407, TAU_LOW=0.06, TAU_HIGH=0.91
```

**Parity with the real model.** The M0 parity test uses a public stand-in checkpoint
(the base Ettin has a random QA head). Regenerate the fixture with our weights — in
`diplomski`, harness on CPU fp32, the same two contracts — and add
`tests/test_parity_finetuned.py` (slow). Then the assistant is proven to reproduce, byte
for byte, the predictions the thesis numbers come from.

Hub upload stays planned (public repo, from Kaggle with a *write* token secret) but is not
on the critical path: `from_pretrained` loads the local folder.

**Done when:** `contract-assistant check` loads the model, runs the sliding-window guard,
answers one question on the short demo contract, and both parity tests pass.

---

## Step B — M1: extraction and the cache (1 day)

| file | contents |
|---|---|
| `pdf.py` | `Document(text, page_starts, filename, kind, sha256)`, `load_document(path)` for `.pdf` (pdfplumber, pages joined with `\n`) and `.txt`; `page_of(doc, offset)` via `bisect` |
| `extract/extractor.py` | `Extractor(settings)` loads `QAModel` once; `extract(doc, progress)` runs the 41 CUAD questions (exact wording from `data/clause_types.json`), keeps the **full n-best** (20) per type with offsets and pages; device auto (MPS fp32 on the M4) |
| `store.py` | `cache/<sha256>.json` (schema in PLAN.md §2.1) — save / load / list; **hard error** when the cache's model block or decode settings differ from the current ones |
| `store.import_harness_preds()` | converts a thesis predictions JSONL (`diplomski/results/<tag>_preds.jsonl`) into caches, offsets recovered with `text.find` — makes M5 instant (§ Step F) |
| `cli.py` | `contract-assistant extract <files…>`, `show <doc> [--tau]`, `check` |

Demo caches with the real model, committed under `cache/demo/`. Expected on the M4 at
~2 s per 1,000 characters: short 20.7k chars ≈ 45 s, medium 49.5k ≈ 1¾ min, long 101.6k
≈ 3½ min.

Tests: `page_of` boundaries on a committed 3-page PDF; cache round trip; cache rejected
on a model mismatch; for the short demo contract, the top candidate matches CUAD gold
(Jaccard ≥ 0.5) for at least 15 of its 19 present types.

**Done when:** three demo caches exist, a timing table (chars, windows, seconds, device)
is in PLAN.md §2.2, and a re-run of `extract` on the same file is a cache hit.

---

## Step C — M2: answers, tiers and the verbatim guard (½ day)

Built **before** the routers, with clause types given explicitly, so the core is
finished and tested without any LLM in the loop.

```python
class Citation(BaseModel):  quote: str; page: int | None; conf: float; start: int; end: int
class AnswerItem(BaseModel): clause_type: str; status: Literal["found", "check", "not_found"]
                             citations: list[Citation]; best_conf: float
class Answer(BaseModel):     question: str; route: RouteResult; tau_low: float; tau_high: float
                             items: list[AnswerItem]; dropped_quotes: int

compose(cache, clause_types, tau_low, tau_high) -> list[AnswerItem]
verify_quotes(items, doc_text) -> (items, dropped)   # a quote not found verbatim is never shown, and is counted
render_markdown(answer) -> str
```

- `found`: best ≥ τ_high → up to 3 citations ≥ τ_high
- `check`: τ_low ≤ best < τ_high → the best citation, marked *"verify"*, with its confidence
- `not_found`: best < τ_low → *"not found (checked; best candidate 0.03)"*
- empty route → *"I can only answer about these clause types: …"*

`contract-assistant ask demo/short_supply_agreement.txt --types "Governing Law,Parties"`
prints the template.

Tests: tier boundaries at exactly τ_low / τ_high; a tampered quote is dropped and counted;
markdown carries page and confidence; empty route message.

---

## Step D — M3: routers and their evaluation (1 day)

| file | contents |
|---|---|
| `router/base.py` | `RouteResult(clause_types, kind, latency_ms, raw)`, `Router` protocol |
| `router/embedding.py` | `bge-base-en-v1.5` with its query prefix; 41 descriptions (CUAD "Details" + 2–3 paraphrases each in `clause_types.json`); cosine ≥ threshold, at most k. No LLM — the baseline |
| `router/clients.py` | `OllamaClient` (`format=` JSON schema, `think=False`, temperature 0, 10 s timeout) and `OpenAICompatClient` |
| `router/llm.py` | `LLMRouter(client, fallback)`: schema = array of the 41-name enum; prompt = names + one-line descriptions + 3 worked examples; Pydantic validation; any failure → embedding router, counted |
| `data/router_eval.jsonl` | **60 questions** with gold clause-type sets: ~30 single, ~20 compound, ~10 out of scope. Written by the assistant, **reviewed by the author before any tuning** |
| `eval/router_eval.py` | set precision / recall / F1 per question (mean, and by kind), fallback rate, median latency — for embedding, LLM (qwen3:1.7b) and, as a comparison row, qwen3:4b |

Local LLM, one-time setup on the Mac (≈ 1.1 GB model, runs on the M4):

```bash
brew install ollama && brew services start ollama && ollama pull qwen3:1.7b
```

The embedding threshold is tuned on the first 30 questions and reported on the other 30.

**Done when:** the router table exists. If the LLM router does not beat the embedding
router on compound questions, that is a reported finding, not a bug to hide.

---

## Step E — M4: the Gradio UI (1 day)

`ui/app.py`, `gradio.Blocks` (Gradio 6 installed; the components in PLAN.md §7 are
unchanged): upload → progress over 41 types → chat → the page of the cited text with
the quote highlighted (`HighlightedText`) → the clause table with **two sliders**
(τ_low, τ_high) that re-tier from the cached n-best without re-running the model →
download JSON. The three demo documents are in a dropdown at start-up.

Tier colours: *found* solid, *check* outlined with the word "verify", *not found* grey.
Never colour alone — the status word is always printed.

`contract-assistant serve [--share]`. Manual acceptance = the 2-minute demo script in
PLAN.md §12, run start to finish without an error.

---

## Step F — M5: end-to-end evaluation on CUAD's test split (½ day, no GPU)

No new inference: `import_harness_preds()` turns the thesis predictions for the 102 test
contracts (`$DIPLOMSKI_DIR/results/…seed3407_e3v_preds.jsonl`) into caches, and the parity test
guarantees they are what the assistant itself would compute.

`eval/e2e_eval.py`, for every (test contract, router-eval question) whose gold types
occur in that contract: route → compose → score each routed type against CUAD gold with
CUAD's match rule. Reported per router (embedding, LLM, **oracle**):

| metric | meaning |
|---|---|
| citation precision / recall by tier | how often *found* is right; how much *check* adds |
| false-citation rate | absent clause cited |
| abstention accuracy | absent clause → *not found* |
| routing loss | each metric with the real router minus the oracle router |
| displayed hallucination rate | dropped quotes / shown quotes — expected 0, measured |
| latency | extraction per document (M4), answer per question per router |

Output: `eval/results/e2e_<router>.json` and one markdown table — the application
chapter's table.

---

## Step G — M6: packaging and the defence (½ day)

- README: install, the one manual model step, `make demo`, screenshots of the three tiers
- `make demo` opens the UI with the three pre-extracted documents — the defence never
  waits on extraction
- optional: Ettin-400M as the extractor (`EXTRACTOR_MODEL_ID`), re-run Step F for a
  150M-vs-400M row in the application table; Hub upload of both models

---

## Order and effort

| # | step | needs | est. |
|---|---|---|---|
| 0 | commit M0 | — | 5 min |
| A | model weights + parity with the real model | **you: one download from the Kaggle UI** | ½ h |
| B | M1 extraction + cache + demo caches | A | 1 day |
| C | M2 answers, tiers, guard | B | ½ day |
| D | M3 routers + 60-question eval | **you: `brew install ollama`**; review of the 60 questions | 1 day |
| E | M4 Gradio UI | C, D | 1 day |
| F | M5 end-to-end evaluation | C, D | ½ day |
| G | M6 packaging, demo | E, F | ½ day |

≈ 4½ working days. Nothing needs a GPU. B and C can start the moment the weights are in
`models/`; D's code can be written in parallel and only its evaluation waits on Ollama.
