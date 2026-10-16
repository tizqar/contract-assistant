# Contract assistant — plan

**Date:** 2026-10-04
**Status:** M0, M1, M3 and M4 done; M2 and M5 built, with the oracle end-to-end rows in
and the router numbers waiting for the review of the evaluation questions. Build order in
[IMPLEMENTATION.md](IMPLEMENTATION.md) (v2, 2026-10-16)
**Companion repo:** `diplomski` (encoder training, evaluation harness, figures). That repo
produces the model and the operating threshold; this repo consumes them. Training never
happens here.

---

## 0. What this is

A local legal assistant for one contract at a time: upload a contract, ask questions in
plain language, get answers that are **verbatim quotes with a page number and a
confidence**, or an explicit *"not found"*. It is the application chapter of the thesis
and the demo at the defence.

It implements §4 of `diplomski/RESEARCH_AND_REDESIGN.md` — *extract once, answer many*:

```
UPLOAD (once per document, ~1 min on an M4)
  PDF/TXT ──pdfplumber──> text + page_starts[]
         ──fine-tuned Ettin (41 clause types × sliding window)──> CACHE
         CACHE = {clause_type: [{quote, start, end, page, conf}, ...]}

QUESTION (many per document, instant)
  "which law governs, and can they terminate without cause?"
     └─> ROUTER: question -> {Governing Law, Termination For Convenience}
           (embedding baseline, or local Qwen3 constrained to the 41-label enum)
         └─> CACHE lookup, tiers by τ_low / τ_high
             └─> VERIFY: every quote must occur verbatim in the document text
                 └─> ANSWER: quotes + page + confidence, or "not found (checked)"
```

The LLM never produces legal content. It picks clause types. Quotes come from the
extractor and are checked against the source text before display, so the **displayed
hallucination rate is zero by construction** — and it is measured, not asserted.

### Why this shape (decisions already made, with the thesis in mind)

| decision | choice | why |
|---|---|---|
| extractor | fine-tuned `ettin-encoder-150m` from the training repo (run 2 recipe: 1024/256, `multi` labels) | 47.87 AUPR on CUAD's official test split; beats published RoBERTa-base, ties RoBERTa-large / DeBERTa-xlarge. 400M is a drop-in later |
| where the expensive model runs | **once, at upload** | per-question latency becomes a cache lookup; the demo feels instant |
| LLM role | **router only**, closed set of 41 labels | removes the LLM from the correctness path entirely |
| LLM choice | **local Qwen3-1.7B via Ollama** (Ollama's default 4-bit quantisation, ~1.1 GB, runs on the M4 CPU/GPU); any OpenAI-compatible API behind the same interface | consistent with the thesis claim (cheap, local, private — legal documents should not leave the machine); the router task is a closed 41-label set, 1.7B is enough. 4B only if M2 numbers say so; a Kaggle/Colab-hosted router is possible but unnecessary |
| answer text | **template** (`Governing Law — found · p. 12 · confidence 0.94` and the quote); LLM phrasing is only measured, behind a verbatim-quote check (section 4) | the template is auditable; LLM phrasing adds nothing to correctness |
| UI | **Gradio 6 `Blocks`** over a UI-independent core package | fastest Python route to upload + chat + highlighted citations; the core stays testable and could sit behind FastAPI later |
| runs on | the M4 (MPS fp32 for Ettin, Ollama for Qwen3) | no GPU dependency; measured ~2 s per 1,000 characters of contract for the 150M model |

---

## 1. Interfaces with the training repo — fixed up front

These are the only things that cross the repo boundary. Everything else is independent.

| artefact | from `diplomski` | into this repo | how |
|---|---|---|---|
| **model weights** | `runs/<RUN_NAME>/final/` (575 MB) | `EXTRACTOR_MODEL_ID` env / config | **public** HuggingFace Hub repo `<user>/ettin-encoder-150m-cuad-qa` (MIT weights, CC-BY-4.0 data; public makes the thesis reproducible from a model id). Uploaded straight from the Kaggle notebook — the files never need to touch the laptop |
| **operating threshold τ** | `results/<tag>_proposal_metrics.json` → `tau_val_selected` | `TAU_LOW` / `TAU_HIGH` in `config.py`, slider defaults in the UI | copied as a number with its provenance string |
| **decode settings** | `max_seq_len=1024, doc_stride=256, top_k=20, max_answer_len=512` | `config.py` | must equal training; the eval harness proved the window mismatch is silent and fatal |
| **clause types + CUAD questions** | `src/schema.py` + the 41 question strings in `data/raw/test.json` | `data/clause_types.json`: `{id, name, cuad_question, description}` | the extractor must be asked with CUAD's exact wording — it was trained on it |
| **inference core** | `QAModel` + n-best postprocessing in `src/eval_cuad_qa.py` | `contract_assistant/extract/qa_core.py` **(vendored, + char offsets)** | the harness returns `(text, conf)`; citations need `(text, conf, start, end)`, which the n-best already has internally. A parity test asserts identical `(text, conf)` lists on a sample contract |
| **sliding-window guard** | `src/windowing.py` | vendored verbatim | `transformers>=5.16` silently truncates every contract to two windows; the pin `<5.16` and the 100 ms behavioural check both come along |

---

## 2. Architecture

```
contract-assistant/
  contract_assistant/
    config.py            model id, decode settings, τ, device, LLM backend -- all env-overridable
    clause_types.py      the 41 clause types with CUAD's exact questions
    pdf.py               pdfplumber -> text, page_starts; page_of(offset); txt passthrough
    extract/
      qa_core.py         vendored QAModel + n-best WITH offsets; device auto (cuda > mps > cpu)
      windowing.py       vendored sliding-window guard
      extractor.py       run 41 questions over a document -> DocumentCache; progress callback
    store.py             cache schema; cache/<sha256>.json + demo/cache/ read/write; harness import
    cuad_match.py        CUAD's span match rule (Jaccard >= 0.5, Parties relaxed)
    router/
      base.py            Router protocol: route(question) -> RouteResult
      embedding.py       bge-base: question vs descriptions and paraphrases, threshold + top-k
      llm.py             LLMRouter(client): JSON-schema-constrained {"clause_types": [...]}
      clients.py         OllamaClient (native, format=schema, think=False) | OpenAICompatClient
      factory.py         build_router(settings): the router that LLM_BACKEND selects
    answer.py            tiers, one citation per type, exact quote guard, Markdown template
    phrase.py            LLM paragraph from the verified quotes; check of its quoted spans
    evaluation/
      questions.py       loads data/router_eval.jsonl
      router_eval.py     set P/R/F1 per router; embedding threshold tuning (section 5)
      e2e.py             citation outcomes on CUAD test contracts, automatic (section 6)
      phrase_eval.py     violation rate of LLM paragraphs against the verified quotes
    ui/
      app.py             Gradio Blocks
      views.py           caches and answers -> component values (pure functions)
    cli.py               import-demo | extract | show | check | ask | router-tune | router-eval
                         | e2e-eval | phrase-eval | serve
  data/
    clause_types.json          41 entries, generated from the training repo
    router_descriptions.json   2-3 hand-written paraphrases per clause type
    router_eval.jsonl          60 questions with gold clause-type sets (section 5)
  eval/results/          results JSON and markdown tables, committed
  tests/
  demo/cache/            the three demo caches, committed
  cache/                 (gitignored) one JSON per processed document
  pyproject.toml  README.md  PLAN.md
```

### 2.1 Data shapes

**Cache, one file per document** (`cache/<sha256>.json`; the demo caches are
`demo/cache/<slug>.json`, committed and read-only):

```json
{
  "schema_version": 1,
  "doc_id": "sha256:…",
  "source": {"filename": "ACME_Supply_Agreement.pdf", "kind": "pdf", "n_chars": 46848, "n_pages": 23},
  "model": {"name": "ettin-encoder-150m_w1024_s256_multi_neg1_seed3407_e3v",
            "path": "models/ettin-150m-cuad-seed3407", "max_seq_len": 1024, "doc_stride": 256,
            "max_answer_len": 512, "top_k": 20, "n_best_size": 20, "device": "mps",
            "origin": "extractor"},
  "page_starts": [0, 2103, 4311],
  "text": "…",
  "clauses": {
    "Governing Law": [{"text": "This Agreement shall be governed by the laws of the State of New York.",
                       "start": 48210, "end": 48290, "page": 12, "conf": 0.94}],
    "Non-Compete": []
  },
  "timing": {"extract_s": 51.3}
}
```

- **Identity.** A cache is reused only when `model.name` (the run tag) and the five decode
  settings equal `Settings.decode_settings()`; `path` and `device` are informational, so
  the harness's `runs/<tag>/final` and the app's `models/…` are the same model.
- **Integrity.** `text` is the full extracted document; on load `sha256(text)` must equal
  `doc_id` and every candidate's `text[start:end]` must equal its `text`.
- **Origin.** `extractor` for caches written by the app; `harness` for caches imported
  from the thesis test predictions (`store.import_harness_preds`), whose offsets come
  from `text.find` and may point at an earlier copy of a repeated passage.
- **No threshold.** τ belongs to answers, not to caches (`TAU_LOW`, `TAU_HIGH` in
  `config.py`).
- `.txt` documents have `page_starts: []` and `page: null`, shown as "n/a".

`clauses[type]` keeps the full n-best (up to 20 candidates with their confidences), not
just the top one, so the τ slider in the UI works without re-running the model.

**Answer** (what the UI renders and what the evaluations score):

```json
{
  "question": "which law governs, and can they terminate without cause?",
  "route": {"clause_types": ["Governing Law", "Termination For Convenience"], "kind": "llm",
            "latency_ms": 1400,
            "raw": "{\"clause_types\": [\"Governing Law\", \"Termination For Convenience\"]}"},
  "tau_low": 0.06,
  "tau_high": 0.91,
  "items": [
    {"clause_type": "Governing Law", "status": "found", "best_conf": 0.94,
     "citation": {"quote": "…", "page": 12, "conf": 0.94, "start": 48210, "end": 48290},
     "withheld": false},
    {"clause_type": "Termination For Convenience", "status": "not_found", "best_conf": 0.03,
     "citation": null, "withheld": false}
  ],
  "dropped_quotes": 0
}
```

`status` follows the confidence of the top candidate: `found` when it is at least τ_high
(0.91); `check`, shown as "possible, verify", from τ_low (0.06) up to τ_high; `not_found`
below τ_low, or when the type has no candidate (`best_conf` 0).

- **One citation per type**, the top candidate. Confidences are a softmax over the n-best
  and the null score, so they sum to at most 1 and at most one candidate can reach τ_high.
  The other candidates are in the sources view.
- **`withheld`** is true when the citation failed the guard of section 4: `citation` is
  then null, the item keeps its tier, and `dropped_quotes` counts it.
- **`route.kind`** is `explicit` (types given to `ask --types`), `oracle` (the gold types,
  in the evaluations), `embedding`, `llm` or `llm_fallback`. `raw` is the LLM's reply, or
  the error class on a fallback.

### 2.2 Extraction (`extract/extractor.py`)

For each of the 41 clause types: CUAD's exact question, sliding window over the whole
text at 1024/256, SQuAD-2.0 n-best with the null candidate, `token_is_max_context`,
dedupe to 20 unique texts, softmax — **identical to the evaluation harness**, so the
numbers measured in the thesis are the numbers the assistant runs on. The only addition
is returning character offsets alongside each candidate.

Cost is the same 41 × windows forward passes the harness pays. Measured (eval harness,
same windows):

| | median contract (6.4k tokens, ~25 pp.) | p90 (28k tokens) |
|---|---|---|
| T4 fp16 | ~12 s | ~50 s |
| **M4, MPS fp32** | **~50 s** | ~4 min |
| Ettin-400M | ×2.7 | ×2.7 |

The UI shows a progress bar over 41 types; the result is cached by content hash, so a
re-upload of the same file costs nothing.

**Measured with the app's extractor** (seed-3407 weights, M4, MPS fp32, 2026-10-16;
`contract-assistant extract --demo`; windows counted with the Governing Law question):

| document | chars | windows | extract | per 1,000 chars |
|---|---|---|---|---|
| `short_supply_agreement.txt` | 20,693 | 6 | 35.8 s | 1.73 s |
| `medium_sponsorship_agreement.txt` | 49,518 | 12 | 75.1 s | 1.52 s |
| `long_outsourcing_agreement.txt` | 101,639 | 31 | 184.1 s | 1.81 s |

Against the thesis test predictions of the same run (Kaggle T4, fp16), on the three demo
contracts:

- **Parity.** On CPU fp32 the app reproduces the harness exactly on the two parity
  contracts (`tests/test_parity_finetuned.py`, texts identical, confidences within 1e-5).
- **Top-1.** The top-1 text is the same for 122 of 123 contract–type pairs. The one change
  is Parties in the long contract, a near-tie of two candidates (0.0846 vs 0.0840).
- **Lower ranks.** 19, 16 and 15 types per contract differ in the n-best: adjacent ranks
  swap, or one candidate at the tail is replaced.
- **Confidences.** The median gap between the two runs is ~1e-6. Only 2 of ~2,450 shared
  candidates differ by more than 5e-3: 7.8e-3 (short, Competitive Restriction Exception)
  and 1.1e-2 (long, Exclusivity).
- **Offsets.** 48, 46 and 66 candidates have a different `start`. The imported caches
  used `text.find`, which picks the first copy of a repeated passage; the extractor's
  offsets are the spans the model actually scored.

**M1 acceptance** (`tests/test_demo_acceptance.py`): the top candidate matches CUAD gold
for 14 of 19 present types (short), 18 of 18 (medium) and 15 of 19 (long), on both the
imported and the extracted caches. Plan 01 asked for 15 of 19 on the short contract; the
model does not reach it. The five misses are:

- Change Of Control and Irrevocable Or Perpetual License: a wrong top-1; for Change Of
  Control a matching span is ranked second.
- Effective Date: the top-1 is the agreement date.
- IP Ownership Assignment: no candidate matches.
- Parties: the top-1 is `the Customer` and the gold is `'the Customer'`, which CUAD's
  rule does not match.

The floors in the test are these measured counts.

### 2.3 Pages (`pdf.py`)

`page_starts[i]` = cumulative character offset where page *i* begins in the extracted
text; `page_of(offset) = bisect_right(page_starts, offset)`. Because the model runs on
the *same* extracted text, offsets map to pages exactly — no alignment problem. For
`.txt` inputs (CUAD's own files) pages are `n/a`; CUAD ships `full_contract_pdf/`, so the
demo can use real PDFs.

---

## 3. Router

Input: a free-form question. Output: a **set** of clause types from the closed list of
41, possibly empty. Two implementations behind one interface, both evaluated. A route
carries the types, how they were chosen (`kind`), the latency and, for the LLM, the `raw`
reply.

**`EmbeddingRouter` (baseline, no LLM).** `bge-base-en-v1.5` with its query instruction
prefix. Each clause type is represented by its CUAD description and two or three
hand-written paraphrases (`data/router_descriptions.json`); a type's score is the maximum
cosine similarity between the question and any of its texts. The router returns the types
scoring at least `EMBEDDING_THRESHOLD`, best first, at most 5 (the largest gold set in
section 5). The threshold, 0.55, is a hand-set placeholder until the value that
`router-tune` chooses on the dev split replaces it. Deterministic and needs no server.
Expected to handle single-intent questions and fail on compound or abstract ones ("the
important dates").

**`LLMRouter` (default).** Qwen3-1.7B through Ollama (`qwen3:4b` for the comparison row)
with `format=` the JSON schema of `{"clause_types": [<enum of 41>]}` — constrained
decoding, so an invalid label cannot be emitted — `think=False` for latency, temperature 0
and a 10 s timeout. The prompt has the task in two sentences, the 41 names with one-line
descriptions and three worked examples (single, compound, out of scope). The examples are
not in the evaluation set; a test enforces it. The reply is validated with Pydantic (only
the 41 names, no other keys) and duplicates are removed, keeping order. `[]` is a valid
answer, not a failure: it means out of scope. On an LLM error, malformed JSON or an
invalid name, the embedding router's route is returned with kind `llm_fallback` and the
error class in `raw`, and a warning is logged. A failed call is not retried; the router
falls back instead.

**Closed-world behaviour.** Questions outside the 41 types ("what about data privacy?")
route to the empty set and the assistant says so, listing what it *can* answer. This is
a feature: the system never pretends.

**Backends.** `clients.py` has `OllamaClient` (native library) and `OpenAICompatClient`
(any OpenAI-compatible endpoint, including a hosted model behind an API key).
`build_router` in `factory.py` reads `LLM_BACKEND`:

| `LLM_BACKEND` | router |
|---|---|
| `ollama` (default) | `LLMRouter` over `OllamaClient` (`LLM_MODEL`, default `qwen3:1.7b`), with the embedding router as fallback. If the server does not list the model at start-up (Ollama not running, or the model not pulled), a warning is logged and the embedding router is used alone, so a missing Ollama costs nothing per question |
| `openai` | `LLMRouter` over `OpenAICompatClient` (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL`; needs the `api` extra), with the embedding router as fallback |
| `none` | the embedding router |

The demo runs `ollama`; the API path exists for the comparison row and for anyone without
a capable laptop.

---

## 4. Answer composition and the guard (`answer.py`)

1. For each routed type, take the top candidate in the cache and tier its confidence:
   `found` at τ_high (0.91) or above, `check` from τ_low (0.06), `not_found` below. One
   citation per type (section 2.1).
2. **Verify** every citation: it is shown only if `text[start:end] == quote`, exactly. The
   extractor's offsets make this exact, so there is no whitespace normalisation and no
   fuzzy matching (`rapidfuzz` is not a dependency). A citation that fails is withheld: the
   item keeps its tier and renders "quote withheld (failed verification)", and
   `dropped_quotes` counts it. That counter is the *displayed hallucination rate* reported
   in the thesis. Expected 0; measured anyway.
3. Render the template:

```
**Governing Law** — found · p. 12 · confidence 0.94
> This Agreement shall be governed by the laws of the State of New York.

**Exclusivity** — possible, verify · p. n/a · confidence 0.42
> …

**Termination For Convenience** — not found (checked; best candidate 0.03)
```

The status word is always printed. Quotes are Markdown-escaped, blockquoted and cut at 400
characters with "…". An empty route answers "I can only answer about these clause types:"
followed by the 41 names.

4. LLM phrasing is measured offline only (`phrase.py`, `phrase-eval`); the app always shows
   the template and there is no `ANSWER_STYLE` setting. The LLM writes one paragraph from
   the verified items. Every span in double quotes, straight or curly, must occur word for
   word in one of the verified quotes, compared with whitespace collapsed; any other span
   is a violation. `phrase-eval` phrases each in-scope evaluation question on each of the
   three demo contracts with the configured LLM, using the gold clause types as the route,
   and reports the violation rate (`eval/results/phrasing_<model>.json`). It exists to
   measure how often a small LLM breaks the quote rule (a thesis number), not to improve
   answers. Replies with no quoted span are counted separately.

---

## 5. Router evaluation set (`data/router_eval.jsonl`)

60 questions, each with its gold set of clause types: 30 single-type, 20 compound and 10
out of scope (empty gold set). One JSON object per line: `id`, `question`, `gold`, `kind`,
`split`. The `dev` and `test` splits hold 30 questions each, stratified (15 single, 10
compound and 5 out of scope per split). The twelve questions below are included as
written. Every clause type is in at least one gold set, and the abstract questions ("the
important dates") are compound, with a gold set. First dozen, to fix the style:

| question | gold set |
|---|---|
| Which law governs this agreement? | Governing Law |
| Who are the parties? | Parties |
| When was it signed, and when does it take effect? | Agreement Date, Effective Date |
| When does it expire, and does it renew on its own? | Expiration Date, Renewal Term, Notice Period To Terminate Renewal |
| Can either side walk away without cause? | Termination For Convenience |
| Is there a non-compete? | Non-Compete |
| Are there exclusivity obligations? | Exclusivity |
| Is liability capped anywhere? Is anything uncapped? | Cap On Liability, Uncapped Liability |
| Can this be assigned if the company is sold? | Anti-Assignment, Change Of Control |
| What licences are granted, and are any perpetual? | License Grant, Irrevocable Or Perpetual License |
| Give me all the important dates. | Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal |
| What does it say about data privacy? | ∅ (out of scope) |

The full 60 are written in M2 and reviewed by the author before any router is tuned or
scored on them; that review is still to come. They are never used to tune the extractor.

**Scoring.** For each question, set precision, recall and F1 of the routed types against
the gold set, averaged over all questions and by kind. An empty gold set scores 1 if the
route is empty and 0 otherwise; a non-empty gold set with an empty route scores 0. The
report adds exact-set accuracy, out-of-scope accuracy (the share of out-of-scope
questions routed to the empty set), the fallback rate of the LLM router, and the median
and p90 routing latency.

**Tuning.** The embedding threshold is tuned on `dev` only: a grid from 0.30 to 0.90 in
steps of 0.01, maximum mean F1, ties to the higher threshold. The chosen value is then set
as `EMBEDDING_THRESHOLD` in `.env` and `.env.example`, with its provenance, before the
embedding router is scored. The LLM router has no threshold. A router is scored once, on
`test`.

- `contract-assistant router-tune` writes `eval/results/embedding_threshold.json`.
- `contract-assistant router-eval --router embedding|llm [--model M] [--split dev|test]`
  scores a router (the split defaults to `test`) and writes
  `eval/results/router_<kind>_<model>_<split>.json` and the table `eval/results/router.md`.

---

## 6. End-to-end evaluation — automatic, on CUAD's test contracts (`evaluation/e2e.py`)

No human labelling and no new inference: CUAD's gold spans *are* the citations, and the
thesis test predictions of the chosen run are imported as caches (`import_harness_preds`,
parity-tested) for all 102 CUAD test contracts. The gold comes from `test.json`. The
in-scope questions of section 5 are each routed once, and the route is reused for every
contract. Out-of-scope questions are scored in the router evaluation only.

Every (test contract, in-scope question) pair is scored, not only the pairs whose gold
types occur in the contract. Asking about a clause the contract does not have is the case
the tiers exist for: 266 of the 325 validation errors are false citations.

Two display policies are scored: *found only* (τ_high) and *shown* (found and check, i.e.
τ_low). A question *asks* for the types of its gold set; the router *routes* the types it
returns. For each type t that is asked or routed, the outcome is:

| case | outcome |
|---|---|
| t routed, shown, gold present, the citation matches a gold span (CUAD's rule, Parties relaxed) | correct |
| t routed, shown, gold present, no match | wrong |
| t routed, shown, gold absent | false citation |
| t routed, not shown, gold present | omission |
| t routed, not shown, gold absent | correct abstention |
| t asked but not routed, gold present | missed by the router |

A quote withheld by the guard is not shown. The metrics:

| metric | definition |
|---|---|
| citation precision | correct / (correct + wrong + false) |
| citation recall | correct on asked types / asked types present in the contract |
| omission rate | (omissions on asked types + missed) / asked types present |
| false-citation rate | false / routed types absent from the contract |
| abstention accuracy | abstentions / routed types absent |
| displayed hallucination rate | withheld quotes / quotes the answer tried to show under the policy (shown + withheld); expected 0, measured |
| routing loss | each metric with the router minus the same metric with the oracle router |

The oracle router returns the gold types of each question, which separates routing errors
from extraction errors.

**Consistency check.** 41 one-type questions with the oracle router reproduce the thesis
operating-point counts on the test split exactly, over its 4,182 (contract, type) pairs.
At τ_low: 1,078 correct, 138 wrong, 595 false, 28 omissions and 2,343 abstentions. At
τ_high: 444 correct, 17 wrong, 143 false, 783 omissions and 2,795 abstentions. So the
evaluation scores what the thesis scored.

Output: `eval/results/e2e_oracle.json` for the oracle router, `e2e_<kind>_<model>.json` for
the others, and the table `eval/results/e2e.md` (one row per router and policy, then the
routing loss). The oracle rows come first, because they need neither Ollama nor a tuned
embedding threshold. They depend on the gold sets, so they are re-run if the review of
section 5 changes a question. The embedding and LLM rows follow that review. Latency is
reported elsewhere: extraction per document in section 2.2, routing latency (median and
p90) in the router table.

---

## 7. UI (`ui/app.py`, Gradio 6 Blocks)

```
┌────────────────────────────────────────────────────────────────────────┐
│ Contract assistant            extractor: ettin-150m-cuad · router: qwen3 │
├────────────────────────────┬───────────────────────────────────────────┤
│ [ Upload PDF / TXT ]       │ ACME_Supply_Agreement.pdf · 23 pages      │
│ Extracting… 27/41 ▓▓▓▓░░   │ ┌───────────────────────────────────────┐ │
│ (48 s)                     │ │ page 12                               │ │
│                            │ │ … This Agreement shall be governed by │ │
│ ┌────────────────────────┐ │ │ the laws of the State of ▓New York▓…  │ │
│ │ chat                   │ │ └───────────────────────────────────────┘ │
│ │ > governing law and    │ │ τ_low ─●─ 0.06  τ_high ──●─ 0.91          │
│ │   termination?         │ │ type · status · conf · page · quote       │
│ │ Governing Law — p.12,  │ │ Parties       found   0.97   1  "ACME…"   │
│ │  0.94: "…New York."    │ │ Governing Law found   0.94  12  "…law…"   │
│ │ Termination For        │ │ Expiration D. verify  0.41   3  "…yrs"    │
│ │  Convenience — not     │ │ …                                         │
│ │  found (checked)       │ │ [ Download JSON ]  [ Show sources ▾ ]     │
│ └────────────────────────┘ │                                           │
└────────────────────────────┴───────────────────────────────────────────┘
```

| component | Gradio | notes |
|---|---|---|
| upload | `gr.File` (pdf, txt) + `gr.Button("Extract")` | content hash → cache hit skips extraction; demo documents open without the extractor model, uploads then say the model is needed |
| progress | `gr.Progress` driven by the extractor's per-type callback | shows type name and elapsed time |
| chat | `gr.Chatbot` + `gr.Textbox` | messages format (no `type=` in Gradio 6); each assistant turn is the rendered template |
| document view | `gr.HighlightedText` | PDF: the citation's page. TXT: a window of 1,500 characters either side of the citation, widened to line breaks and labelled with its character range. The cited span is highlighted; never the whole contract |
| citation picker | `gr.Radio` | picks which citation of a compound answer is in view |
| clause table | `gr.Dataframe` | all types, status word on every row (found / verify / not found, styled by CSS class), best candidate, conf, page; clicking a row shows that type's top candidate |
| thresholds | two `gr.Slider`, τ_low and τ_high, and a Reset button | defaults 0.06 / 0.91; τ_low is clamped to ≤ τ_high; filters the cached n-best live, no re-run |
| sources | `gr.Accordion` | the answer JSON and the other n-best candidates |
| export | `gr.DownloadButton` | the cache JSON |
| state | `gr.State` | the loaded cache and the last answer; one document per session; switching documents resets the chat |

Gradio 6 takes `theme` and `css` in `launch()`, not in `Blocks`.

Why Gradio and not Streamlit: Streamlit re-executes the script on every interaction, which
fights a 600 MB model and a chat history; Gradio's event model keeps the model loaded once
and is the standard for ML demos. Why not FastAPI + a JS front end: a week of work for no
thesis value; the core package is UI-agnostic, so that door stays open.

---

## 8. Milestones

| # | deliverable | definition of done | est. |
|---|---|---|---|
| **M0** | repo skeleton; model on the Hub; `clause_types.json`; vendored `qa_core.py` with offsets + guard | **parity test passes**: on 2 CUAD test contracts, `(text, conf)` lists equal the training harness's byte for byte; `pytest` green | ½ day |
| **M1** | `extract` CLI → cache JSON; pages for PDFs; device auto | runs on the M4; timing table for 5 contracts of different lengths; re-upload is a cache hit | 1 day |
| **M2** | both routers; 60-question set; `router-eval` CLI | set P/R/F1 table for embedding vs LLM router; LLM fallback rate reported | 1 day |
| **M3** | answer composer + verification guard; `ask` CLI | `ask <doc> "<q>"` prints the template; every shown quote verified; unit tests for found / check / not found and for the guard rejecting a tampered quote | ½ day |
| **M4** | Gradio UI | upload → progress → chat → highlighted page → clause table with τ slider → download; a 2-minute demo script written down | 1 day |
| **M5** | end-to-end evaluation on the 102 CUAD test contracts | the application-chapter table (section 6) for both routers + oracle router; latency numbers | ½ day |
| M6 *(optional)* | LLM phrasing behind the guard; OpenAI-compatible backend; Ettin-400M as extractor | guard-violation rate measured; backend switch documented | ½ day each |

None of this waits on Kaggle. M0–M4 run on the M4 alone.

---

## 9. Risks

| risk | mitigation |
|---|---|
| `transformers>=5.16` sliding-window regression reaches this repo through a fresh install | pin `transformers>=5.13,<5.16`; the vendored guard runs at model load and aborts |
| vendored inference drifts from the harness | the M0 parity test is in CI and re-run whenever `qa_core.py` changes |
| MPS fp32 too slow for a live demo on long contracts | extraction happens before the demo; cache ships with the pre-extracted demo documents; progress bar for live uploads |
| Qwen3 router is slow or flaky | `think=False`, temperature 0, constrained schema, 10 s timeout, automatic fallback to the embedding router (which is also the honest "no LLM" baseline) |
| PDF text extraction differs from the text CUAD annotated | irrelevant for the demo (we quote what *we* extracted); for the e2e evaluation use CUAD's `.txt` so gold offsets line up |
| τ chosen badly | it comes from the thesis's val-selected value; the slider lets the user see the trade-off, and the e2e eval reports both display policies (τ_low and τ_high) |

---

## 10. Development setup

- Python 3.12, `uv` (or `pip`), `pyproject.toml` with console script `contract-assistant`.
- Dependencies: `torch`, `transformers>=5.13,<5.16`, `tokenizers<0.23`, `pdfplumber`,
  `gradio>=6.29,<7`, `pydantic>=2`, `numpy`, `sentence-transformers` (embedding router),
  `ollama>=0.5` (local LLM; 0.5 added `think`), `httpx`, `python-dotenv`. The verbatim
  guard is an exact match, so there is no fuzzy-matching dependency.
- Extras: `dev` (`pytest`, `ruff`, `fpdf2`) and `api` (`openai`, for the OpenAI-compatible
  client). Install with `uv pip install -e ".[dev]"`.
- Ollama: `brew install ollama && brew services start ollama && ollama pull qwen3:1.7b`
  (`qwen3:4b` for the comparison row). `LLM_BACKEND=none` routes with the embedding router
  only.
- Env (see `.env.example`): `EXTRACTOR_MODEL_ID`, `EXTRACTOR_NAME`, `EXTRACTOR_DEVICE` (auto),
  `TAU_LOW`, `TAU_HIGH`, `TAU_SOURCE`, `EMBEDDING_MODEL_ID`, `EMBEDDING_THRESHOLD`,
  `LLM_BACKEND` (`ollama|openai|none`), `LLM_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `CACHE_DIR`.
- `make test` (pytest incl. parity test; slow tests excluded), `make lint` (`ruff check`),
  `make demo` (`contract-assistant serve`), `make import-demo` (thesis predictions to caches),
  `make router-eval` (the LLM router on the test split) and `make e2e` (the LLM router end
  to end).
- `make ui-dev` runs the dev app with fakes and no model, with reload on save. In VS Code,
  open it with Simple Browser: Show at http://127.0.0.1:7860.

Documentation language: English in the repo, as in the training repo; the thesis text is
Serbian.

---

## 11. Decisions taken 2026-10-04

| point | decision |
|---|---|
| Hub model | **public** — `<user>/ettin-encoder-150m-cuad-qa`, uploaded from Kaggle |
| router LLM | **Qwen3-1.7B**, Ollama default quantisation (Q4_K_M, ~1.1 GB). 4B is measured in M2 only as a comparison; a hosted router (Kaggle/Colab/API) stays an option behind `LLM_BACKEND` |
| demo documents | **mandatory**, chosen from CUAD's official test split by length and clause coverage — see §12. Pre-extracted caches ship with the repo |

### Decisions taken 2026-10-16

| point | decision |
|---|---|
| extractor | **Ettin-150M, seed 3407** — best validation AUPR of the three seeds (52.52; test 53.21). Loaded from a local folder (`models/`), Hub later. Ettin-400M (test 57.31) is a configuration-only upgrade |
| answer tiers | **found** (conf ≥ τ_high = 0.91), **check** (τ_low = 0.06 ≤ conf < 0.91), **not found**. Both thresholds chosen on validation in `diplomski/src/operating_point.py`; test, once: *found* precision 0.735, everything ≥ τ_low recall 0.867, omission 0.023, verbatim 1.000 |
| why tiers | validation errors are mostly false citations of clause types the contract does not contain (266 of 325), while citations of present clauses are nearly always right (Jaccard 0.94) |
| build order | answers and guard before routers, so the core is tested without an LLM |
| end-to-end evaluation | reuses the thesis test predictions (imported as caches; parity-tested) — no new inference |

## 12. Demo documents (`demo/`)

Chosen by data: the split's short / median-ish / p90 length, maximising the number of
clause types present so every demo question has something to find. Texts are the exact
CUAD test texts (CC-BY-4.0); gold spans are kept alongside so the demo can be checked
against ground truth live.

| file | CUAD title | chars | present types |
|---|---|---|---|
| `demo/short_supply_agreement.txt` | AgapeAtpCorp … Supply Agreement (2019) | 20,693 | 19 |
| `demo/medium_sponsorship_agreement.txt` | N2K Inc … Sponsorship Agreement (1997) | 49,518 | 18 |
| `demo/long_outsourcing_agreement.txt` | Tri-City Bankshares … Outsourcing Agreement (1998) | 101,639 | 19 |

`demo/demo_documents.json` holds titles, lengths, present clause types and gold spans.
M1 produces `demo/cache/<slug>.json` for all three (≈ 40 s + 100 s + 200 s on the M4) and
they are committed, so `contract-assistant serve` opens with three ready documents. The
PDFs with real page numbers come from CUAD's full release (`CUAD_v1.zip →
full_contract_pdf/`, same titles) — an M6 nicety; the `.txt` demo works without them.

Demo script (2 minutes): open the short contract → "who are the parties and which law
governs?" (Governing Law found at 1.00, Parties to verify at 0.18) → "is there a source code
escrow?" (not found, checked, best candidate 0.00) → "give me all the important dates"
(compound routing, five quotes: three found, two to verify) → slide τ_high down and watch
rows move from verify to found in the table → switch to the long contract and ask about
source-code escrow (a rare type the model finds here, to verify at 0.33).

On the short contract, "can either side terminate for convenience?" is answered *found*
(0.97) with the six-month notice clause, which CUAD's annotators did not label as
Termination For Convenience: a false citation by the gold standard, though the clause
reads like one.
