# Contract assistant — plan

**Date:** 2026-10-04
**Status:** M0 done; build order in [IMPLEMENTATION.md](IMPLEMENTATION.md) (v2, 2026-10-16)
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
         └─> CACHE lookup, threshold τ
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
| answer text | **template by default** (`Governing Law — p. 12, 0.94: "…"`), optional LLM phrasing behind a verbatim-quote guard | the template is auditable; LLM phrasing adds nothing to correctness |
| UI | **Gradio 5 `Blocks`** over a UI-independent core package | fastest Python route to upload + chat + highlighted citations; the core stays testable and could sit behind FastAPI later |
| runs on | the M4 (MPS fp32 for Ettin, Ollama for Qwen3) | no GPU dependency; measured ~2 s per 1,000 characters of contract for the 150M model |

---

## 1. Interfaces with the training repo — fixed up front

These are the only things that cross the repo boundary. Everything else is independent.

| artefact | from `diplomski` | into this repo | how |
|---|---|---|---|
| **model weights** | `runs/<RUN_NAME>/final/` (575 MB) | `EXTRACTOR_MODEL_ID` env / config | **public** HuggingFace Hub repo `<user>/ettin-encoder-150m-cuad-qa` (MIT weights, CC-BY-4.0 data; public makes the thesis reproducible from a model id). Uploaded straight from the Kaggle notebook — the files never need to touch the laptop |
| **operating threshold τ** | `results/<tag>_proposal_metrics.json` → `tau_val_selected` | `cache.tau_default`, slider default in the UI | copied as a number with its provenance string |
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
    pdf.py               pdfplumber -> text, page_starts; page_of(offset); txt passthrough
    extract/
      qa_core.py         vendored QAModel + n-best WITH offsets; device auto (cuda > mps > cpu)
      windowing.py       vendored sliding-window guard
      extractor.py       run 41 questions over a document -> ClauseHit[]; progress callback
    store.py             cache/<sha256>.json read/write; doc index; fingerprint = sha256(text)
    router/
      base.py            Router protocol: route(question) -> RouteResult
      embedding.py       bge-base: question vs 41 descriptions, cosine, threshold + top-k
      llm.py             LLMRouter(client): JSON-schema-constrained {"clause_types": [...]}
      clients.py         OllamaClient (native, format=schema, think=False) | OpenAICompatClient
    answer.py            compose(cache, route, τ) -> Answer; verify quotes; template rendering
    ui/app.py            Gradio Blocks
    cli.py               extract | ask | router-eval | e2e-eval | serve
  data/
    clause_types.json    41 entries, generated from the training repo
    router_eval.jsonl    60 questions with gold clause-type sets (section 5)
  eval/
    router_eval.py       set P/R/F1 per router
    e2e_eval.py          citation accuracy on CUAD test contracts, automatic (section 6)
  tests/
  cache/                 (gitignored) one JSON per processed document
  pyproject.toml  README.md  PLAN.md
```

### 2.1 Data shapes

**Cache, one file per document** (`cache/<sha256>.json`):

```json
{
  "schema_version": 1,
  "doc_id": "sha256:…",
  "source": {"filename": "ACME_Supply_Agreement.pdf", "kind": "pdf", "n_chars": 46848, "n_pages": 23},
  "model": {"id": "<user>/ettin-encoder-150m-cuad-qa", "max_seq_len": 1024, "doc_stride": 256,
            "top_k": 20, "max_answer_len": 512, "device": "mps"},
  "tau_default": 0.42,
  "page_starts": [0, 2103, 4311],
  "clauses": {
    "Governing Law": [{"text": "This Agreement shall be governed by the laws of the State of New York.",
                       "start": 48210, "end": 48290, "page": 12, "conf": 0.94}],
    "Non-Compete": []
  },
  "timing": {"extract_s": 51.3}
}
```

`clauses[type]` keeps the full n-best (up to 20 candidates with their confidences), not
just the top one, so the τ slider in the UI works without re-running the model.

**Answer** (what the UI renders and what the evaluations score):

```json
{
  "question": "which law governs, and can they terminate without cause?",
  "router": {"kind": "llm", "clause_types": ["Governing Law", "Termination For Convenience"], "latency_ms": 1400},
  "tau": 0.42,
  "items": [
    {"clause_type": "Governing Law", "status": "present",
     "citations": [{"quote": "…", "page": 12, "conf": 0.94, "start": 48210, "end": 48290}]},
    {"clause_type": "Termination For Convenience", "status": "absent", "best_conf": 0.07, "citations": []}
  ],
  "verified": true
}
```

`status` ∈ `present` (conf ≥ τ), `absent` (best candidate < τ), `low_confidence` (within a
band under τ, shown greyed with its number so the user can judge).

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

### 2.3 Pages (`pdf.py`)

`page_starts[i]` = cumulative character offset where page *i* begins in the extracted
text; `page_of(offset) = bisect_right(page_starts, offset)`. Because the model runs on
the *same* extracted text, offsets map to pages exactly — no alignment problem. For
`.txt` inputs (CUAD's own files) pages are `n/a`; CUAD ships `full_contract_pdf/`, so the
demo can use real PDFs.

---

## 3. Router

Input: a free-form question. Output: a **set** of clause types from the closed list of
41, possibly empty. Two implementations behind one interface, both evaluated.

**`EmbeddingRouter` (baseline, no LLM).** `bge-base-en-v1.5` with its query instruction
prefix; cosine similarity between the question and each clause type's description (CUAD's
own "Details: …" text plus a few hand-written paraphrases per type); return every type
above a similarity threshold, at most *k*. Deterministic, ~10 ms, no dependency. Expected
to handle single-intent questions and fail on compound/abstract ones ("the important
dates").

**`LLMRouter` (default).** Qwen3-1.7B or -4B through Ollama with `format=` the JSON schema
of `{"clause_types": [<enum of 41>]}` — constrained decoding, so an invalid label cannot
be emitted — `think=False` for latency, temperature 0. Prompt = the 41 names with
one-line descriptions, three worked examples (single, compound, out-of-scope), the
question. Pydantic validates the result; on any failure fall back to the embedding
router and mark `router.kind = "embedding (fallback)"`.

**Closed-world behaviour.** Questions outside the 41 types ("what about data privacy?")
route to the empty set and the assistant says so, listing what it *can* answer. This is
a feature: the system never pretends.

**Backends.** `clients.py` has `OllamaClient` (native library) and `OpenAICompatClient`
(any OpenAI-compatible endpoint, including a hosted model behind an API key; `base_url`,
`model`, `api_key` from env). Switching is `LLM_BACKEND=ollama|openai`. The demo runs
`ollama`; the API path exists for the comparison row and for anyone without a capable
laptop.

---

## 4. Answer composition and the guard (`answer.py`)

1. For each routed type, take the cache's candidates; `present` if the best conf ≥ τ.
2. **Verify** every quote: `quote` must occur in the document text (whitespace-normalised
   comparison; the extractor's offsets make this exact in practice). A quote that fails is
   dropped and counted — this counter is the *displayed hallucination rate* reported in
   the thesis. Expected 0; measured anyway.
3. Render the template:

```
Governing Law — p. 12, confidence 0.94
  "This Agreement shall be governed by the laws of the State of New York."
Termination For Convenience — not found (checked; best candidate 0.07)
```

4. Optional (`ANSWER_STYLE=llm`): ask the LLM to phrase one paragraph **using only the
   quotes above, verbatim**. Post-check: every quoted substring in its output must be one
   of the verified quotes, else discard and show the template. Off by default; it exists
   to measure how often a small LLM breaks the quote rule (a thesis number), not to
   improve answers.

---

## 5. Router evaluation set (`data/router_eval.jsonl`)

60 questions, hand-written, each with its gold set of clause types. Mix: ~30 single-type,
~20 compound, ~10 out-of-scope or abstract. Scored with set precision / recall / F1 per
question, averaged; both routers; also the fallback rate of the LLM router. First dozen,
to fix the style:

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

The full 60 are written in M2 and reviewed by the author before any router is tuned on
them. They are never used to tune the extractor.

---

## 6. End-to-end evaluation — automatic, on CUAD's test contracts (`eval/e2e_eval.py`)

No human labelling needed: CUAD's gold spans *are* the citations. For each of the 102
official test contracts and each router-eval question whose gold types are present in
that contract:

- route → cache lookup at τ → citations
- **citation accuracy**: for each routed type that is gold-present, does the top citation
  match a gold span (Jaccard ≥ 0.5, CUAD's rule)?
- **false citation rate**: for each routed type that is gold-absent, did we cite anyway?
- **abstention accuracy**: gold-absent → `absent` status?
- **router error contribution**: the same metrics with an oracle router, so routing and
  extraction errors are separated
- **displayed hallucination rate**: verified-quote failures / quotes shown (expect 0)
- **latency**: upload-time extraction per document (M4), answer latency with each router

Output: one table for the application chapter, per router, with the oracle-router row as
the ceiling. This reuses `test.json` and the clause-type match rule from the training
repo; it needs the cache for the 102 contracts, which is one overnight run on the M4
(~1.5 h at ~50 s/contract) or 20 min on Kaggle.

---

## 7. UI (`ui/app.py`, Gradio 5 Blocks)

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
│ │ > governing law and    │ │ Clauses found      τ ──●──── 0.42         │
│ │   termination?         │ │ type · conf · page · quote                │
│ │ Governing Law — p.12,  │ │ Parties        0.97   1  "ACME Corp…"     │
│ │  0.94: "…New York."    │ │ Governing Law  0.94  12  "…governed by…"  │
│ │ Termination For        │ │ Expiration D.  0.91   3  "…five (5) yrs"  │
│ │  Convenience — not     │ │ …                                         │
│ │  found (checked)       │ │ [ Download JSON ]  [ Show sources ▾ ]     │
│ └────────────────────────┘ │                                           │
└────────────────────────────┴───────────────────────────────────────────┘
```

| component | Gradio | notes |
|---|---|---|
| upload | `gr.File` (pdf, txt) + `gr.Button("Extract")` | content hash → cache hit skips extraction |
| progress | `gr.Progress` driven by the extractor's per-type callback | shows type name and elapsed time |
| chat | `gr.Chatbot(type="messages")` + `gr.Textbox` | each assistant turn is the rendered template; a "sources" accordion holds the raw `Answer` JSON |
| document view | `gr.HighlightedText` | renders the **page** containing the last answer's citations with spans coloured by clause type — never the whole 300k-char contract at once |
| clause table | `gr.Dataframe` | all types, best candidate, conf, page; filtered live by the τ `gr.Slider` (cache holds the full n-best, no re-run) |
| export | `gr.DownloadButton` | the cache JSON |
| state | `gr.State(doc_id)` | one document per session; switching documents resets the chat |

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
| **M3** | answer composer + verification guard; `ask` CLI | `ask <doc> "<q>"` prints the template; every shown quote verified; unit tests for present/absent/low-confidence and for the guard rejecting a tampered quote | ½ day |
| **M4** | Gradio UI | upload → progress → chat → highlighted page → clause table with τ slider → download; a 2-minute demo script written down | 1 day |
| **M5** | end-to-end evaluation on the 102 CUAD test contracts | the application-chapter table (section 6) for both routers + oracle router; latency numbers | ½ day + one overnight extraction run |
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
| τ chosen badly | it comes from the thesis's val-selected value; the slider lets the user see the trade-off, and the e2e eval reports accuracy at three τ values |

---

## 10. Development setup

- Python 3.12, `uv` (or `pip`), `pyproject.toml` with console script `contract-assistant`.
- Dependencies: `torch`, `transformers>=5.13,<5.16`, `tokenizers<0.23`, `pdfplumber`,
  `gradio>=5`, `pydantic>=2`, `numpy`, `sentence-transformers` (embedding router),
  `ollama` (local LLM), `openai` (compat client, optional), `rapidfuzz` (whitespace-tolerant
  quote check), `pytest`, `ruff`.
- Ollama: `brew install ollama && ollama pull qwen3:1.7b` (or `qwen3:4b`).
- Env: `EXTRACTOR_MODEL_ID`, `EXTRACTOR_DEVICE` (auto), `TAU` (default from the thesis),
  `LLM_BACKEND=ollama|openai|none`, `LLM_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `ANSWER_STYLE=template|llm`, `CACHE_DIR`.
- `make test` (pytest incl. parity test), `make demo` (`contract-assistant serve`),
  `make e2e` (section 6).

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
M1 produces `cache/demo/<slug>.json` for all three (≈ 40 s + 100 s + 200 s on the M4) and
they are committed, so `contract-assistant serve` opens with three ready documents. The
PDFs with real page numbers come from CUAD's full release (`CUAD_v1.zip →
full_contract_pdf/`, same titles) — an M6 nicety; the `.txt` demo works without them.

Demo script (2 minutes): open the short contract → "who are the parties and which law
governs?" (two high-confidence quotes with pages) → "can either side terminate for
convenience?" (absent, checked, best candidate shown) → "give me all the important dates"
(compound routing, 4–5 quotes) → slide τ down and watch low-confidence rows appear in the
table → switch to the long contract and ask about source-code escrow (a rare type the
model finds here).
