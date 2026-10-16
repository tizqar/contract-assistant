# contract-assistant

A local legal assistant for one contract at a time: upload a contract, ask questions in
plain language, get **verbatim quotes with a page number and a confidence** — or an
explicit *"not found"*.

The expensive model (a fine-tuned 150M-parameter encoder, trained in the companion
`diplomski` repo) runs **once at upload** and extracts all 41 CUAD clause types with
character offsets. Every later question is a cache lookup: a small local LLM (Qwen3 via
Ollama) only maps the question to clause types; it never writes legal content. Quotes
are verified against the source text before they are shown.

**Status:** M0 done (skeleton, vendored inference core with offsets, parity test, demo documents). Build order: [IMPLEMENTATION.md](IMPLEMENTATION.md).

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
| fine-tuned extractor (HF Hub id) | `EXTRACTOR_MODEL_ID` |
| operating threshold τ (val-selected) | `TAU` |
| 41 clause types + CUAD's question wording | `data/clause_types.json` |
| inference core + sliding-window guard | vendored, with a parity test |

Training, evaluation against published CUAD numbers, and the thesis figures stay in
`diplomski`.
