"""The words the web app shows, in plain language for readers who are not lawyers or developers.

Every template reaches these as `copy`. The technical vocabulary (clause types, confidence,
thresholds, model names) stays out of the main screens and lives under Advanced and in the
help popover.
"""

from __future__ import annotations

# Status badges. "confirmed" and "rejected" are the reader's decisions, see decisions.py.
STATUS = {
    "found": "Found",
    "check": "Check this",
    "not_found": "Not found",
    "confirmed": "Confirmed by you",
    "rejected": "Rejected by you",
    "definition": "Defined term",
    "related": "Closest passage",
}
STATUS_HELP = {
    "found": "The assistant is confident this is the passage for this clause.",
    "check": "A likely passage. Read it, then confirm it or reject it.",
    "not_found": "The contract was checked and no passage for this clause was found.",
    "confirmed": "You confirmed this passage; it now counts as found.",
    "rejected": "You rejected this passage; the next candidate, if any, is shown.",
    "definition": "The sentence of the contract that defines the term, copied word for word.",
    "related": "A passage close to your question; it is not one of the clauses checked.",
}

AI_BADGE = "AI"
AI_NOTICE = (
    "AI text can be wrong. Quotes are copied word for word from the contract; "
    "check everything else against it."
)
AI_WITHHELD = "The assistant's text was held back because {reason}."

CONFIRM = "Confirm"
REJECT = "Not this"
UNDO = "Undo"
SHOW_IN_DOCUMENT = "Show in document"
EXPLAIN = "Explain"
EXPLAIN_SR = "Explain in Serbian"
COPY_QUOTE = "Copy quote"
STRONG_MATCH = "strong match"
POSSIBLE_MATCH = "possible match"
PAGE = "Page {page}"

# The sections of the overview, in order: id, title, one line under the title.
SECTIONS = (
    ("summary", "Summary", "What the contract says, written from the clauses found."),
    (
        "facts",
        "Key facts",
        "Parties, dates, term, notice and governing law, each copied from the contract.",
    ),
    ("checks", "Things to check", "Points a reader usually looks at, from a fixed rule list."),
    ("duties", "Who must do what", "Obligations read from the clauses, in the contract's words."),
    ("clauses", "All clauses", "Every clause the assistant looks for, with what it found."),
    ("terms", "Defined terms", "Terms the contract defines, with the defining sentence."),
)

OVERVIEW = "Overview"
DOCUMENT = "Document"
CONTRACTS = "Contracts"
ADD_CONTRACT = "Add a contract"
SAMPLES = "Or try a sample contract"
ANALYSING = "Reading the contract"
AI_READY = "Local AI ready"
AI_OFF = "AI off"
REPORT = "Report"
HELP_TITLE = "How to read this"
HELP = (
    (
        "What the assistant did",
        (
            "It read the contract once and looked for the clauses a lawyer checks first: who "
            "the parties are, how long it lasts, how it ends, who is liable, and about forty "
            "more."
        ),
    ),
    (
        "Found, Check this, Not found",
        (
            "Found means the assistant is confident about the passage. Check this means it "
            "found a likely passage but is not sure; read it, then press Confirm or Not this. "
            "Not found means the contract was checked and nothing matched."
        ),
    ),
    (
        "Quotes and AI text",
        (
            "Everything in a quote block is copied word for word from the contract. Text "
            "marked AI was written by a language model on this computer from those quotes and "
            "checked against them; it can still be wrong."
        ),
    ),
    (
        "Nothing leaves this computer",
        "The contract, the analysis and your decisions stay in the cache folder of this app.",
    ),
)
WHAT_HAPPENED = (
    "The assistant checked this contract for {total} clauses: {found} found, "
    "{check} to check, {not_found} not found."
)
