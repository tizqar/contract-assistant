import re
import time
import zlib

import numpy as np

from contract_assistant.clause_types import app_clause_types
from contract_assistant.config import load_settings
from contract_assistant.extract.qa_core import Candidate
from contract_assistant.router.base import RouteResult


class FakeQA:
    """QA function returning fixed answers located with str.find; records its calls."""

    def __init__(self, answers: dict[str, list[tuple[str, float]]] | None = None):
        self.answers = answers or {}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, question: str, context: str, **kwargs) -> list[Candidate]:
        self.calls.append((question, kwargs))
        candidates = []
        for text, conf in self.answers.get(question, []):
            start = context.find(text)
            if start >= 0:
                candidates.append(Candidate(text, conf, start, start + len(text)))
        return candidates


FAKE_DIMENSIONS = 256


class FakeEncoder:
    """Bag-of-words hash encoder; records the texts given to each encode method."""

    def __init__(self):
        self.query_texts: list[str] = []
        self.document_texts: list[str] = []

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self.query_texts.extend(texts)
        return self._encode(texts)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_texts.extend(texts)
        return self._encode(texts)

    @staticmethod
    def _encode(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), FAKE_DIMENSIONS))
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, zlib.crc32(word.encode()) % FAKE_DIMENSIONS] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


ROUTE_KEYWORDS: dict[str, list[str]] = {
    "parties": ["Parties"],
    "law": ["Governing Law"],
    "governs": ["Governing Law"],
    "terminat": ["Termination For Convenience"],
    "dates": ["Agreement Date", "Effective Date", "Expiration Date", "Renewal Term"],
    "escrow": ["Source Code Escrow"],
    "exclusive": ["Exclusivity"],
    "payment": ["Payment Terms"],
    "deliver": ["Delivery"],
    "confidential": ["Confidentiality"],
    "indemn": ["Indemnification"],
    "dispute": ["Dispute Resolution"],
    "arbitration": ["Dispute Resolution"],
    "force majeure": ["Force Majeure"],
    "personal data": ["Data Protection"],
}
SLOW_QA_DELAY_S = 0.1
LAW_CONFIDENCE = 0.95
LAW_SENTENCE = re.compile(r"[^.\n]*\blaws?\b[^.\n]*\.")


class FakeRouter:
    """Router that matches clause type names and a few keywords in the question."""

    kind = "explicit"

    def __init__(self):
        self.names = [c.name for c in app_clause_types(load_settings())]

    def route(self, question: str) -> RouteResult:
        started = time.perf_counter()
        lowered = question.lower()
        first_seen: dict[str, int] = {}
        for name in self.names:
            self._note(first_seen, name, lowered.find(name.lower()))
        for keyword, names in ROUTE_KEYWORDS.items():
            for name in names:
                self._note(first_seen, name, lowered.find(keyword))
        ordered = sorted(first_seen, key=lambda name: first_seen[name])
        latency_ms = (time.perf_counter() - started) * 1000
        return RouteResult(clause_types=ordered, kind="explicit", latency_ms=latency_ms)

    @staticmethod
    def _note(first_seen: dict[str, int], name: str, position: int) -> None:
        if position >= 0:
            first_seen[name] = min(position, first_seen.get(name, position))


class FixedRouter:
    """Router returning a prepared route per question; records the questions it was asked."""

    kind = "explicit"

    def __init__(self, routes: dict[str, RouteResult]):
        self.routes = routes
        self.questions: list[str] = []

    def route(self, question: str) -> RouteResult:
        self.questions.append(question)
        return self.routes[question]


class FakeLLMClient:
    """LLM client returning queued replies or raising queued errors; records its calls."""

    def __init__(self, replies: list[str | Exception], model: str = "fake-llm"):
        self.model = model
        self.replies = list(replies)
        self.calls: list[tuple[str, str, dict]] = []
        self.text_calls: list[tuple[str, str, int]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        self.calls.append((system, user, schema))
        return self._reply()

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        self.text_calls.append((system, user, max_tokens))
        return self._reply()

    def _reply(self) -> str:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


QUOTED = re.compile(r'"([^"]+)"')
ECHO_QUOTES = 3


class EchoTextClient:
    """Text client that quotes the first quotes of the request back; faithful by construction."""

    model = "fake-gen"

    def __init__(self):
        self.text_calls: list[tuple[str, str, int]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        return "{}"

    def complete_text(self, system: str, user: str, max_tokens: int) -> str:
        self.text_calls.append((system, user, max_tokens))
        quotes = QUOTED.findall(user)[:ECHO_QUOTES]
        if not quotes:
            return "The request names no clause."
        return " ".join(f'The contract states "{quote}".' for quote in quotes)


class _SlowQA(FakeQA):
    def __init__(self, delay_s: float, law_question: str):
        super().__init__()
        self.delay_s = delay_s
        self.law_question = law_question

    def __call__(self, question: str, context: str, **kwargs) -> list[Candidate]:
        time.sleep(self.delay_s)
        candidates = super().__call__(question, context, **kwargs)
        match = LAW_SENTENCE.search(context)
        if question == self.law_question and match:
            sentence = match.group()
            start = match.start() + len(sentence) - len(sentence.lstrip())
            sentence = sentence.strip()
            candidates.append(Candidate(sentence, LAW_CONFIDENCE, start, start + len(sentence)))
        return candidates


def slow_fake_qa(delay_s: float = SLOW_QA_DELAY_S) -> FakeQA:
    """Return a FakeQA that sleeps per call and cites the first sentence on law."""
    clause_types = app_clause_types(load_settings())
    law = next(c for c in clause_types if c.name == "Governing Law")
    return _SlowQA(delay_s, law.cuad_question)
