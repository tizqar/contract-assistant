import re
import zlib

import numpy as np

from contract_assistant.extract.qa_core import Candidate


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
