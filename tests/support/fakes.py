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
