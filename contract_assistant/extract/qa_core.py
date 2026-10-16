"""Sliding-window extractive QA over a whole contract, returning character offsets.

Same windowing and n-best postprocessing as the thesis evaluation driver
(`diplomski/src/eval_cuad_qa.py`); `tests/test_parity.py` checks that both agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .windowing import assert_sliding_windows


@dataclass(frozen=True)
class Candidate:
    text: str
    conf: float
    start: int
    end: int


def _strip_bounds(context: str, start: int, end: int) -> tuple[int, int]:
    """Bounds of `context[start:end].strip()` inside `context`."""
    while start < end and context[start].isspace():
        start += 1
    while end > start and context[end - 1].isspace():
        end -= 1
    return start, end


def nbest_to_candidates(
    prelim: Sequence[tuple[float, int, int]],
    null_score: float,
    context: str,
    n_best_size: int,
    top_k: int,
) -> list[Candidate]:
    """Deduplicate scored spans, then softmax them against the null score."""
    if not prelim:
        return []

    # Dedupe while walking in score order, as transformers' compute_predictions_logits
    # does; truncating first lets duplicates split the probability mass.
    best: dict[str, tuple[float, int, int]] = {}
    for score, start, end in sorted(prelim, key=lambda x: -x[0]):
        if len(best) >= n_best_size:
            break
        start, end = _strip_bounds(context, start, end)
        text = context[start:end]
        if text and text not in best:
            best[text] = (float(score), start, end)
    if not best:
        return []

    texts = list(best)
    scores = np.array([best[t][0] for t in texts] + [null_score], dtype=np.float64)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()

    ranked = sorted(zip(texts, probs[:-1]), key=lambda x: -x[1])
    return [Candidate(t, float(p), best[t][1], best[t][2]) for t, p in ranked[:top_k]]


def resolve_device(spec: str = "auto"):
    """Map 'auto' to cuda, then mps, then cpu; pass any other torch device string through."""
    import torch

    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QAModel:
    """Fine-tuned span extractor applied to arbitrarily long text."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        batch_size: int = 16,
        fp32: bool | None = None,
        max_seq_len: int = 1024,
        doc_stride: int = 256,
    ):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = resolve_device(device)
        # Half precision only on CUDA; it was never validated on MPS or CPU.
        if fp32 is None:
            fp32 = self.device.type != "cuda"
        self.dtype = torch.float32 if fp32 else torch.float16
        self.batch_size = batch_size
        self.model_id = model_id

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        assert_sliding_windows(self.tok, max_seq_len, doc_stride)
        self.model = (
            AutoModelForQuestionAnswering.from_pretrained(model_id, dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    def __call__(
        self,
        question: str,
        context: str,
        *,
        max_seq_len: int = 1024,
        doc_stride: int = 256,
        max_answer_len: int = 512,
        top_k: int = 20,
        n_best_size: int = 20,
    ) -> list[Candidate]:
        """Ranked answer candidates for one question over the whole context."""
        torch = self.torch
        enc = self.tok(
            question,
            context,
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        n_windows = enc["input_ids"].shape[0]

        # Only context tokens may start or end an answer.
        offsets = []
        for w in range(n_windows):
            seq_ids = enc.sequence_ids(w)
            offsets.append(
                [
                    tuple(o) if seq_ids[k] == 1 else None
                    for k, o in enumerate(enc["offset_mapping"][w].tolist())
                ]
            )

        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, n_windows, self.batch_size):
                out = self.model(
                    input_ids=enc["input_ids"][i : i + self.batch_size].to(self.device),
                    attention_mask=enc["attention_mask"][i : i + self.batch_size].to(
                        self.device
                    ),
                )
                starts.append(out.start_logits.float().cpu())
                ends.append(out.end_logits.float().cpu())
        start_logits = torch.cat(starts).numpy()
        end_logits = torch.cat(ends).numpy()

        null_score = float(min(start_logits[w][0] + end_logits[w][0] for w in range(n_windows)))

        # A token may start an answer only in the window where it is most central
        # (CUAD's rule: min(left, right) + 0.01 * window length).
        home_window: dict[int, tuple[float, int]] = {}
        for w in range(n_windows):
            ctx = [k for k, o in enumerate(offsets[w]) if o is not None]
            if not ctx:
                continue
            first, last = ctx[0], ctx[-1]
            length = last - first + 1
            for k in ctx:
                centrality = min(k - first, last - k) + 0.01 * length
                char_start = offsets[w][k][0]
                if char_start not in home_window or centrality > home_window[char_start][0]:
                    home_window[char_start] = (centrality, w)

        prelim: list[tuple[float, int, int]] = []
        for w in range(n_windows):
            sl, el, offs = start_logits[w], end_logits[w], offsets[w]
            top_starts = sl.argsort()[-n_best_size:][::-1]
            top_ends = el.argsort()[-n_best_size:][::-1]
            for si in top_starts:
                if offs[si] is None or home_window.get(offs[si][0], (0, -1))[1] != w:
                    continue
                for ei in top_ends:
                    if offs[ei] is None or ei < si or ei - si + 1 > max_answer_len:
                        continue
                    prelim.append((float(sl[si] + el[ei]), offs[si][0], offs[ei][1]))

        return nbest_to_candidates(prelim, null_score, context, n_best_size, top_k)
