"""Extractive QA over a whole contract with explicit sliding windows -- WITH offsets.

Vendored from `diplomski/src/eval_cuad_qa.py::QAModel`, the driver that produced every
number in the thesis. Everything up to the return is kept character for character:
window construction, `token_is_max_context`, the n-best walk, dedupe to `n_best_size`
UNIQUE texts, softmax against the null candidate. The one change is the return shape --
each candidate also carries the character offsets of its (highest-scoring) occurrence,
which is what a citation needs and what the evaluation harness throws away.

`tests/test_parity.py` asserts that this module and the harness return the same
`(text, conf)` lists on real contracts; `nbest_to_candidates` is split out of `__call__`
only so that the pure-Python tail can be unit-tested without a model.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .windowing import assert_sliding_windows


@dataclass(frozen=True)
class Candidate:
    text: str      # stripped -- exactly the string the harness scores
    conf: float    # softmax probability against the null ("no clause") candidate
    start: int     # char offsets of the stripped text: context[start:end] == text
    end: int


def _strip_bounds(context: str, a: int, b: int) -> tuple[int, int]:
    """Bounds of context[a:b].strip() inside context."""
    while a < b and context[a].isspace():
        a += 1
    while b > a and context[b - 1].isspace():
        b -= 1
    return a, b


def nbest_to_candidates(
    prelim: Sequence[tuple[float, int, int]],
    null_score: float,
    context: str,
    n_best_size: int,
    top_k: int,
) -> list[Candidate]:
    """The harness's n-best tail, verbatim in behaviour.

    ORDER IS LOAD-BEARING. Mirrors transformers' compute_predictions_logits (the
    squad-style postprocessing CUAD's train.py calls): walk the candidates sorted by
    score and DEDUPE AS YOU GO, stopping once `n_best_size` unique texts are held, then
    append the null score and softmax over those. Truncating before deduping lets
    duplicates split the probability mass and deflates every confidence (measured in
    the thesis: AUPR 32.29 -> 38.68 from this alone).
    """
    if not prelim:
        return []
    ordered = sorted(prelim, key=lambda x: -x[0])

    # text -> (score, start, end) of its FIRST occurrence in score order = best-scoring
    seen: dict[str, tuple[float, int, int]] = {}
    for score, a, b in ordered:
        if len(seen) >= n_best_size:
            break
        # Strip whitespace: CUAD's get_jaccard splits on " " alone, so a trailing "\n"
        # welded to the last word turns an exact match into a miss. Normalises the
        # PREDICTION, never the metric -- and the offsets follow the stripped text.
        sa, sb = _strip_bounds(context, a, b)
        text = context[sa:sb]
        if text and text not in seen:
            seen[text] = (float(score), sa, sb)
    if not seen:
        return []

    # The null ([CLS]) span never appears in `prelim` -- CLS is not a context token --
    # so it is always appended, matching version_2_with_negative.
    texts = list(seen)
    scores = np.array([seen[t][0] for t in texts] + [null_score], dtype=np.float64)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()

    ranked = sorted(zip(texts, probs[:-1]), key=lambda x: -x[1])
    return [Candidate(t, float(p), seen[t][1], seen[t][2]) for t, p in ranked[:top_k]]


def resolve_device(spec: str = "auto"):
    """'auto' -> cuda > mps > cpu; otherwise 'cpu' | 'mps' | 'cuda' | 'cuda:N'."""
    import torch

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


class QAModel:
    """Extractive QA over arbitrarily long context, with explicit sliding windows.

    transformers 5.x dropped the `question-answering` pipeline, so windowing and
    SQuAD-2.0 n-best postprocessing are implemented here directly -- what CUAD's own
    `utils_qa.py` does, with the three settings that matter (window, stride, answer
    length) explicit rather than pipeline defaults.
    """

    def __init__(self, model_id: str, device: str = "auto", batch_size: int = 16,
                 fp32: bool | None = None, max_seq_len: int = 1024, doc_stride: int = 256):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = resolve_device(device)
        # fp16 only helps on CUDA; on CPU it is slower and less accurate, and the MPS
        # half-precision path was never validated against the thesis numbers.
        if fp32 is None:
            fp32 = self.device.type != "cuda"
        self.dtype = torch.float32 if fp32 else torch.float16
        self.batch_size = batch_size
        self.model_id = model_id

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # transformers >= 5.16 silently returns two windows per contract instead of all
        # of them; this aborts before a single wrong citation can be produced.
        assert_sliding_windows(self.tok, max_seq_len, doc_stride)
        self.model = (
            AutoModelForQuestionAnswering
            .from_pretrained(model_id, dtype=self.dtype)
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
        """-> candidates ranked best-first, empty answers dropped, offsets attached."""
        torch = self.torch
        enc = self.tok(
            question,
            context,
            truncation="only_second",        # never truncate the question
            max_length=max_seq_len,
            stride=doc_stride,               # overlap so a span split across a
            return_overflowing_tokens=True,  # window boundary is still recoverable
            return_offsets_mapping=True,     # to map tokens back to characters
            padding="max_length",
            return_tensors="pt",
        )
        n_feat = enc["input_ids"].shape[0]

        # Only context tokens (sequence_id == 1) may start or end an answer.
        offsets_per_feat = []
        for f in range(n_feat):
            seq_ids = enc.sequence_ids(f)
            offsets_per_feat.append([
                tuple(o) if seq_ids[k] == 1 else None
                for k, o in enumerate(enc["offset_mapping"][f].tolist())
            ])

        starts, ends = [], []
        with torch.no_grad():
            for i in range(0, n_feat, self.batch_size):
                out = self.model(
                    input_ids=enc["input_ids"][i:i + self.batch_size].to(self.device),
                    attention_mask=enc["attention_mask"][i:i + self.batch_size].to(self.device),
                )
                starts.append(out.start_logits.float().cpu())
                ends.append(out.end_logits.float().cpu())
        start_logits = torch.cat(starts).numpy()
        end_logits = torch.cat(ends).numpy()

        # SQuAD-2.0 null candidate: the [CLS] span, i.e. "no clause here".
        # Convention is the *minimum* null score across windows.
        null_score = float(min(
            start_logits[f][0] + end_logits[f][0] for f in range(n_feat)
        ))

        # token_is_max_context: with overlapping windows the same token appears in
        # several of them, and CUAD only lets a token START an answer in the window
        # where it sits most centrally. Scoring rule is CUAD's verbatim:
        # min(left, right) + 0.01 * window_length.
        best_feat: dict[int, tuple[float, int]] = {}
        for f in range(n_feat):
            ctx_idx = [k for k, o in enumerate(offsets_per_feat[f]) if o is not None]
            if not ctx_idx:
                continue
            first, last = ctx_idx[0], ctx_idx[-1]
            length = last - first + 1
            for k in ctx_idx:
                score = min(k - first, last - k) + 0.01 * length
                char_start = offsets_per_feat[f][k][0]
                if char_start not in best_feat or score > best_feat[char_start][0]:
                    best_feat[char_start] = (score, f)

        # Candidate spans, in the order CUAD's compute_predictions_logits produces them.
        prelim: list[tuple[float, int, int]] = []
        for f in range(n_feat):
            sl, el, offs = start_logits[f], end_logits[f], offsets_per_feat[f]
            top_s = sl.argsort()[-n_best_size:][::-1]
            top_e = el.argsort()[-n_best_size:][::-1]
            for si in top_s:
                if offs[si] is None or best_feat.get(offs[si][0], (0, -1))[1] != f:
                    continue
                for ei in top_e:
                    if offs[ei] is None or ei < si or ei - si + 1 > max_answer_len:
                        continue
                    prelim.append((float(sl[si] + el[ei]), offs[si][0], offs[ei][1]))

        return nbest_to_candidates(prelim, null_score, context, n_best_size, top_k)
