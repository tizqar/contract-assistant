"""Guard against tokenizer sliding-window regressions.

Both the feature builder and the evaluator read whole contracts through

    tok(question, context, truncation="only_second", max_length=L, stride=S,
        return_overflowing_tokens=True, return_offsets_mapping=True)

and rely on it returning EVERY window over the context. That contract broke silently.

MEASURED (2026-09-25), jhu-clsp/ettin-encoder-150m, 14,049-token context, L=1024, S=256:

    transformers 5.13.0  PreTrainedTokenizerFast  19 windows, covers 100% of the text
    transformers 5.17.0  TokenizersBackend         2 windows, covers   7.5% of the text

5.17.0's new `TokenizersBackend` returns the first window plus ONE short remainder and
drops everything after it. Nothing raises. Downstream, the builder emitted 8x fewer
features and only clause types that live on a contract's first page, and the evaluator
would have read the first ~1,600 tokens of each contract and scored the model as garbage
in a way indistinguishable from a bad model. The Colab notebook's `transformers>=5`
pulled 5.17.0 the day it came out; the local venv still had 5.13.0.

`assert_sliding_windows` costs ~100 ms and is called before any real work in both
drivers. It does not care WHICH version is installed, only whether the behaviour holds,
so it survives both future fixes and future regressions.
"""
from __future__ import annotations


def assert_sliding_windows(tok, max_seq_len: int, doc_stride: int) -> int:
    """Raise SystemExit unless overflow windowing covers a long synthetic context.

    Returns the number of windows produced, for logging.
    """
    sentence = "This Agreement shall be governed by the laws of the State of New York. "
    # Long enough for >= 3 windows at any window size up to 8192.
    ctx = (sentence * 12 + "\n") * max(120, (3 * max_seq_len) // 150)

    enc = tok(
        "Which law governs?",
        ctx,
        truncation="only_second",
        max_length=max_seq_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    n = len(enc["input_ids"])
    seq_ids = enc.sequence_ids(n - 1)
    ctx_idx = [k for k, s in enumerate(seq_ids) if s == 1]
    last_char = enc["offset_mapping"][n - 1][ctx_idx[-1]][1] if ctx_idx else 0
    coverage = last_char / len(ctx)

    if n < 3 or coverage < 0.98:
        import transformers

        raise SystemExit(
            "\nTOKENIZER SLIDING-WINDOW CHECK FAILED\n"
            f"  transformers {transformers.__version__}, tokenizer class "
            f"{type(tok).__name__}\n"
            f"  {n} window(s) over a {len(ctx):,}-char context at max_length="
            f"{max_seq_len} / stride={doc_stride}; the last window ends at "
            f"{100 * coverage:.1f}% of the text.\n"
            "  `return_overflowing_tokens` is not producing sliding windows, so every "
            "contract would be silently cut to its first window or two.\n"
            "  Known bad: transformers 5.17.0 (TokenizersBackend). Known good: 5.13.0.\n"
            "  Fix:  pip install 'transformers>=5,<5.17'  then restart the runtime.\n"
            "  See src/windowing.py."
        )
    return n
