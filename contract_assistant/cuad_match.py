"""CUAD's span match rule, ported from diplomski/src/cuad_metrics.py."""

IOU_THRESH = 0.5
RELAXED_CLAUSE_TYPES = ("Parties",)


def get_jaccard(gt: str, pred: str) -> float:
    """Word-set Jaccard overlap of two spans, as in CUAD's evaluate.py."""
    remove_tokens = [".", ",", ";", ":"]
    for token in remove_tokens:
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower()
    pred = pred.lower()
    gt = gt.replace("/", " ")
    pred = pred.replace("/", " ")

    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))

    intersection = gt_words.intersection(pred_words)
    union = gt_words.union(pred_words)
    return len(intersection) / len(union)


def is_match(gold: str, pred: str, relaxed: bool = False) -> bool:
    """Whether pred matches gold; relaxed also accepts gold as a substring of pred."""
    if get_jaccard(gold, pred) >= IOU_THRESH:
        return True
    return relaxed and gold in pred
