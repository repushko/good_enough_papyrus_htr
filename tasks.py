"""Task tables and metrics: T1 document form, T2 dating, T3 documentary vs literary."""
import numpy as np
from sklearn.metrics import f1_score

from config import T2_MAX_WIDTH
from corpus import GAP


def text_of(lines):
    return "\n".join(lines)


def t1(docs):
    d = docs[docs["gram_type"].notna() & (docs["letters"] > 0)].copy()
    d["y"] = d["gram_type"]
    return d


def t2(docs):
    d = docs[docs["date_mid"].notna() & (docs["letters"] > 0)].copy()
    d["y"] = d["date_mid"].astype(float)
    d["evaluated"] = d["date_width"] <= T2_MAX_WIDTH
    return d


def t3(docs):
    d = docs[(~docs["tm_in_both"]) & (docs["letters"] > 0)].copy()
    d["y"] = d["source"]
    return d


TABLE = {"T1": t1, "T2": t2, "T3": t3}
METRIC = {"T1": "macro_f1", "T2": "mae", "T3": "macro_f1"}


def macro_f1(y, pred, w=None):
    return float(f1_score(y, pred, average="macro", zero_division=0, sample_weight=w))


def mae(y, pred, w=None):
    err = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    return float(np.average(err, weights=w))


def score(task, y, pred, w=None):
    return mae(y, pred, w) if task == "T2" else macro_f1(y, pred, w)


def baseline(task, train_y, test_y):
    if task == "T2":
        return mae(test_y, np.full(len(test_y), np.median(train_y)))
    values, counts = np.unique(train_y, return_counts=True)
    return macro_f1(test_y, np.full(len(test_y), values[counts.argmax()], dtype=object))


def line_features(lines):
    """The layout-only view of a document: statistics of its line lengths."""
    lens = np.array([sum(1 for c in ln if c not in (GAP,)) for ln in lines] or [0], float)
    return [len(lens), lens.mean(), lens.std(), lens.min(), lens.max(),
            np.median(lens), np.percentile(lens, 25), np.percentile(lens, 75)]
