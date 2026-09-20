"""Driver: build the corpus, degrade it to an exact CER, run a task, score the curve."""
import argparse
import json

import numpy as np
import pandas as pd

import analyze
import corpus
import degrade
import models
import tasks
from config import CER_GRID, OUT, TEST_SEEDS, TRAIN_SEED


def texts(lines_col):
    return ["\n".join(ls) for ls in lines_col]


def fit(task, name, train, **kw):
    m = models.build(name, task, **kw)
    if name == "linelen":
        return m.fit(list(train["lines"]), train["y"].to_numpy())
    return m.fit(texts(train["lines"]), train["y"].to_numpy())


def predict(name, m, frame):
    return m.predict(list(frame["lines"]) if name == "linelen" else texts(frame["lines"]))


def degraded(frame, c, seed, variant="base", p=0):
    if c == 0 and p == 0:
        return frame
    out = degrade.degrade_split(frame, c, seed, variant, p)
    return frame.drop(columns=["lines"]).merge(out[["doc_id", "lines"]], on="doc_id")


def experiment(task, name, regime="clean", cers=(0, 10), seeds=(1,), limit=None, variant="base",
               p=0):
    docs = corpus.load()
    table = tasks.TABLE[task](docs)
    if limit:
        table = pd.concat([g.sample(min(limit, len(g)), random_state=0)
                           for _, g in table.groupby("split")])
    rows = []
    folds = sorted(table["fold"].dropna().unique()) if task == "T1" else [None]
    for c in cers:
        for seed in seeds:
            for fold in folds:
                if fold is None:
                    train, test = table[table.split == "train"], table[table.split == "test"]
                else:
                    train, test = table[table.fold != fold], table[table.fold == fold]
                if task == "T2":
                    test = test[test["evaluated"]]
                tr = degraded(train, c, TRAIN_SEED, variant) if regime == "matched" else train
                m = fit(task, name, tr)
                te = degraded(test, c, seed, variant, p)
                pred = predict(name, m, te)
                rows.append(pd.DataFrame({"c": float(c), "seed": seed, "unit": te["doc_id"],
                                          "letters": te["letters"], "y": te["y"], "pred": pred}))
    return pd.concat(rows, ignore_index=True)


def search_curve(cers=(0, 10, 20, 30), seed=1, limit=None, preisigke=None):
    """T4: every search method over the degraded test corpus, as recall@20 and filter P/R/F1."""
    import search
    docs = corpus.load()
    train = docs[docs.split == "train"]
    test = docs[docs.split == "test"]
    if limit:
        test = test.sample(min(limit, len(test)), random_state=0)
    sets = search.build_queries(docs, preisigke=preisigke)
    clean = texts(test["lines"])
    queries = sorted({q for qs in sets.values() for q in qs})
    queries = [q for q in queries if any(q in u for u in clean)]
    rel = search.relevance(queries, clean)
    ranker = search.Ranker(texts(train["lines"]))
    rows = []
    for c in cers:
        # keep the degraded frame in the order relevance was indexed in
        frame = degraded(test, c, seed).set_index("doc_id").loc[test["doc_id"]]
        per = search.run_methods(queries, texts(frame["lines"]), rel, ranker=ranker)
        for m, pq in per.items():
            rows.append({"c": float(c), "method": m, **search.metrics(pq)})
    return pd.DataFrame(rows)


def curve_of(task, preds):
    per_c = {c: df for c, df in preds.groupby("c")}
    cv = analyze.curve(task, per_c)
    return cv, analyze.thresholds(cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["corpus", "experiment", "search", "queries"])
    ap.add_argument("--task", default="T3")
    ap.add_argument("--model", default="tfidf")
    ap.add_argument("--regime", default="clean", choices=["clean", "matched"])
    ap.add_argument("--cers", type=float, nargs="*", default=CER_GRID)
    ap.add_argument("--seeds", type=int, nargs="*", default=TEST_SEEDS)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--lost-lines", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.step == "corpus":
        corpus.main()
        return
    if a.step == "queries":
        import search
        queries = search.build_queries(corpus.load())
        json.dump(queries, open(OUT / "queries.json", "w"))
        print({k: len(v) for k, v in queries.items()})
        return
    if a.step == "search":
        cv = search_curve([c for c in a.cers if c <= 50], a.seeds[0], a.limit)
        cv.to_csv(OUT / "search_curve.csv", index=False)
        print(cv.to_string(index=False))
        return

    preds = experiment(a.task, a.model, a.regime, a.cers, a.seeds, a.limit, a.variant,
                       a.lost_lines)
    tag = f"{a.task}_{a.model}_{a.regime}"
    preds.to_parquet(OUT / f"{tag}.parquet")
    cv, th = curve_of(a.task, preds)
    cv.to_csv(OUT / f"{tag}_curve.csv", index=False)
    print(cv.to_string(index=False))
    print(tag, th)


if __name__ == "__main__":
    main()
