"""Paired bootstrap over units and seeds; retention and the c95 / c90 thresholds."""
import numpy as np
import pandas as pd

from config import CI, LETTER_BINS, MIN_BIN, N_BOOT, SEED
from tasks import score

LOWER_IS_BETTER = {"mae"}


def draws(n_units, n_seeds, n_boot=N_BOOT, seed=SEED):
    r = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield (np.bincount(r.integers(n_units, size=n_units), minlength=n_units),
               np.bincount(r.integers(n_seeds, size=n_seeds), minlength=n_seeds))


def curve(task, preds, n_boot=N_BOOT):
    """preds: {cer: DataFrame(unit, seed, y, pred)} -> point, interval and paired retention."""
    cers = sorted(preds)
    units = sorted({u for df in preds.values() for u in df["unit"]})
    seeds = sorted({s for df in preds.values() for s in df["seed"]})
    ui = {u: i for i, u in enumerate(units)}
    si = {s: i for i, s in enumerate(seeds)}
    coded = {c: (df["unit"].map(ui).to_numpy(), df["seed"].map(si).to_numpy(), df)
             for c, df in preds.items()}
    point = {c: score(task, df["y"], df["pred"]) for c, (_, _, df) in coded.items()}
    reps = {c: [] for c in cers}
    for uc, sc in draws(len(units), len(seeds), n_boot):
        for c, (u, s, df) in coded.items():
            w = uc[u] * sc[s]
            reps[c].append(score(task, df["y"], df["pred"], w) if w.sum() else np.nan)
    lo_q, hi_q = (1 - CI) / 2, 1 - (1 - CI) / 2
    rows = []
    base = np.array(reps[0.0] if 0.0 in reps else reps[cers[0]], float)
    lower = task in ("T2",)
    for c in cers:
        rep = np.array(reps[c], float)
        ret = base / rep if lower else rep / base
        ret = ret[np.isfinite(ret)]
        rows.append({"c": c, "point": point[c],
                     "lo": float(np.nanquantile(rep, lo_q)),
                     "hi": float(np.nanquantile(rep, hi_q)),
                     "ret": (point[cers[0]] / point[c]) if lower else (point[c] / point[cers[0]]),
                     "ret_lo": float(np.quantile(ret, lo_q)) if len(ret) else np.nan,
                     "ret_hi": float(np.quantile(ret, hi_q)) if len(ret) else np.nan,
                     "n": len(preds[c])})
    return pd.DataFrame(rows)


def thresholds(curve_df):
    """Highest CER keeping >=95% / >=90% of the clean score at every level up to it."""
    out = {}
    for name, keep in (("c95", 0.95), ("c90", 0.90)):
        best = None
        for row in curve_df.sort_values("c").itertuples():
            if row.ret_lo >= keep:
                best = row.c
            else:
                break
        out[name] = best
    return out


def size_bin(letters, edges=LETTER_BINS):
    i = np.searchsorted(edges, letters, side="right")
    if i == 0:
        return "0"
    if i == len(edges):
        return f"≥{edges[-1]}"
    return f"{edges[i - 1]}–{edges[i] - 1}"


def by_size(task, preds, sizes, n_boot=200):
    """The same curve per fragment-size bin; bins with fewer than 30 units get no threshold."""
    rows = []
    bins = {u: size_bin(n) for u, n in sizes.items()}
    for b in sorted(set(bins.values())):
        keep = {u for u, v in bins.items() if v == b}
        sub = {c: df[df["unit"].isin(keep)] for c, df in preds.items()}
        n = len(sub[min(sub)])
        if n < MIN_BIN:
            continue
        cv = curve(task, sub, n_boot)
        rows.append({"bin": b, "n": n, **thresholds(cv),
                     **{f"m{c:g}": v for c, v in zip(cv["c"], cv["point"])}})
    return pd.DataFrame(rows)
