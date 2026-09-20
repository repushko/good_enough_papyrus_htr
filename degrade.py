"""Seeded degradation of L3 text to an exact CER, under the Levenshtein invariant."""
import hashlib

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from config import MIX, VARIANTS
from corpus import GAP, GREEK

ALPHA = np.array(sorted(GREEK))
IDX = {c: i for i, c in enumerate(ALPHA)}


def rng(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng([int.from_bytes(h[i:i + 4], "little") for i in range(0, 32, 4)])


def apportion(total, weights, caps, r):
    """Largest-remainder apportionment of `total` over `weights`, respecting `caps`."""
    w = np.asarray(weights, float)
    out = np.zeros(len(w), int)
    left = int(total)
    while left > 0 and w.sum() > 0:
        exact = left * w / w.sum()
        add = np.minimum(np.floor(exact).astype(int), caps - out)
        out += add
        left -= add.sum()
        if left <= 0:
            break
        room = caps - out
        frac = exact - np.floor(exact)
        order = np.lexsort((r.random(len(w)), -frac))
        for i in order:
            if left == 0:
                break
            if room[i] > 0:
                out[i] += 1
                room[i] -= 1
                left -= 1
        w = np.where(caps - out > 0, w, 0.0)
    return out


def letter_positions(text):
    """Offsets of the scored letters: the sequence the CER is measured on."""
    return [i for i, c in enumerate(text) if c in IDX]


def positions(n, k, r):
    """k uniformly random pairwise non-adjacent offsets in a sequence of n letters."""
    c = np.sort(r.choice(n - k + 1, size=k, replace=False))
    return c + np.arange(k)


def burst_positions(n, k, r, lo, hi):
    """k substituted letters grouped into runs of lo..hi consecutive letters."""
    sizes = []
    while sum(sizes) < k:
        sizes.append(min(int(r.integers(lo, hi + 1)), k - sum(sizes)))
    free = n - k - (len(sizes) - 1)
    if free < 0:
        return positions(n, k, r)
    cuts = np.sort(r.choice(free + len(sizes), size=len(sizes), replace=False))
    out = []
    for i, size in enumerate(sizes):
        at = cuts[i] + sum(sizes[:i])
        out.extend(range(at, at + size))
    return np.array(sorted(set(out))[:k])


def ops_for(k, mix, r):
    p = np.array([mix.get("sub", 0), mix.get("ins", 0), mix.get("del", 0)], float)
    return r.choice(3, size=k, p=p / p.sum())


def capacity(letters, variant="base"):
    return len(letters) if variant == "bursty" else (len(letters) + 1) // 2


def degrade_doc(lines, k, r, variant="base"):
    """Apply exactly k single-step edits to one document; each costs one Levenshtein step."""
    text = "\n".join(lines)
    letters = letter_positions(text)
    k = min(int(k), capacity(letters, variant))
    if k == 0:
        return list(lines), []
    spec = VARIANTS.get(variant, {})
    mix = {**MIX, **{key: v for key, v in spec.items() if key in ("sub", "ins", "del")}}
    for attempt in range(20):
        rr = rng(r, attempt)
        if variant == "bursty":
            slots = burst_positions(len(letters), k, rr, *spec["run"])
        else:
            slots = positions(len(letters), k, rr)
        ops = ops_for(len(slots), mix, rr)
        edits = _letters(text, letters, slots, ops, rr)
        out = _apply(text, edits)
        clean = "".join(c for c in text if c in IDX)
        dirty = "".join(c for c in out if c in IDX)
        if Levenshtein.distance(clean, dirty) == k:
            return out.split("\n"), edits
        mix = {"sub": 1, "ins": 0, "del": 0}  # substitutions alone cannot shift the alignment
    raise RuntimeError("degradation invariant not met")


def _letters(text, letters, slots, ops, r):
    """(offset, op, new letter) per slot: a substitution, an insertion or a deletion."""
    edits = []
    for j, op in zip(slots, ops):
        i = letters[j]
        old = text[i]
        prev = text[letters[j - 1]] if j > 0 else None
        nxt = text[letters[j + 1]] if j + 1 < len(letters) else None
        if op == 2 and (old == prev or old == nxt):
            op = 0  # deleting a letter equal to a neighbour would not cost one step
        if op == 0:
            pool = [c for c in ALPHA if c != old]
        elif op == 1:
            pool = [c for c in ALPHA if c != prev and c != old]
        else:
            pool = [""]
        edits.append((int(i), int(op), pool[r.integers(len(pool))]))
    return edits


def _apply(text, edits):
    out, at = [], 0
    for i, op, new in edits:
        out.append(text[at:i])
        if op == 0:
            out.append(new)
        elif op == 1:
            out.append(new + text[i])
        at = i + 1
    out.append(text[at:])
    return "".join(out)


def lost_lines(lines, p, r):
    """Drop a seeded fraction p of the text-bearing lines, keeping at least one."""
    idx = [i for i, ln in enumerate(lines) if any(c in IDX for c in ln)]
    drop = int(round(p / 100 * len(idx)))
    drop = min(drop, max(len(idx) - 1, 0))
    gone = set(r.choice(idx, size=drop, replace=False)) if drop else set()
    return [ln for i, ln in enumerate(lines) if i not in gone]


def degrade_split(docs, c, seed, variant="base", p=0):
    """Degrade a split to exactly c% CER: the budget is round(c * letters) over the split."""
    r = rng(c, seed, variant, p)
    kept = []
    for doc_id, lines in zip(docs["doc_id"], docs["lines"]):
        ls = list(lines)
        if p:
            ls = lost_lines(ls, p, rng(doc_id, p, seed))
        kept.append(ls)
    counts = np.array([sum(1 for ln in ls for ch in ln if ch in IDX) for ls in kept])
    caps = np.array([capacity(letter_positions("\n".join(ls)), variant) for ls in kept])
    budget = apportion(int(round(c / 100 * counts.sum())), counts.astype(float), caps, r)
    out = []
    for doc_id, ls, k in zip(docs["doc_id"], kept, budget):
        lines, _ = degrade_doc(ls, int(k), rng(doc_id, c, seed, variant, p), variant)
        out.append(lines)
    return pd.DataFrame({"doc_id": list(docs["doc_id"]), "lines": out, "k": budget,
                         "letters": counts})
