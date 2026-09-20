"""EpiDoc editions -> letters-only text (L3), document table, grouped splits."""
import hashlib, os, re, unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import numpy as np, pandas as pd
from lxml import etree

from config import IDP, GRAMMATEUS, OUT, SEED, FRACTIONS, SHINGLE, JACCARD, NUM_PERM, CV_FOLDS

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
GAP = "▯"
GREEK = frozenset("αβγδεζηθικλμνξοπρστυφχψωϛϙϡ")
FOLD = {"ς": "σ", "ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϕ": "φ", "ϱ": "ρ", "ϰ": "κ", "ϖ": "π",
        "ϝ": "ϛ", "ϟ": "ϙ"}
DROP = frozenset("reg corr ex rdg g am note certainty figure figDesc desc ref handShift "
                 "milestone".split())
LOSS = frozenset({"lost", "undefined"})
SCHEME = "affine32"
_SP, _GAPS, _YEAR = re.compile(r"\s+"), re.compile(f"{GAP}(?:\\s*{GAP})+"), re.compile(r"-?\d{4}")


def normalize(text):
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch):
            continue
        if ch == GAP:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            ch = FOLD.get(ch.lower(), ch.lower())
            if ch in GREEK:
                out.append(ch)
    return _GAPS.sub(GAP, _SP.sub(" ", "".join(out)).strip())


def l3(line):
    return line.replace(" ", "")


def n_letters(text):
    return sum(1 for c in text if c not in (GAP, " ", "\n"))


class Walker:
    """Keeps the ink: drops the editorial layer, marks each contiguous loss with one gap token."""

    def __init__(self):
        self.lines, self.buf, self.n = [], [], None

    def inner(self, el):
        if el.text:
            self.buf.append(el.text)
        for child in el:
            self.visit(child)

    def visit(self, el):
        if isinstance(el.tag, str):
            name = etree.QName(el).localname
            if name in ("lb", "l"):
                self.brk(el)
                if name == "l":
                    self.inner(el)
            elif name in DROP:
                pass
            elif name == "gap":
                self.buf.append(GAP)
            elif name == "supplied":
                loss = el.get("reason") in LOSS
                if loss:
                    self.buf.append(GAP)
                for lb in el.iter(f"{TEI}lb"):
                    self.brk(lb)
                    if loss:
                        self.buf.append(GAP)
            elif name == "space":
                self.buf.append(" ")
            else:
                self.inner(el)
        if el.tail:
            self.buf.append(el.tail)

    def brk(self, el):
        self.flush(el.get("break") == "no")
        self.n = el.get("n")

    def flush(self, runs_on):
        text = normalize("".join(self.buf))
        self.buf.clear()
        if self.n is None and not text:
            return
        self.lines.append({"text": text, "runs_on": runs_on})

    def done(self):
        self.flush(False)
        return self.lines


def render(path, source):
    root = etree.parse(str(path)).getroot()
    first = root.find(f".//{TEI}div[@type='edition']")
    if first is None or first.get(XML_LANG) != "grc":
        return None
    if root.find(f".//{TEI}ref[@type='reprint-in']") is not None:
        return None
    w = Walker()
    for div in root.findall(f".//{TEI}div[@type='edition']"):
        w.inner(div)
    lines = w.done()
    ids = {}
    for idno in root.iterfind(f".//{TEI}publicationStmt/{TEI}idno"):
        k, v = idno.get("type"), (idno.text or "").strip()
        if k and v and k not in ids:
            ids[k] = v
    tm = ids.get("TM", "").split()
    hyb = ("dclp-hybrid", "ddb-hybrid") if source == "dclp" else ("ddb-hybrid",)
    return {"doc_id": f"{source}/{path.stem}", "source": source,
            "tm": int(tm[0]) if tm and tm[0].isdigit() else None,
            "hybrid": next((ids[k] for k in hyb if ids.get(k)), None),
            "hgv": ids.get("HGV", "").split(),
            "lines": [l3(ln["text"]) for ln in lines],
            "words": [ln["text"] for ln in lines],
            "letters": sum(n_letters(ln["text"]) for ln in lines),
            "text_lines": sum(1 for ln in lines if n_letters(ln["text"]))}


def _render(args):
    return render(*args)


def parse_hgv(path):
    root = etree.parse(str(path)).getroot()
    lo = hi = nome = None
    for od in root.iter(f"{TEI}origDate"):
        when = _year(od.get("when"))
        a, b = _year(od.get("notBefore")) or when, _year(od.get("notAfter")) or when
        if a is not None and b is not None:
            lo, hi = a, b
            break
    for place in root.iter(f"{TEI}placeName"):
        text = " ".join("".join(place.itertext()).split())
        if place.get("subtype") == "nome" and text:
            nome = text
            break
    return {"hgv_id": path.stem, "date_lo": lo, "date_hi": hi, "nome": nome}


def _year(v):
    m = _YEAR.match(v.strip()) if v else None
    return int(m.group()) if m else None


def build(workers=None):
    workers = workers or min(12, os.cpu_count() or 1)
    jobs = [(p, "ddbdp") for p in sorted((IDP / "DDB_EpiDoc_XML").rglob("*.xml"))]
    jobs += [(p, "dclp") for p in sorted((IDP / "DCLP").rglob("*.xml"))]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_render, jobs, chunksize=200) if r]
    docs = pd.DataFrame(rows)

    hgv_files = sorted((IDP / "HGV_meta_EpiDoc").rglob("*.xml"))
    with ProcessPoolExecutor(workers) as ex:
        hgv = pd.DataFrame(list(ex.map(parse_hgv, hgv_files, chunksize=200)))
    dates = {r.hgv_id: (r.date_lo, r.date_hi) for r in hgv.itertuples() if pd.notna(r.date_lo)}
    first = [next((dates[i] for i in ids if i in dates), None) for ids in docs["hgv"]]
    docs["date_lo"] = pd.array([d[0] if d else None for d in first], dtype="Int32")
    docs["date_hi"] = pd.array([d[1] if d else None for d in first], dtype="Int32")
    docs["date_mid"] = (docs["date_lo"].astype("Float64") + docs["date_hi"]) / 2
    docs["date_width"] = (docs["date_hi"] - docs["date_lo"]).astype("Int32")

    if GRAMMATEUS.exists():
        g = pd.read_csv(GRAMMATEUS, skipinitialspace=True, dtype=str)
        g.columns = [c.strip() for c in g.columns]
        g["hybrid"] = g["papyri.info"].map(
            lambda u: unquote(str(u).strip().rstrip("/").rsplit("/", 1)[-1]))
        g = g.drop_duplicates(["hybrid", "Type", "Subtype"])
        g = g[~g["hybrid"].duplicated(keep=False)].set_index("hybrid")
        ddb = docs["source"] == "ddbdp"
        for col, src in (("gram_type", "Type"), ("gram_subtype", "Subtype")):
            m = docs["hybrid"].map(g[src])
            docs[col] = [v if isinstance(v, str) and d else None for v, d in zip(m, ddb)]
    else:
        docs["gram_type"] = docs["gram_subtype"] = None

    both = set(docs.loc[docs.source == "ddbdp", "tm"].dropna()) & \
        set(docs.loc[docs.source == "dclp", "tm"].dropna())
    docs["tm_in_both"] = docs["tm"].isin(both)
    return docs.reset_index(drop=True)


def _shingles(text, k):
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _signature(text):
    from datasketch import MinHash
    if len(text) < SHINGLE:
        return None
    m = MinHash(num_perm=NUM_PERM, seed=1, scheme=SCHEME)
    m.update_batch([s.encode() for s in _shingles(text, SHINGLE)])
    return m.hashvalues


def _sig_chunk(texts):
    return [_signature(t) for t in texts]


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def split_of(key):
    h = int.from_bytes(hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:8], "big") / 2 ** 64
    edge = 0.0
    for name, frac in FRACTIONS.items():
        edge += frac
        if h < edge:
            return name
    return "test"


def splits(docs, workers=None):
    """Group by TM and by near-duplicate text, then assign groups to train/val/test."""
    from datasketch import LeanMinHash, MinHashLSH
    from sklearn.model_selection import StratifiedGroupKFold
    workers = workers or min(12, os.cpu_count() or 1)
    texts = ["".join(ls).replace(GAP, "") for ls in docs["lines"]]
    chunks = [texts[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_sig_chunk, chunks))
    sigs = [None] * len(texts)
    for w, part in enumerate(parts):
        for j, sig in enumerate(part):
            sigs[w + j * workers] = sig

    u = Union(len(docs))
    by_tm = {}
    for i, tm in enumerate(docs["tm"]):
        if pd.notna(tm):
            u.union(by_tm.setdefault(tm, i), i)
    lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
    lean = {}
    for i, hv in enumerate(sigs):
        if hv is None:
            continue
        lean[i] = LeanMinHash(seed=1, hashvalues=hv, scheme=SCHEME)
        for j in lsh.query(lean[i]):
            if lean[i].jaccard(lean[j]) >= JACCARD:
                u.union(i, j)
        lsh.insert(i, lean[i])

    root = [u.find(i) for i in range(len(docs))]
    key = {r: docs["doc_id"].iat[r] for r in set(root)}
    docs["group"] = root
    docs["split"] = [split_of(key[r]) for r in root]

    docs["fold"] = pd.array([None] * len(docs), dtype="Int8")
    t1 = docs[docs["gram_type"].notna()]
    if len(t1):
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        for f, (_, idx) in enumerate(cv.split(t1, t1["gram_type"], t1["group"])):
            docs.loc[t1.index[idx], "fold"] = f
    return docs


def load():
    return pd.read_parquet(OUT / "documents.parquet")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs = splits(build())
    docs.to_parquet(OUT / "documents.parquet")
    print(f"{len(docs)} documents, {docs.letters.sum():,} letters")
    print(docs.groupby("split").size().to_dict())


if __name__ == "__main__":
    main()
