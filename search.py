"""T4 search: the four query sets and the eight methods, scored on clean-text relevance."""
import re
import unicodedata

import numpy as np
from lxml import etree
from numba import njit, prange
from scipy import sparse

from config import BM25, IDP, SEED
from corpus import GAP, GREEK, TEI, XML_LANG, Walker, normalize
from degrade import rng

CODE = {c: i for i, c in enumerate(sorted(GREEK))}
NALPHA = len(CODE)
STEM_ENDINGS = ("ομαι", "ευσ", "εω", "αω", "οω", "μι", "οσ", "ον", "ησ", "ασ", "ισ", "υσ",
                "ωσ", "ων", "αι", "οι", "ω", "η", "α")
FORMULAE = ("χαίρειν", "πολλὰ χαίρειν", "ἐρρῶσθαί σε εὔχομαι", "ἔρρωσο", "ἐρρῶσθαι", "ἔτους",
            "ὁμολογεῖ", "ὁμολογοῦμεν", "ἀπέχω παρὰ σοῦ", "πρὸ μὲν πάντων",
            "εὔχομαί σε ὑγιαίνειν", "τὸ προσκύνημά σου ποιῶ", "παρὰ τῷ κυρίῳ Σαράπιδι",
            "διὸ ἀξιῶ", "διευτύχει", "εὐτύχει", "ἐπιδέδωκα", "σεσημείωμαι",
            "Αὐτοκράτορος Καίσαρος", "ἔγραψα ὑπὲρ αὐτοῦ", "μὴ εἰδότος γράμματα", "ὡς ἐτῶν",
            "ἀσπάζομαι", "ἀσπάζεταί σε", "ἀπόδος", "καθὼς πρόκειται", "κυρία ἡ συγγραφή",
            "ἐπερωτηθεὶς ὡμολόγησα", "ὑπατείας", "ἐν ἀγυιᾷ", "μετὰ κυρίου", "δηλῶ",
            "ἐπεὶ οὖν", "ἐρρῶσθαί σε βούλομαι")


def encode(unit):
    """A unit as letter codes with -1 where a match may not cross (gap or line break)."""
    return np.array([CODE.get(c, -1) for c in unit], dtype=np.int64)


@njit(cache=True)
def _semiglobal(q, t):
    m = len(q)
    peq = np.zeros(NALPHA, np.uint64)
    for i in range(m):
        peq[q[i]] |= np.uint64(1) << np.uint64(i)
    one, last = np.uint64(1), np.uint64(1) << np.uint64(m - 1)
    pv, mv, score, best = ~np.uint64(0), np.uint64(0), m, m
    for j in range(len(t)):
        c = t[j]
        if c < 0:
            pv, mv, score = ~np.uint64(0), np.uint64(0), m
            continue
        eq = peq[c]
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = ph << one
        pv = (mh << one) | ~(xv | ph)
        mv = ph & xv
        if score < best:
            best = score
    return best


@njit(parallel=True, cache=True)
def distances(q, units, starts, ends):
    out = np.empty(len(starts), np.int64)
    for i in prange(len(starts)):
        out[i] = _semiglobal(q, units[starts[i]:ends[i]])
    return out


def pack(units):
    flat = np.concatenate([encode(u) for u in units]) if units else np.zeros(0, np.int64)
    lens = np.array([len(u) for u in units])
    ends = np.cumsum(lens)
    return flat, ends - lens, ends


class CaseWalker(Walker):
    """The ink walker without normalisation, so capitalisation survives for the name queries."""

    def flush(self, runs_on):
        text = "".join(self.buf)
        self.buf.clear()
        if self.n is None and not text.strip():
            return
        self.lines.append({"text": text, "runs_on": runs_on})


def _letters_only(token):
    return "".join(c for c in normalize(token) if c in CODE)


def name_forms(paths):
    """Capitalised, non-sentence-initial whole tokens of an edition, as letters-only forms."""
    out = []
    for path in paths:
        root = etree.parse(str(path)).getroot()
        first = root.find(f".//{TEI}div[@type='edition']")
        if first is None or first.get(XML_LANG) != "grc":
            continue
        w = CaseWalker()
        for div in root.findall(f".//{TEI}div[@type='edition']"):
            w.inner(div)
        prev, forms = ".", set()
        for line in w.done():
            tokens = line["text"].split()
            for j, tok in enumerate(tokens):
                cut = (j == len(tokens) - 1 and line["runs_on"])
                initial = prev[-1:] in (".", "·", ";", ":")
                letters = [c for c in unicodedata.normalize("NFD", tok) if c.isalpha()]
                if not cut and not initial and GAP not in tok and letters and letters[0].isupper():
                    form = _letters_only(tok)
                    if len(form) >= 4:
                        forms.add(form)
                prev = tok
        out.append(forms)
    return out


def stems(preisigke_text):
    """Headword stems: the first word of a line, minus its longest inflectional ending."""
    out = set()
    for line in preisigke_text.splitlines():
        m = re.match(r"\s*(\S+)\s*[.,]", line)
        if not m:
            continue
        form = _letters_only(m.group(1))
        if len(form) < 4:
            continue
        for end in STEM_ENDINGS:
            if form.endswith(end) and len(form) - len(end) >= 4:
                form = form[:-len(end)]
                break
        out.add(form)
    return out


def build_queries(docs, preisigke=None, n_pre=1000, n_names=500, n_syn=100):
    """The four query sets of the paper; each query keeps at least one relevant test document."""
    r = rng(SEED, "queries")
    train = docs[docs.split == "train"]
    test = docs[docs.split == "test"]
    train_text = "\n".join("\n".join(ls) for ls in train["lines"])
    test_units = ["\n".join(ls) for ls in test["lines"]]
    corpus = "\n".join(test_units)

    q = {}
    if preisigke:
        cand = sorted(s for s in stems(preisigke) if s in train_text)
        q["preisigke"] = list(r.choice(cand, size=min(n_pre, len(cand)), replace=False))
    counts = {}
    index = xml_index()
    for forms in name_forms([index[d] for d in train["doc_id"] if d in index]):
        for f in forms:
            counts[f] = counts.get(f, 0) + 1
    names = sorted(f for f, c in counts.items() if c >= 2)
    q["names"] = list(r.choice(names, size=min(n_names, len(names)), replace=False))
    q["formulae"] = [f for f in map(_letters_only, FORMULAE) if f]
    syn = set()
    for length in range(3, 13):
        pool = [seg for u in test_units for seg in re.split(f"[{GAP}\n]", u) if len(seg) >= length]
        if not pool:
            continue
        weights = np.array([len(s) - length + 1 for s in pool], float)
        for _ in range(n_syn * 3):
            if len([s for s in syn if len(s) == length]) >= n_syn:
                break
            seg = pool[r.choice(len(pool), p=weights / weights.sum())]
            at = r.integers(len(seg) - length + 1)
            syn.add(seg[at:at + length])
    q["synthetic"] = sorted(syn)
    return {k: [x for x in v if x in corpus] for k, v in q.items()}


def xml_index():
    """doc_id -> edition file, from the same walk corpus.py uses to build the document table."""
    return {f"{src}/{p.stem}": p
            for top, src in (("DDB_EpiDoc_XML", "ddbdp"), ("DCLP", "dclp"))
            for p in (IDP / top).rglob("*.xml")}


def ngrams(unit, n):
    """Overlapping letter n-grams; none spans a gap token or a line break."""
    return [seg[i:i + n] for seg in re.split(f"[{GAP}\n]", unit)
            for i in range(len(seg) - n + 1)]


def query_terms(q, n):
    """A query's terms: its n-grams, or the whole query when it is shorter than n."""
    return ngrams(q, n) if len(q) >= n else [q]


def counts(units, ns, vocab=None):
    """CSR term counts over letter n-grams. A vocabulary is built when none is given;
    with one, terms outside it are dropped (they carry no weight)."""
    build = vocab is None
    vocab = {} if build else vocab
    indptr, indices, data = [0], [], []
    for u in units:
        c = {}
        for n in ns:
            for g in ngrams(u, n):
                j = vocab.setdefault(g, len(vocab)) if build else vocab.get(g)
                if j is not None:
                    c[j] = c.get(j, 0) + 1
        indices.extend(c)
        data.extend(c.values())
        indptr.append(len(indices))
    m = sparse.csr_matrix((np.array(data, np.float32), indices, indptr),
                          shape=(len(units), len(vocab)))
    return m, vocab


def term_counts(terms_per_row, vocab):
    """CSR counts of pre-tokenised rows (used for queries) against a fixed vocabulary."""
    indptr, indices, data = [0], [], []
    for terms in terms_per_row:
        c = {}
        for g in terms:
            j = vocab.get(g)
            if j is not None:
                c[j] = c.get(j, 0) + 1
        indices.extend(c)
        data.extend(c.values())
        indptr.append(len(indices))
    return sparse.csr_matrix((np.array(data, np.float32), indices, indptr),
                             shape=(len(terms_per_row), len(vocab)))


def l2(m):
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1))).ravel()
    inv = np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)
    return sparse.diags(inv.astype(np.float32)) @ m


class Ranker:
    """TF-IDF cosine and BM25 over letter n-grams, with IDF fitted on clean training units.

    tfidf_cos uses 2-4-grams and the smoothed IDF ln((1 + n) / (1 + df)) + 1; terms unseen in
    training get weight 0.  bm25 uses n-grams of BM25["n"] and ln((N - df + .5)/(df + .5) + 1),
    with the length normalisation taken over the corpus being scored.
    """

    def __init__(self, train_units):
        self.tf_counts, self.tf_vocab = counts(train_units, (2, 3, 4))
        n = self.tf_counts.shape[0]
        df = np.diff(self.tf_counts.tocsc().indptr)
        self.tf_idf = np.where(df > 0, np.log((1 + n) / (1 + df)) + 1, 0.0).astype(np.float32)

        self.bm_counts, self.bm_vocab = counts(train_units, (BM25["n"],))
        df = np.diff(self.bm_counts.tocsc().indptr)
        self.bm_idf = np.log((n - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

    def index(self, units):
        """Weighted matrices of the corpus that is actually scored (the degraded one)."""
        tf, _ = counts(units, (2, 3, 4), self.tf_vocab)
        bm, _ = counts(units, (BM25["n"],), self.bm_vocab)
        length = np.asarray(bm.sum(1)).ravel()
        avg = length.mean() or 1.0
        norm = BM25["k1"] * (1 - BM25["b"] + BM25["b"] * length / avg)
        coo = bm.tocoo()
        w = coo.data * (BM25["k1"] + 1) / (coo.data + norm[coo.row])
        return (l2(tf.multiply(self.tf_idf)).tocsr(),
                sparse.csr_matrix((w.astype(np.float32), (coo.row, coo.col)), shape=bm.shape))

    def scores(self, index, q, method):
        """Score every unit of an indexed corpus against one query."""
        if method == "tfidf_cos":
            terms = [g for n in (2, 3, 4) for g in ngrams(q, n)]
            qv = l2(term_counts([terms], self.tf_vocab).multiply(self.tf_idf).tocsr())
            return np.asarray((index[0] @ qv.T).todense()).ravel()
        qv = term_counts([query_terms(q, BM25["n"])], self.bm_vocab).multiply(self.bm_idf)
        return np.asarray((index[1] @ qv.tocsr().T).todense()).ravel()


def relevance(queries, clean_units):
    return {q: {i for i, u in enumerate(clean_units) if q in u} for q in queries}


METHODS = ("exact", "lev1", "lev2", "normdist", "minedit", "trigram", "tfidf_cos", "bm25")


def _top(score, top, high=True):
    """Indices of the best `top` units, dropping those a method would not retrieve at all."""
    order = np.argsort(-score if high else score, kind="stable")[:top]
    return [int(i) for i in order if (score[i] > 0 if high else score[i] < np.inf)]


def run_methods(queries, units, clean_rel, methods=None, top=100, ranker=None):
    """Scores of every method over one degraded corpus; ranking keeps the top 100 units.

    `ranker` is a Ranker fitted on the clean training units; it is required for tfidf_cos
    and bm25, whose IDF must not be re-fitted on the degraded corpus being scored.
    """
    methods = methods or METHODS
    flat, starts, ends = pack(units)
    out = {m: {} for m in methods}
    tri = [set(u[i:i + 3] for i in range(len(u) - 2) if GAP not in u[i:i + 3]
               and "\n" not in u[i:i + 3]) for u in units]
    vector = {"tfidf_cos", "bm25"} & set(methods)
    if vector and ranker is None:
        raise ValueError(f"{sorted(vector)} need a Ranker fitted on the clean train units")
    index = ranker.index(units) if vector else None
    for q in queries:
        d = distances(encode(q), flat, starts, ends)
        m_len = len(q)
        rel = clean_rel[q]
        for m in methods:
            hits, ranked = None, []
            if m == "exact":
                hits = set(np.flatnonzero(d == 0).tolist())
            elif m == "lev1":
                hits = set(np.flatnonzero(d <= 1).tolist())
            elif m == "lev2":
                hits = set(np.flatnonzero(d <= 2).tolist())
            elif m == "normdist":
                hits = set(np.flatnonzero(d / m_len <= 0.25).tolist())
                ranked = [i for i in np.argsort(d, kind="stable")[:top] if i in hits]
            elif m == "minedit":
                ranked = [int(i) for i in np.argsort(d, kind="stable")[:top] if d[i] < m_len]
            elif m == "trigram":
                qt = {q[i:i + 3] for i in range(m_len - 2)}
                share = np.array([len(qt & t) / max(len(qt), 1) for t in tri])
                hits = set(np.flatnonzero(share >= 0.5).tolist())
                ranked = _top(share, top)
            else:
                ranked = _top(ranker.scores(index, q, m), top)
            out[m][q] = (hits, ranked, rel)
    return out


def metrics(per_query):
    """Micro filter precision/recall/F1 and the ranking scores the paper reports."""
    tp = hits = rel_n = 0
    filters = False
    rec20, mrr, ndcg = [], [], []
    for hits_set, ranked, rel in per_query.values():
        if hits_set is not None:
            filters = True
            tp += len(hits_set & rel)
            hits += len(hits_set)
            rel_n += len(rel)
        if ranked and rel:
            top20 = [u for u in ranked[:20] if u in rel]
            rec20.append(len(top20) / min(len(rel), 20))
            first = next((i for i, u in enumerate(ranked) if u in rel), None)
            mrr.append(0.0 if first is None else 1 / (first + 1))
            gain = sum(1 / np.log2(i + 2) for i, u in enumerate(ranked[:10]) if u in rel)
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
            ndcg.append(gain / ideal if ideal else 0.0)
    p = tp / hits if hits else 0.0
    rc = tp / rel_n if rel_n else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    # a ranking-only method (minedit, tfidf_cos, bm25) has no filter, so it scores no P/R/F1
    return {"filter_p": p if filters else None, "filter_r": rc if filters else None,
            "filter_f1": f1 if filters else None,
            "recall20": float(np.mean(rec20)) if rec20 else None,
            "mrr": float(np.mean(mrr)) if mrr else None,
            "ndcg10": float(np.mean(ndcg)) if ndcg else None}
