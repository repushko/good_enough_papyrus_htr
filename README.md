# Which papyrus HTR is good enough?

Reference code for *Which papyrus HTR is good enough? Character-error-rate tolerance of four
papyrological tasks on Greek texts* (A. Repushko and E. Chepel).

The paper asks how accurate handwritten text recognition of Greek papyri has to be before four
papyrological tasks stop working. Editions are reduced to what a perfect HTR would return
(letters only), degraded to an **exact** character error rate, and fed to each task.

This repository is a compact reimplementation of that pipeline: it is the minimum needed to
reproduce what the paper reports, not the production code that produced the published numbers.
See [Differences from the paper's own run](#differences-from-the-papers-own-run).

## Layout

| File | Paper |
|---|---|
| `config.py` | seeds, CER grid, split fractions, size bins, BM25 settings |
| `corpus.py` | EpiDoc → ink → letters-only (L3); document table; TM and near-duplicate groups; splits (Sects. 3.2, 3.3, 5.1, App. A) |
| `degrade.py` | exact-CER degradation, lost lines, error-shape variants (Sect. 5.3) |
| `tasks.py` | T1–T3 label tables, macro-F1 / MAE, trivial baselines, line-length features (Sect. 4) |
| `models.py` | TF-IDF, fastText, char-CNN, ByT5-small, line-length control (Sect. 6, App. C) |
| `search.py` | T4 query sets and the eight search methods (Sects. 4.4, 6) |
| `analyze.py` | paired bootstrap, retention, c95 / c90, size bins (Sect. 7) |
| `run.py` | command line |

Task identifiers here follow the repository, not the paper: documentary versus literary is `T4`
and search is `T5`.

## Data

Two inputs are downloaded separately and are not redistributed here:

* **papyri.info EpiDoc** (CC-BY), pinned at commit `fd88c0c` — the editions themselves.
* **Grammateus export** `papyri.csv` — document-type labels, needed for T1 only.

Put them under a data directory and point `HTR_DATA` at it (it defaults to `./data`):

```
data/
  idp.data/          # git clone https://github.com/papyri/idp.data
  grammateus/papyri.csv
```

One of the four T4 query sets is built from the OCR text of the Preisigke supplements, which is
in copyright; pass it explicitly with `search.build_queries(docs, preisigke=text)`. Without it
the other three sets are used.

## Run

```sh
pip install -r requirements.txt
export HTR_DATA=/path/to/data

python run.py corpus                                     # ~20 s on 12 cores
python run.py experiment --task T3 --model tfidf --cers 0 5 10 20 30 --seeds 1 2 3
python run.py experiment --task T1 --model tfidf --regime matched --cers 0 10 20
python run.py search     --cers 0 10 20 30               # all eight methods
python run.py queries                                    # just the query sets
```

`--limit N` samples N documents per split for a quick check; note that recall@20 is defined
against the corpus being searched, so a limited run is not comparable to a full one.
`--variant bursty|sub_only|uniform|uneven` and `--lost-lines 10|20|30` reproduce the
supplementary figure and the lost-lines appendix. fastText needs `fasttext`; char-CNN and
ByT5-small need `torch` (and `transformers`).

## Differences from the paper's own run

* Splits come out at 50,919 / 6,397 / 6,530 documents against the paper's 51,042 / 6,370 /
  6,434. The grouping rule is the same; the near-duplicate components differ slightly because
  the MinHash candidates are queried in a different order. Corpus totals are identical: 63,846
  documents and 18,594,905 letters.
* The degradation budget is spent over the task's own evaluation set rather than over the whole
  split, so each experiment hits its target CER exactly on the text it scores.
* Edits are placed on the scored letter sequence (non-adjacent there), not within the runs
  between gap tokens. The paper's rule leaves letters on either side of a gap non-adjacent for
  placement although they are adjacent once the gaps are stripped for scoring, which lets a few
  edits cancel in heavily fragmented documents; the paper's own code recovers by redrawing.
  Placing on the scored sequence gives the same distribution and makes the invariant hold by
  construction.
* Hyperparameters are not re-tuned here. The values in `models.FROZEN` and `config.BM25` are the
  ones the paper tuned once on the clean validation split and then froze.

Because of the first three, numbers from this code are close to the published ones but not
identical to the decimal.

## Data licence

The editions are CC-BY from papyri.info and are not included here. Cite the papyri.info
`idp.data` repository and the Grammateus project if you use them.
