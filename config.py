import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("HTR_DATA", ROOT / "data"))
IDP = DATA / "idp.data"
GRAMMATEUS = DATA / "grammateus" / "papyri.csv"
OUT = ROOT / "out"

SEED = 20260913
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SHINGLE, JACCARD, NUM_PERM, CV_FOLDS = 5, 0.8, 128, 5

CER_GRID = [0, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 40, 50]
TEST_SEEDS = [1, 2, 3]
TRAIN_SEED, VAL_SEED = 11, 21
MIX = {"sub": 3, "ins": 1, "del": 1}
VARIANTS = {"sub_only": {"sub": 1, "ins": 0, "del": 0}, "uniform": {"sub": 1, "ins": 1, "del": 1},
            "bursty": {"run": (2, 4)}, "uneven": {"sd": 0.5}}
LOST_LINES = [10, 20, 30]
LOST_LINES_CER = [0, 5, 10, 15, 20, 30]

T2_MAX_WIDTH = 100
N_BOOT, CI, MIN_BIN = 1000, 0.95, 30
LETTER_BINS = [1, 20, 50, 100, 200, 500, 1000, 2000]
QUERY_BINS = [3, 5, 7, 10, 13]

BM25 = {"n": 6, "k1": 0.3, "b": 0.0}
