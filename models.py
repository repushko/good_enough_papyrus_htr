"""The models the paper reports, with the hyperparameters frozen in its Appendix C."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline

from tasks import line_features

FROZEN = {"tfidf": {"min_df": 2, "C": 100.0, "alpha": 1.0},
          "fasttext": {"dim": 100, "ngrams": 5, "epochs": 30, "lr": 1.0, "bins": 25},
          "charcnn": {"lr": 1e-3, "epochs": 20, "patience": 4, "max_len": 4096},
          "byt5": {"lr": 1e-4, "head_lr": 1e-3, "epochs": 3, "max_bytes": 2048}}


def vec(min_df=2):
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, min_df=min_df)


def tfidf(task, **kw):
    p = {**FROZEN["tfidf"], **kw}
    head = (Ridge(alpha=p["alpha"]) if task == "T2" else
            LogisticRegression(C=p["C"], max_iter=2000, class_weight="balanced"))
    return make_pipeline(vec(p["min_df"]), head)


class LineLength:
    """T3 control: gradient-boosted trees that see line-length statistics and no letters."""

    def __init__(self):
        self.m = HistGradientBoostingClassifier(max_leaf_nodes=31, random_state=0)

    def fit(self, lines, y):
        self.m.fit(np.array([line_features(ls) for ls in lines]), y)
        return self

    def predict(self, lines):
        return self.m.predict(np.array([line_features(ls) for ls in lines]))


class FastText:
    """Supervised fastText over letters, so its word n-grams are letter n-grams."""

    def __init__(self, task, **kw):
        self.task, self.p = task, {**FROZEN["fasttext"], **kw}

    def _tok(self, text):
        return " ".join("<nl>" if c == "\n" else "<gap>" if c == "▯" else c for c in text)

    def fit(self, texts, y):
        import tempfile

        import fasttext
        if self.task == "T2":
            self.bins = np.floor(np.asarray(y, float) / self.p["bins"]).astype(int)
            labels = [f"b{b}" for b in self.bins]
        else:
            labels = list(y)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for text, lab in zip(texts, labels):
                f.write(f"__label__{lab} {self._tok(text)}\n")
            path = f.name
        self.m = fasttext.train_supervised(
            path, dim=self.p["dim"], wordNgrams=self.p["ngrams"], epoch=self.p["epochs"],
            lr=self.p["lr"], minCount=1, bucket=2_000_000, loss="softmax", thread=1)
        return self

    def predict(self, texts):
        rows = [self._tok(t) for t in texts]
        if self.task != "T2":
            return np.array([lab[0].replace("__label__", "")
                             for lab in self.m.predict(rows)[0]])
        labs, probs = self.m.predict(rows, k=-1)
        out = []
        for ls, ps in zip(labs, probs):
            mids = np.array([int(l.replace("__label__b", "")) * self.p["bins"]
                             + self.p["bins"] / 2 for l in ls])
            out.append(float((mids * np.array(ps)).sum() / np.sum(ps)))
        return np.array(out)


def _torch():
    import torch
    return torch


class CharCNN:
    """Letter embeddings, three stages of residual dilated convolutions, max+mean pooling."""

    def __init__(self, task, classes=None, **kw):
        self.task, self.classes, self.p = task, classes, {**FROZEN["charcnn"], **kw}

    def _encode(self, texts):
        torch = _torch()
        from corpus import GAP, GREEK
        idx = {c: i + 1 for i, c in enumerate(sorted(GREEK))}
        idx[GAP], idx["\n"] = len(idx) + 1, len(idx) + 2
        n = self.p["max_len"]
        out = torch.zeros(len(texts), n, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [idx.get(c, 0) for c in t[:n]]
            out[i, :len(ids)] = torch.tensor(ids)
        return out

    def _net(self, n_out):
        torch = _torch()
        from torch import nn

        class Block(nn.Module):
            def __init__(self, ch, d):
                super().__init__()
                self.f = nn.Sequential(nn.BatchNorm1d(ch), nn.GELU(),
                                       nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d),
                                       nn.Conv1d(ch, ch, 1))

            def forward(self, x):
                return x + self.f(x)

        layers = [nn.Embedding(30, 64, padding_idx=0)]
        net = nn.Sequential(nn.Conv1d(64, 256, 7, padding=3),
                            *[m for _ in range(3) for m in
                              (Block(256, 1), Block(256, 2), nn.MaxPool1d(2))])
        head = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out))
        return layers[0], net, head

    def fit(self, texts, y, val=None):
        torch = _torch()
        from torch import nn
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.task == "T2":
            self.mu, self.sd = float(np.mean(y)), float(np.std(y)) or 1.0
            target = torch.tensor((np.asarray(y, float) - self.mu) / self.sd).float()
            n_out, loss_fn = 1, nn.SmoothL1Loss()
        else:
            self.classes = sorted(set(y))
            lut = {c: i for i, c in enumerate(self.classes)}
            target = torch.tensor([lut[v] for v in y])
            n_out, loss_fn = len(self.classes), nn.CrossEntropyLoss()
        self.emb, self.net, self.head = (m.to(self.dev) for m in self._net(n_out))
        params = [*self.emb.parameters(), *self.net.parameters(), *self.head.parameters()]
        opt = torch.optim.AdamW(params, lr=self.p["lr"], weight_decay=0.01)
        x = self._encode(texts)
        for _ in range(self.p["epochs"]):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), 32):
                b = perm[i:i + 32]
                opt.zero_grad()
                out = self._forward(x[b].to(self.dev))
                loss = loss_fn(out.squeeze(-1) if n_out == 1 else out, target[b].to(self.dev))
                loss.backward()
                opt.step()
        return self

    def _forward(self, ids):
        torch = _torch()
        h = self.net(self.emb(ids).transpose(1, 2))
        return self.head(torch.cat([h.max(-1).values, h.mean(-1)], -1))

    def predict(self, texts):
        torch = _torch()
        x, out = self._encode(texts), []
        with torch.no_grad():
            for i in range(0, len(x), 64):
                out.append(self._forward(x[i:i + 64].to(self.dev)).cpu())
        z = torch.cat(out)
        if self.task == "T2":
            return (z.squeeze(-1).numpy() * self.sd + self.mu)
        return np.array([self.classes[i] for i in z.argmax(-1).numpy()])


class ByT5:
    """The ByT5-small encoder with masked mean pooling and a linear head."""

    def __init__(self, task, path="google/byt5-small", **kw):
        self.task, self.path, self.p = task, path, {**FROZEN["byt5"], **kw}

    def fit(self, texts, y):
        torch = _torch()
        from torch import nn
        from transformers import T5EncoderModel
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = T5EncoderModel.from_pretrained(self.path).to(self.dev)
        if self.task == "T2":
            self.mu, self.sd = float(np.mean(y)), float(np.std(y)) or 1.0
            target = torch.tensor((np.asarray(y, float) - self.mu) / self.sd).float()
            n_out, loss_fn = 1, nn.SmoothL1Loss()
        else:
            self.classes = sorted(set(y))
            lut = {c: i for i, c in enumerate(self.classes)}
            target = torch.tensor([lut[v] for v in y])
            n_out, loss_fn = len(self.classes), nn.CrossEntropyLoss()
        self.head = nn.Linear(self.enc.config.d_model, n_out).to(self.dev)
        opt = torch.optim.AdamW([{"params": self.enc.parameters(), "lr": self.p["lr"]},
                                 {"params": self.head.parameters(), "lr": self.p["head_lr"]}],
                                weight_decay=0.01)
        ids, mask = self._encode(texts)
        for _ in range(self.p["epochs"]):
            perm = torch.randperm(len(ids))
            for i in range(0, len(ids), 8):
                b = perm[i:i + 8]
                opt.zero_grad()
                out = self._forward(ids[b], mask[b])
                loss = loss_fn(out.squeeze(-1) if n_out == 1 else out, target[b].to(self.dev))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.enc.parameters(), 1.0)
                opt.step()
        return self

    def _encode(self, texts):
        torch = _torch()
        n = self.p["max_bytes"]
        ids = torch.zeros(len(texts), n, dtype=torch.long)
        mask = torch.zeros(len(texts), n, dtype=torch.long)
        for i, t in enumerate(texts):
            b = t.encode()[:n - 1] + b"\x01"
            ids[i, :len(b)] = torch.tensor([c + 3 for c in b])
            mask[i, :len(b)] = 1
        return ids, mask

    def _forward(self, ids, mask):
        h = self.enc(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).last_hidden_state
        m = mask.to(self.dev).unsqueeze(-1)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))

    def predict(self, texts):
        torch = _torch()
        ids, mask = self._encode(texts)
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 16):
                out.append(self._forward(ids[i:i + 16], mask[i:i + 16]).cpu())
        z = torch.cat(out)
        if self.task == "T2":
            return z.squeeze(-1).numpy() * self.sd + self.mu
        return np.array([self.classes[i] for i in z.argmax(-1).numpy()])


def build(name, task, **kw):
    if name == "tfidf":
        return tfidf(task, **kw)
    if name == "fasttext":
        return FastText(task, **kw)
    if name == "charcnn":
        return CharCNN(task, **kw)
    if name == "byt5":
        return ByT5(task, **kw)
    if name == "linelen":
        return LineLength()
    raise ValueError(name)
