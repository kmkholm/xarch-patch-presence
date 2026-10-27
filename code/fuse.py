"""Evidence fusion: combine the learned score with two explicit
instruction-evidence scores by gradient-boosted trees (PS_LEARNER=logreg: logistic
regression) that are cross-fitted
over the CVE folds.

Features per triplet (target t, vulnerable reference v, patched reference p):
  1. logit of the learned score
  2. overlap  = cos(b_t, b_p) - cos(b_t, b_v)   (bags of normalised instructions)
  3. delta    = (|D+ in t| - |D- in t|) / (|D+| + |D-|),  D+ = b_p - b_v, D- = b_v - b_p
  4. 1 if the two references have identical instruction bags (no visible change)
  5-7. the same three scores computed on constant anchors (immediates and
       displacements <= 0xFFFF), which are more stable across compilers and
       optimisation levels than whole instructions

For test fold k the regression is fitted on the out-of-fold predictions of the
other four folds. Calibration scores for conformal risk control on fold k are
produced without fold k: for every other fold j, a regression fitted on the
folds outside {j, k} scores fold j (written with split = "xcal").
Usage: python fuse.py full [x86ref nogate ...]
Output: runs/<config>+fused/fold<k>.csv
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT
from finetune import CONFIGS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
_BAGS = {}


def bags_for(keys):
    missing = keys - set(_BAGS)
    if missing:
        for line in open(os.path.join(ROOT, "functions", "debian_funcs.jsonl")):
            d = json.loads(line)
            if d["key"] in missing:
                _BAGS[d["key"]] = Counter(" ".join(i) for i in d["insns"])
    return _BAGS


def cos(a, b):
    num = sum(a[k] * b[k] for k in a if k in b)
    return num / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values())) + 1e-9)


def evidence(T, V, P):
    dp, dm = P - V, V - P
    n = sum(dp.values()) + sum(dm.values())
    delta = 0.0 if n == 0 else (sum(min(T[k], c) for k, c in dp.items()) - sum(min(T[k], c) for k, c in dm.items())) / n
    return [cos(T, P) - cos(T, V), delta, float(n == 0)]


_CONSTS = {}


def consts_for(key):
    if not _CONSTS:
        _CONSTS.update(json.load(open(os.path.join(ROOT, "functions", "debian_consts.json"))))
    return Counter(_CONSTS.get(key, {}))


_ANCH = {}
USE_ANCHORS = os.environ.get("PS_ANCHORS", "1") == "1"


def anchors_for(key):
    if not _ANCH:
        _ANCH.update(json.load(open(os.path.join(ROOT, "functions", "debian_anchors.json"))))
    return [tuple(x) for x in _ANCH.get(key, [])]


def triplet_features(t, bags):
    """All explicit evidence features of one benchmark triplet."""
    f = evidence(bags[t["target"]], bags[t["vref"]], bags[t["pref"]]) + \
        evidence(consts_for(t["target"]), consts_for(t["vref"]), consts_for(t["pref"]))
    if USE_ANCHORS:
        from anchors import anchor_features
        f += anchor_features(anchors_for(t["target"]), anchors_for(t["vref"]), anchors_for(t["pref"]), cap=(80, 120))
    if USE_VEX:
        for part in ("ops", "cmpc"):
            f += evidence(vex_for(t["target"], part), vex_for(t["vref"], part), vex_for(t["pref"], part))
    return f


_VEX = {}
USE_VEX = os.environ.get("PS_VEX", "0") == "1"


def vex_for(key, part):
    if not _VEX:
        _VEX.update(json.load(open(os.path.join(ROOT, "functions", "debian_vex.json"))))
    return Counter(_VEX.get(key, {}).get(part, {}))


LEARNER = os.environ.get("PS_LEARNER", "hgb")


def fit(X, y):
    if LEARNER == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    # gradient-boosted trees capture interactions between the evidence streams,
    # e.g. anchor evidence matters most when the instruction bags do not change
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0).fit(X, y)


def main():
    for config in sys.argv[1:]:
        ref = CONFIGS[config]["ref"]
        trip = list(csv.DictReader(open(os.path.join(ROOT, "labels", f"triplets_{ref}.csv"))))
        bags = bags_for({t[k] for t in trip for k in ("target", "vref", "pref")})
        feat = {(t["cve"], t["arch"], t["version"], t["function"]): triplet_features(t, bags) for t in trip}
        rows = []
        for k in range(5):
            for r in csv.DictReader(open(os.path.join(RUNS, config, f"fold{k}.csv"))):
                if r["split"] != "test":
                    continue
                s = min(max(float(r["score"]), 1e-5), 1 - 1e-5)
                learned = [] if os.environ.get("PS_NOLEARNED") == "1" else [np.log(s / (1 - s))]
                r["fold"], r["x"] = k, learned + feat[(r["cve"], r["arch"], r["version"], r["function"])]
                rows.append(r)
        X = np.array([r["x"] for r in rows])
        y = np.array([int(r["label"]) for r in rows])
        fold = np.array([r["fold"] for r in rows])
        out = os.path.join(RUNS, f"{config}+fused" + os.environ.get("PS_SUFFIX", ""))
        os.makedirs(out, exist_ok=True)
        cols = ["split", "cve", "project", "function", "arch", "version", "kind", "size", "label", "score"]
        for k in range(5):
            with open(os.path.join(out, f"fold{k}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(cols)
                m = fit(X[fold != k], y[fold != k])
                idx = np.where(fold == k)[0]
                for i, pi in zip(idx, m.predict_proba(X[idx])[:, 1]):
                    w.writerow(["test"] + [rows[i][c] for c in cols[1:-1]] + [f"{pi:.6f}"])
                for j in range(5):
                    if j == k:
                        continue
                    sel = (fold != k) & (fold != j)
                    mj = fit(X[sel], y[sel])
                    idx = np.where(fold == j)[0]
                    p = mj.predict_proba(X[idx])[:, 1]
                    for i, pi in zip(idx, p):
                        w.writerow(["xcal"] + [rows[i][c] for c in cols[1:-1]] + [f"{pi:.6f}"])
        if os.environ.get("PS_IMPORTANCE") == "1":
            # permutation importance: mean AUROC drop on held-out folds when one feature is shuffled
            from sklearn.metrics import roc_auc_score
            rng = np.random.default_rng(0)
            drop = np.zeros(X.shape[1])
            for k in range(5):
                m = fit(X[fold != k], y[fold != k])
                Xt, yt = X[fold == k], y[fold == k]
                base = roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
                for f in range(X.shape[1]):
                    Xp = Xt.copy()
                    Xp[:, f] = rng.permutation(Xp[:, f])
                    drop[f] += (base - roc_auc_score(yt, m.predict_proba(Xp)[:, 1])) / 5
            print(config, "permutation importance (AUROC drop):", np.round(drop, 3).tolist())


if __name__ == "__main__":
    main()
