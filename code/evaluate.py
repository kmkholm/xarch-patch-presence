"""Evaluate fine-tuned runs: discrimination, Mondrian conformal decisions, and
a version-string baseline.

Conformal rule (class-conditional, per architecture, fitted on the fold's
calibration CVEs):
  tau_p = conformal (1-alpha) quantile of scores of calibration VULNERABLE pairs
  tau_v = conformal beta quantile of scores of calibration PATCHED pairs
  verdict = patched if s > tau_p ; vulnerable if s < tau_v ; abstain otherwise
so P(verdict = patched | vulnerable) <= alpha under exchangeability.
Confidence intervals: 1,000 bootstrap resamples over CVEs.
Usage: python evaluate.py full xarch nogate ...
Output: results/<config>_summary.json
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "..", "runs"), os.path.join(HERE, "..", "results")
MIN_CAL = 20


def cq(x, level):
    """Finite-sample conformal quantile."""
    x = np.sort(np.asarray(x))
    n = len(x)
    k = min(n - 1, max(0, math.ceil((n + 1) * level) - 1))
    return x[k]


def load(config):
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, config, "fold*.csv"))):
        fold = int(os.path.basename(f)[4:-4])
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                r["fold"], r["label"], r["score"] = fold, int(r["label"]), float(r["score"])
                rows.append(r)
    return rows


def conformal(rows, alpha, beta, test_arches=None):
    out = []
    for fold in sorted({r["fold"] for r in rows}):
        cal = [r for r in rows if r["fold"] == fold and r["split"] == "calib"]
        if not cal:  # runs without a held-out calibration split (e.g. fused scores)
            continue
        tst = [r for r in rows if r["fold"] == fold and r["split"] == "test"
               and (test_arches is None or r["arch"] in test_arches)]
        for arch in sorted({r["arch"] for r in tst}):
            ca = [r for r in cal if r["arch"] == arch]
            cv = [r["score"] for r in ca if r["label"] == 0]
            cp = [r["score"] for r in ca if r["label"] == 1]
            if len(cv) < MIN_CAL:
                cv = [r["score"] for r in cal if r["label"] == 0]
            if len(cp) < MIN_CAL:
                cp = [r["score"] for r in cal if r["label"] == 1]
            tp, tv = cq(cv, 1 - alpha), cq(cp, beta)
            for r in tst:
                if r["arch"] != arch:
                    continue
                pat, vul = r["score"] > tp, r["score"] < tv
                out.append(dict(r, verdict="patched" if pat and not vul else "vuln" if vul and not pat else "abstain"))
    return out


def crc_threshold(units, alpha, side):
    """Conformal risk control (Angelopoulos et al., ICLR 2024) with CVEs as the
    exchangeable units. units: list of score arrays, one per calibration CVE,
    holding the scores of the class whose error is controlled. side="high":
    loss L_j(lam) = fraction of the CVE's scores above lam (false patched);
    side="low": fraction below lam (false vulnerable). Returns the least
    conservative lam with n/(n+1) * mean_j L_j(lam) + 1/(n+1) <= alpha."""
    n = len(units)
    if n == 0:
        return np.inf if side == "high" else -np.inf
    cand = np.unique(np.concatenate([np.asarray(u) for u in units] + [np.array([0.0, 1.0])]))

    def risk(lam):
        if side == "high":
            return np.mean([np.mean(np.asarray(u) > lam) for u in units])
        return np.mean([np.mean(np.asarray(u) < lam) for u in units])

    ok = [lam for lam in cand if n / (n + 1) * risk(lam) + 1 / (n + 1) <= alpha]
    if not ok:
        return np.inf if side == "high" else -np.inf
    return min(ok) if side == "high" else max(ok)


def conformal_crc(rows, alpha, beta, test_arches=None):
    """Cross-conformal, CVE-level risk control. For the test CVEs of fold k, the
    calibration units are the out-of-fold test predictions of all other folds
    (about 68 CVEs), so every CVE is calibrated by models that never trained
    on it. Thresholds are fitted per architecture (Mondrian over CVE units)."""
    out = []
    folds = sorted({r["fold"] for r in rows})
    test_rows = [r for r in rows if r["split"] == "test"]
    for fold in folds:
        # fused runs carry their own calibration scores computed without the test fold
        cal = [r for r in rows if r["fold"] == fold and r["split"] == "xcal"] or \
              [r for r in test_rows if r["fold"] != fold]
        tst = [r for r in test_rows if r["fold"] == fold and (test_arches is None or r["arch"] in test_arches)]
        for arch in sorted({r["arch"] for r in tst}):
            ca = [r for r in cal if r["arch"] == arch]
            by_v, by_p = defaultdict(list), defaultdict(list)
            for r in ca:
                (by_p if r["label"] == 1 else by_v)[r["cve"]].append(r["score"])
            tp = crc_threshold(list(by_v.values()), alpha, "high")
            tv = crc_threshold(list(by_p.values()), beta, "low")
            for r in tst:
                if r["arch"] != arch:
                    continue
                pat, vul = r["score"] > tp, r["score"] < tv
                out.append(dict(r, verdict="patched" if pat and not vul else "vuln" if vul and not pat else "abstain"))
    return out


def cve_false_patched(dec):
    """Mean over CVEs of the fraction of vulnerable pairs reported as patched
    (the quantity bounded by conformal_crc)."""
    by = defaultdict(list)
    for d in dec:
        if d["label"] == 0:
            by[d["cve"]].append(d["verdict"] == "patched")
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def summarise(dec):
    vul = [d for d in dec if d["label"] == 0]
    pat = [d for d in dec if d["label"] == 1]
    decided = [d for d in dec if d["verdict"] != "abstain"]
    correct = [d for d in decided if (d["verdict"] == "patched") == (d["label"] == 1)]
    return {
        "n": len(dec),
        "false_patched_rate": sum(d["verdict"] == "patched" for d in vul) / max(len(vul), 1),
        "false_vuln_rate": sum(d["verdict"] == "vuln" for d in pat) / max(len(pat), 1),
        "coverage": len(decided) / max(len(dec), 1),
        "decided_accuracy": len(correct) / max(len(decided), 1),
    }


def discrimination(rows):
    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows])
    pred = s > 0.5
    tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {"n": len(rows), "auroc": roc_auc_score(y, s) if len(set(y)) > 1 else float("nan"),
            "balanced_acc": (tpr + tnr) / 2, "acc": float((pred == (y == 1)).mean()),
            "false_patched_rate@0.5": float((pred & (y == 0)).sum() / max((y == 0).sum(), 1))}


def version_baseline(rows):
    """Predicts from the upstream version string only: 'backport' pairs are
    called vulnerable although patched; every other kind is called correctly."""
    ok = [r["kind"] != "backport" for r in rows]
    bp = [r for r in rows if r["kind"] == "backport"]
    return {"acc": float(np.mean(ok)), "backport_pairs": len(bp), "backport_acc": 0.0}


def boot(rows, fn, key, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["cve"]].append(r)
    cves = list(by)
    vals = []
    for _ in range(B):
        samp = [r for c in rng.choice(cves, len(cves)) for r in by[c]]
        try:
            vals.append(fn(samp)[key])
        except ValueError:
            pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    os.makedirs(RES, exist_ok=True)
    for config in sys.argv[1:]:
        rows = load(config)
        test = [r for r in rows if r["split"] == "test"]
        arches = None
        if config.split("+")[0] == "xarch":
            arches = {"armel", "armhf", "arm64", "mipsel", "mips64el"}
            test = [r for r in test if r["arch"] in arches]
        summ = {"config": config, "folds": sorted({r["fold"] for r in rows}),
                "discrimination": discrimination(test),
                "auroc_ci": boot(test, discrimination, "auroc"),
                "by_arch": {a: discrimination([r for r in test if r["arch"] == a])
                            for a in sorted({r["arch"] for r in test})},
                "by_kind": {k: discrimination([r for r in test if r["kind"] == k])
                            for k in sorted({r["kind"] for r in test})},
                "version_baseline": version_baseline(test), "conformal": {}}
        # the decision a version scanner cannot make: inside an NVD-affected range,
        # separate still-vulnerable builds from back-ported fixes
        inrange = [r for r in test if r["kind"] in ("vuln", "backport")]
        summ["in_range_vuln_vs_backport"] = discrimination(inrange)
        summ["in_range_auroc_ci"] = boot(inrange, discrimination, "auroc")
        for alpha in (0.05, 0.10):
            dec = conformal(rows, alpha, 0.10, arches)
            s = summarise(dec)
            s["false_patched_ci"] = boot(dec, summarise, "false_patched_rate") if dec else None
            s["coverage_ci"] = boot(dec, summarise, "coverage") if dec else None
            s["by_arch"] = {a: summarise([d for d in dec if d["arch"] == a]) for a in sorted({d["arch"] for d in dec})}
            s["backport"] = summarise([d for d in dec if d["kind"] == "backport"])
            summ["conformal"][str(alpha)] = s
            dc = conformal_crc(rows, alpha, 0.10, arches)
            c2 = summarise(dc)
            c2["cve_false_patched"] = cve_false_patched(dc)
            c2["false_patched_ci"] = boot(dc, summarise, "false_patched_rate")
            c2["coverage_ci"] = boot(dc, summarise, "coverage")
            c2["by_arch"] = {a: dict(summarise([d for d in dc if d["arch"] == a]),
                                     cve_false_patched=cve_false_patched([d for d in dc if d["arch"] == a]))
                             for a in sorted({d["arch"] for d in dc})}
            c2["backport"] = summarise([d for d in dc if d["kind"] == "backport"])
            summ.setdefault("crc", {})[str(alpha)] = c2
        json.dump(summ, open(os.path.join(RES, f"{config}_summary.json"), "w"), indent=1)
        d = summ["discrimination"]
        c = summ["conformal"]["0.05"]
        k = summ["crc"]["0.1"]
        print(f"{config}: AUROC {d['auroc']:.3f} {summ['auroc_ci']} balAcc {d['balanced_acc']:.3f} | "
              f"split a=.05 FPR {c['false_patched_rate']:.3f} cov {c['coverage']:.3f} | CRC a=.10 FPR "
              f"{k['false_patched_rate']:.3f} (CVE-avg {k['cve_false_patched']:.3f}) cov {k['coverage']:.3f} "
              f"acc|decided {c['decided_accuracy']:.3f} | vuln-vs-backport AUROC {summ['in_range_vuln_vs_backport']['auroc']:.3f}")


if __name__ == "__main__":
    main()
