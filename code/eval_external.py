"""External tests: stripped OpenWrt router builds and real firmware
(Cisco-Talos Dataset-Vulnerability).

For each target, the score comes from the fold model in which the CVE was a
test CVE (never seen in training); CVEs outside the benchmark are scored by the
mean of all fold models. Conformal thresholds are those of the same fold,
fitted by CVE-level conformal risk control on the out-of-fold benchmark
predictions of the matching Debian architecture (target CVE left out).
Usage: python eval_external.py [--config full]
Output: results/external_{openwrt,cisco}.csv and results/external_summary.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import ROOT, build_vocab, load_functions, pad_batch
from evaluate import crc_threshold, load
from anchors import anchor_features
from fuse import anchors_for, bags_for, consts_for, evidence, fit, triplet_features
from collections import Counter
from model import FunctionEncoder, PatchHead

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "..", "runs"), os.path.join(HERE, "..", "results")
ARCH_MAP = {"mips_24kc": "mipsel", "mipsel_24kc": "mipsel", "arm_cortex-a7_neon-vfpv4": "armhf",
            "aarch64_cortex-a53": "arm64", "x86_64": "amd64", "arm32": "armel", "mips32": "mipsel"}


_CAL = {}


def thresholds(config, cve, arch, alpha, beta=0.10):
    """CVE-level conformal risk control on the out-of-fold benchmark predictions
    of the matching Debian architecture, leaving out the target's own CVE."""
    key = (config, cve, arch, alpha)
    if key not in _CAL:
        rows = [r for r in load(config) if r["split"] == "test" and r["arch"] == arch and r["cve"] != cve]
        by_v, by_p = defaultdict(list), defaultdict(list)
        for r in rows:
            (by_p if r["label"] == 1 else by_v)[r["cve"]].append(r["score"])
        _CAL[key] = (crc_threshold(list(by_v.values()), alpha, "high"),
                     crc_threshold(list(by_p.values()), beta, "low"))
    return _CAL[key]


def verdict(s, tp, tv):
    pat, vul = s > tp, s < tv
    return "patched" if pat and not vul else "vuln" if vul and not pat else "abstain"


def load_model(config, fold, vocab_n, dev):
    ck = torch.load(os.path.join(RUNS, config, f"fold{fold}.pt"), map_location=dev)
    enc = FunctionEncoder(vocab_n, gate=ck["gate"]).to(dev).eval()
    enc.load_state_dict(ck["enc"])
    head = PatchHead(refs=ck["refs"]).to(dev).eval()
    head.load_state_dict(ck["head"])
    return enc, head


@torch.no_grad()
def score(enc, head, funcs, rows, dev):
    out = []
    for i in range(0, len(rows), 64):
        b = rows[i:i + 64]
        z = [enc(pad_batch([funcs[r[k]] for r in b]).to(dev)) for k in ("target", "vref", "pref")]
        out += torch.sigmoid(head(*z).float()).cpu().tolist()
    return out


def refs_for(trip):
    """Debian fix pair per (cve, function, series, architecture), as in the benchmark."""
    ref = {}
    for t in trip:
        ref.setdefault((t["cve"], t["function"], t["series"], t["arch"]), (t["vref"], t["pref"]))
        ref.setdefault((t["cve"], t["function"], "*", t["arch"]), (t["vref"], t["pref"]))
    return ref


def metrics(rows):
    y = np.array([r["label"] == "patched" for r in rows])
    s = np.array([r["score"] for r in rows])
    m = {"n": len(rows), "n_vuln": int((~y).sum()), "n_patched": int(y.sum()),
         "auroc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
         "acc@0.5": float(((s > 0.5) == y).mean())}
    for a in ("0.05", "0.1"):
        v = [r[f"verdict_{a}"] for r in rows]
        dec = [(vv, yy) for vv, yy in zip(v, y) if vv != "abstain"]
        m[f"alpha{a}"] = {
            "false_patched_rate": float(np.mean([vv == "patched" for vv, yy in zip(v, y) if not yy])) if (~y).any() else None,
            "coverage": len(dec) / len(v),
            "decided_accuracy": float(np.mean([(vv == "patched") == yy for vv, yy in dec])) if dec else None}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(RES, exist_ok=True)
    vocab = build_vocab()
    folds = json.load(open(os.path.join(ROOT, "labels", "folds.json")))
    fold_of = {c: f["fold"] for f in folds for c in f["test"]}
    ref_mode = "x86" if a.config == "x86ref" else "same"
    trip = list(csv.DictReader(open(os.path.join(ROOT, "labels", f"triplets_{ref_mode}.csv"))))
    ref = refs_for(trip)

    # ---- OpenWrt targets located through the dynamic symbol table
    ow = [r for r in csv.DictReader(open(os.path.join(ROOT, "labels", "openwrt_located.csv"))) if r["how"] == "dynsym"]
    rows = []
    for r in ow:
        ra = "amd64" if ref_mode == "x86" else ARCH_MAP[r["arch"]]
        k = ref.get((r["cve"], r["function"], r["series"], ra)) or ref.get((r["cve"], r["function"], "*", ra))
        if k is None:
            continue
        rows.append(dict(r, target=f"{r['elf']}|{r['addr']}", vref=k[0], pref=k[1], set="openwrt",
                         device=r["arch"]))
    # ---- real firmware
    for r in csv.DictReader(open(os.path.join(ROOT, "labels", f"cisco_triplets_{ref_mode}.csv"))):
        rows.append(dict(r, set="cisco", series="1.0.2"))
    keys = {r[k] for r in rows for k in ("target", "vref", "pref")}
    funcs, bag, cst, anc = {}, {}, {}, {}
    for fn in ("debian_funcs.jsonl", "openwrt_funcs.jsonl", "cisco_funcs.jsonl"):
        f, _ = load_functions(fn, vocab, keep=keys)
        funcs.update(f)
        for line in open(os.path.join(ROOT, "functions", fn)):
            d = json.loads(line)
            if d["key"] in keys:
                bag[d["key"]] = Counter(" ".join(i) for i in d["insns"])
                cst[d["key"]] = Counter(d["consts"]) if "consts" in d else consts_for(d["key"])
                anc[d["key"]] = [tuple(x) for x in d["anchors"]] if "anchors" in d else anchors_for(d["key"])
    rows = [r for r in rows if all(r[k] in funcs for k in ("target", "vref", "pref"))]
    print("targets", len(rows), {s: sum(r["set"] == s for r in rows) for s in ("openwrt", "cisco")})

    avail = [k for k in range(5) if os.path.exists(os.path.join(RUNS, a.config, f"fold{k}.pt"))]
    per_fold = defaultdict(list)
    for i, r in enumerate(rows):
        f = fold_of.get(r["cve"])
        for k in ([f] if f in avail else avail):
            per_fold[k].append(i)
    scores = defaultdict(list)
    for k in avail:
        enc, head = load_model(a.config, k, len(vocab), dev)
        idx = per_fold[k]
        for i, s in zip(idx, score(enc, head, funcs, [rows[i] for i in idx], dev)):
            scores[i].append((k, s))
        del enc, head
        torch.cuda.empty_cache()
    lg = lambda v: float(np.log(min(max(v, 1e-5), 1 - 1e-5) / (1 - min(max(v, 1e-5), 1 - 1e-5))))
    dbags = bags_for({t[k] for t in trip for k in ("target", "vref", "pref")})
    dfeat = {(t["cve"], t["arch"], t["version"], t["function"]): triplet_features(t, dbags) for t in trip}
    X, Y = [], []
    for r in load(a.config):
        if r["split"] == "test":
            X.append([lg(r["score"])] + dfeat[(r["cve"], r["arch"], r["version"], r["function"])])
            Y.append(r["label"])
    lr = fit(np.array(X), np.array(Y))
    out = []
    for i, r in enumerate(rows):
        if not scores[i]:
            continue
        s_learned = float(np.mean([x[1] for x in scores[i]]))
        ev = evidence(bag[r["target"]], bag[r["vref"]], bag[r["pref"]]) + evidence(cst[r["target"]], cst[r["vref"]], cst[r["pref"]]) + anchor_features(anc[r["target"]], anc[r["vref"]], anc[r["pref"]], cap=(80, 120))
        s = float(lr.predict_proba([[lg(s_learned)] + ev])[0, 1])
        arch = ARCH_MAP[r["arch"]]
        rr = {"set": r["set"], "device": r["device"], "cve": r["cve"], "function": r["function"],
              "arch": r["arch"], "upstream": r["upstream"], "label": r["label"], "score": s, "learned": s_learned,
              "unseen_cve": r["cve"] in fold_of, "kind": r.get("kind", "")}
        for al in (0.05, 0.10):
            tp, tv = thresholds(a.config + "+fused", r["cve"], arch, al)
            rr[f"verdict_{al:g}" if al != 0.10 else "verdict_0.1"] = verdict(s, tp, tv)
        out.append(rr)
    summary = {}
    for st in ("openwrt", "cisco"):
        rs = [r for r in out if r["set"] == st]
        if not rs:
            continue
        with open(os.path.join(RES, f"external_{st}_{a.config}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
        summary[st] = metrics(rs)
        if st == "openwrt":
            summary[st]["by_arch"] = {ar: metrics([r for r in rs if r["arch"] == ar]) for ar in sorted({r["arch"] for r in rs})}
    summary["models"] = avail
    json.dump(summary, open(os.path.join(RES, f"external_summary_{a.config}.json"), "w"), indent=1)
    print(json.dumps({k: (v if k == "models" else {kk: vv for kk, vv in v.items() if kk != "by_arch"})
                      for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
