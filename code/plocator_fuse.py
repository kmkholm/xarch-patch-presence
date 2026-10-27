"""Final PLocator numbers with the full evidence fusion, for the variants
evaluated here.

Evidence per case: instruction and constant evidence (fuse.evidence), anchor
evidence over the whole function (anchors.anchor_features), and, when the
patch file is available, patch-region anchor evidence: the source lines
deleted and added by the patch are mapped to instructions of the two gcc -O0
reference builds through their DWARF line tables (patchmap.py, after PLocator
Sec. 4.2.1). Calls to imported functions are named through the PLT and
internal callees carry a constant fingerprint.

zero-shot : learned scores of the Debian fold models (results/plocator_rows.csv),
            fusion fitted on the Debian out-of-fold predictions with the
            features that exist for Debian (no patch-region evidence).
trained   : learned scores of the CVE-grouped cross-validation on PLocator
            (results/plocator_cv_rows.csv), fusion cross-fitted over the same
            CVE folds (fold k fitted on the other four).
Output: results/plocator_final.json
"""
import csv
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from elftools.elf.elffile import ELFFile

import patchmap as pm
import plocator_eval as pe
from anchors import anchor_features, anchor_sequence, callee_fingerprint, plt_map
from extract_constants import const_bag
from extract_functions import disassembler, read_function, tokens
from fuse import bags_for, evidence, fit, triplet_features

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PDIR = os.path.join(pe.PL, "repo", "org_data", "patches")


def describe(rel, addr, size):
    """Instruction bag, constant bag and anchor sequence of one function."""
    with open(os.path.join(pe.BIN, rel), "rb") as fh:
        elf = ELFFile(fh)
        md = disassembler(elf.get_machine_arch(), elf.elfclass, elf.little_endian, False)
        code = read_function(elf, fh, addr, size)
        anc = anchor_sequence(md, code, addr, plt=plt_map(elf), callee=lambda t: callee_fingerprint(md, elf, fh, t))
        return Counter(" ".join(x) for x in tokens(md, code, addr)), const_bag(md, code, addr), anc


def patch_regions(cve, func, ref, proj):
    """Anchor paths of the patched hunks in the two reference builds."""
    dirs = [d for d in os.listdir(PDIR) if d.startswith(cve + "_")]
    if not dirs:
        return ([], []), ([], [])
    vtag, ftag = ref["vul"].split("/")[3], ref["fix"].split("/")[3]
    ch = pm.changed_lines(os.path.join(PDIR, dirs[0]), os.path.join(pe.PL, "src", proj), vtag, ftag)
    vp, vq, fp, fq = [], [], [], []
    for base, (de, vc, ad, fc) in ch.items():
        a, sz = pe.sym(ref["vul"])[0][func]
        p1, q1, _ = pm.region(os.path.join(pe.BIN, ref["vul"].replace(".strip", "")), a, sz, de, vc, base)
        a, sz = pe.sym(ref["fix"])[0][func]
        p2, q2, _ = pm.region(os.path.join(pe.BIN, ref["fix"].replace(".strip", "")), a, sz, ad, fc, base)
        vp += p1
        vq += q1
        fp += p2
        fq += q2
    return (vp, vq), (fp, fq)


def items_with_evidence():
    rows = list(csv.DictReader(open(os.path.join(pe.PL, "repo", "org_data", "dataset_wo_irr.csv"), encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cve"], r["vul_func"])].append(r)
    out = []
    for (cve, func), rs in groups.items():
        ref = {lab: next((r["rel_bin_path"] for r in rs if r["label"] == lab and r["compiler"] == "gcc"
                          and r["optim"] == "O0"), None) for lab in ("vul", "fix")}
        if None in ref.values():
            continue
        R = {}
        for lab, rel in ref.items():
            nm = pe.sym(rel)[0]
            if func in nm:
                R[lab] = describe(rel, *nm[func])
        if len(R) < 2:
            continue
        vreg, freg = patch_regions(cve, func, ref, rs[0]["oss"])
        for r in rs:
            addr = int(r["tgt_func"].replace("sub_", ""), 16)
            size = pe.sym(r["rel_bin_path"])[1].get(addr)
            if not size:
                continue
            T = describe(r["rel_bin_path"], addr, size)
            st = "Same" if (r["compiler"], r["optim"]) == ("gcc", "O0") else "XC" if r["compiler"] == "clang" else "XO"
            base = evidence(T[0], R["vul"][0], R["fix"][0]) + evidence(T[1], R["vul"][1], R["fix"][1])
            whole = anchor_features(T[2], R["vul"][2], R["fix"][2], cap=(80, 120))
            full = anchor_features(T[2], R["vul"][2], R["fix"][2], vreg, freg)
            out.append({"cve": cve, "function": func, "setting": st, "label": int(r["label"] == "fix"),
                        "debian_like": base + whole, "full": base + full})
    return out


def acc_table(rows, score):
    res = {}
    for st in ("Same", "XO", "XC"):
        rs = [r for r in rows if r["setting"] == st]
        y = np.array([r["label"] for r in rs])
        p = np.array([score(r) for r in rs]) > 0.5
        tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
        res[st] = {"accuracy": float((p == y).mean()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                   "n": len(rs)}
    res["Average"] = {k: float(np.mean([res[s][k] for s in ("Same", "XO", "XC")])) for k in ("accuracy", "precision", "recall")}
    return res


def main():
    items = items_with_evidence()
    lg = lambda s: float(np.log(min(max(s, 1e-5), 1 - 1e-5) / (1 - min(max(s, 1e-5), 1 - 1e-5))))
    zs = list(csv.DictReader(open(os.path.join(RES, "plocator_rows.csv"))))
    cv = list(csv.DictReader(open(os.path.join(RES, "plocator_cv_rows.csv"))))
    assert len(zs) == len(cv) == len(items)
    for it, a, b in zip(items, zs, cv):
        assert it["cve"] == a["cve"] == b["cve"] and it["setting"] == a["setting"] == b["setting"]
        it["zs"], it["cv"] = float(a["learned"]), float(b["learned"])
    # zero-shot: fusion weights from the Debian out-of-fold predictions (same feature set)
    trip = list(csv.DictReader(open(os.path.join(pe.ROOT, "labels", "triplets_same.csv"))))
    bags = bags_for({t[k] for t in trip for k in ("target", "vref", "pref")})
    feat = {(t["cve"], t["arch"], t["version"], t["function"]): triplet_features(t, bags) for t in trip}
    X, y = [], []
    for k in range(5):
        for r in csv.DictReader(open(os.path.join(HERE, "..", "runs", "full", f"fold{k}.csv"))):
            if r["split"] == "test":
                X.append([lg(float(r["score"]))] + feat[(r["cve"], r["arch"], r["version"], r["function"])])
                y.append(int(r["label"]))
    lr = fit(np.array(X), np.array(y))
    for it in items:
        it["zs_fused"] = float(lr.predict_proba([[lg(it["zs"])] + it["debian_like"]])[0, 1])
    # trained: cross-fitted over the CVE folds of plocator_cv.py
    cves = sorted({it["cve"] for it in items})
    random.Random(0).shuffle(cves)
    fold_of = {c: i % 5 for i, c in enumerate(cves)}
    yc = np.array([it["label"] for it in items])
    fo = np.array([fold_of[it["cve"]] for it in items])
    for key, cols in (("cv_fused", lambda it: [lg(it["cv"])] + it["full"]),
                      ("cv_evidence_only", lambda it: it["full"])):
        Xc = np.array([cols(it) for it in items])
        for k in range(5):
            m = fit(Xc[fo != k], yc[fo != k])
            for i in np.where(fo == k)[0]:
                items[i][key] = float(m.predict_proba(Xc[i:i + 1])[0, 1])
    out = {"zero_shot_learned": acc_table(items, lambda r: r["zs"]),
           "zero_shot_fused": acc_table(items, lambda r: r["zs_fused"]),
           "trained_learned": acc_table(items, lambda r: r["cv"]),
           "trained_evidence_anchors": acc_table(items, lambda r: r["cv_evidence_only"]),
           "trained_fused": acc_table(items, lambda r: r["cv_fused"]),
           "published_accuracy": {"BinXray": [49.3, 13.9, 4.8, 22.7], "Robin": [72.6, 54.1, 48.0, 58.2],
                                  "PS3": [65.1, 45.4, 43.2, 51.2], "PLocator": [89.0, 80.1, 82.9, 84.0]}}
    json.dump(out, open(os.path.join(RES, "plocator_final.json"), "w"), indent=1)
    for k, v in out.items():
        if k != "published_accuracy":
            print(f"{k:26s}", {s: round(100 * v[s]["accuracy"], 1) for s in ("Same", "XO", "XC", "Average")})


if __name__ == "__main__":
    main()
