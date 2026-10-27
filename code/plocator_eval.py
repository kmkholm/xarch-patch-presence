"""Direct comparison on the PLocator benchmark (TOSEM 2026) with its own protocol.

PLocator's 1,090 test cases (73 CVEs, OpenSSL / libxml2 / tcpdump / FreeType,
x86-32) are targets built with gcc -O0..-O3 and clang -O0 from the last
vulnerable and the first fixed version. References are the gcc -O0 builds of
the two versions, and results are reported for three settings:
  Same  targets gcc -O0
  XO    targets gcc -O1..-O3 (cross-optimisation)
  XC    targets clang -O0 (cross-compiler)
The model is applied without any training on this benchmark: the learned
stream comes from the Debian fold model in which the CVE was a test CVE (all
five fold models are averaged for CVEs outside the Debian benchmark), and the
fusion weights are fitted on the Debian out-of-fold predictions.
Output: results/plocator_{rows.csv,summary.json}
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from elftools.elf.elffile import ELFFile
from sklearn.metrics import roc_auc_score

from common import ROOT, build_vocab, pad_batch, tensorise
from eval_external import load_model
from extract_functions import disassembler, read_function, tokens
from fuse import bags_for, evidence, fit
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PL = os.path.join(ROOT, "plocator")
BIN = os.path.join(PL, "bin")
RES = os.path.join(HERE, "..", "results")


def symbols(path):
    """name -> (addr, size) and addr -> size from an unstripped ELF."""
    by_name, by_addr = {}, {}
    with open(path, "rb") as fh:
        sec = ELFFile(fh).get_section_by_name(".symtab")
        for s in sec.iter_symbols():
            if s["st_info"]["type"] == "STT_FUNC" and s["st_value"] and s["st_size"]:
                by_name.setdefault(s.name, (s["st_value"], s["st_size"]))
                by_addr[s["st_value"]] = s["st_size"]
    return by_name, by_addr


_SYM = {}


def sym(rel):
    if rel not in _SYM:
        _SYM[rel] = symbols(os.path.join(BIN, rel.replace(".strip", "")))
    return _SYM[rel]


def func_tokens(rel, addr, size):
    with open(os.path.join(BIN, rel), "rb") as fh:
        elf = ELFFile(fh)
        md = disassembler(elf.get_machine_arch(), elf.elfclass, elf.little_endian, False)
        return tokens(md, read_function(elf, fh, addr, size), addr)


def main():
    rows = list(csv.DictReader(open(os.path.join(PL, "repo", "org_data", "dataset_wo_irr.csv"), encoding="utf-8")))
    # reference builds: gcc -O0 of the vulnerable and of the fixed version of each CVE/function
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cve"], r["vul_func"])].append(r)
    items, skipped = [], 0
    for (cve, func), rs in groups.items():
        ref = {}
        for lab in ("vul", "fix"):
            c = [r for r in rs if r["label"] == lab and r["compiler"] == "gcc" and r["optim"] == "O0"]
            if c:
                ref[lab] = c[0]["rel_bin_path"]
        if len(ref) < 2:
            skipped += len(rs)
            continue
        rt = {}
        for lab, rel in ref.items():
            nm = sym(rel)[0]
            if func not in nm:
                break
            rt[lab] = func_tokens(rel, *nm[func])
        if len(rt) < 2:
            skipped += len(rs)
            continue
        for r in rs:
            addr = int(r["tgt_func"].replace("sub_", ""), 16)
            size = sym(r["rel_bin_path"])[1].get(addr)
            if not size:
                skipped += 1
                continue
            ins = func_tokens(r["rel_bin_path"], addr, size)
            setting = "Same" if (r["compiler"], r["optim"]) == ("gcc", "O0") else "XC" if r["compiler"] == "clang" else "XO"
            items.append(dict(cve=cve, function=func, project=r["oss"], setting=setting, compiler=r["compiler"],
                              optim=r["optim"], label=int(r["label"] == "fix"), t=ins, v=rt["vul"], p=rt["fix"]))
    print("cases", len(items), "skipped", skipped)

    vocab = build_vocab()
    dev = "cuda"
    folds = json.load(open(os.path.join(ROOT, "labels", "folds.json")))
    fold_of = {c: f["fold"] for f in folds for c in f["test"]}
    learned = defaultdict(list)
    for k in range(5):
        enc, head = load_model("full", k, len(vocab), dev)
        idx = [i for i, it in enumerate(items) if fold_of.get(it["cve"], k) == k]
        with torch.no_grad():
            for b in range(0, len(idx), 64):
                bi = idx[b:b + 64]
                z = [enc(pad_batch([tensorise(items[i][x], vocab) for i in bi]).to(dev)) for x in ("t", "v", "p")]
                for i, s in zip(bi, torch.sigmoid(head(*z).float()).cpu().tolist()):
                    learned[i].append(s)
        del enc, head
        torch.cuda.empty_cache()

    # fusion weights from the Debian out-of-fold predictions
    trip = list(csv.DictReader(open(os.path.join(ROOT, "labels", "triplets_same.csv"))))
    bags = bags_for({t[k] for t in trip for k in ("target", "vref", "pref")})
    feat = {(t["cve"], t["arch"], t["version"], t["function"]): evidence(bags[t["target"]], bags[t["vref"]], bags[t["pref"]])
            for t in trip}
    X, y = [], []
    for k in range(5):
        for r in csv.DictReader(open(os.path.join(HERE, "..", "runs", "full", f"fold{k}.csv"))):
            if r["split"] == "test":
                s = min(max(float(r["score"]), 1e-5), 1 - 1e-5)
                X.append([np.log(s / (1 - s))] + feat[(r["cve"], r["arch"], r["version"], r["function"])])
                y.append(int(r["label"]))
    lr = fit(np.array(X), np.array(y))

    out = []
    for i, it in enumerate(items):
        s = min(max(float(np.mean(learned[i])), 1e-5), 1 - 1e-5)
        bag = lambda ins: Counter(" ".join(x) for x in ins)
        ev = evidence(bag(it["t"]), bag(it["v"]), bag(it["p"]))
        fused = float(lr.predict_proba([[np.log(s / (1 - s))] + ev])[0, 1])
        out.append({k: it[k] for k in ("cve", "function", "project", "setting", "compiler", "optim", "label")}
                   | {"learned": s, "overlap": (ev[0] + 1) / 2, "fused": fused, "unseen_cve": it["cve"] in fold_of})
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "plocator_rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    summ = {}
    for st in ("Same", "XO", "XC", "All"):
        rs = [r for r in out if st == "All" or r["setting"] == st]
        yy = np.array([r["label"] for r in rs])
        summ[st] = {"n": len(rs)}
        for key in ("learned", "overlap", "fused"):
            ss = np.array([r[key] for r in rs])
            summ[st][key] = {"accuracy": float(((ss > 0.5) == yy).mean()),
                             "auroc": float(roc_auc_score(yy, ss)) if 0 < yy.sum() < len(yy) else None}
    json.dump(summ, open(os.path.join(RES, "plocator_summary.json"), "w"), indent=1)
    for st, v in summ.items():
        print(st, v["n"], {k: (round(v[k]["accuracy"], 3), round(v[k]["auroc"], 3) if v[k]["auroc"] else None)
                           for k in ("learned", "overlap", "fused")})


if __name__ == "__main__":
    main()
