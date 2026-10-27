"""Non-learned reference point: instruction-overlap score.

score = (cos(t, r_p) - cos(t, r_v) + 1) / 2 over bags of normalised
instructions. Written in the same format as the fine-tuning runs so that
evaluate.py applies the same metrics and conformal rule.
Usage: python baseline_overlap.py same|x86
Output: runs/overlap_<ref>/fold<k>.csv
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

from common import ROOT

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def cos(a, b):
    num = sum(a[k] * b[k] for k in a if k in b)
    return num / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values())) + 1e-9)


def main():
    ref = sys.argv[1]
    trip = list(csv.DictReader(open(os.path.join(ROOT, "labels", f"triplets_{ref}.csv"))))
    keys = {t[k] for t in trip for k in ("target", "vref", "pref")}
    bags = {}
    for line in open(os.path.join(ROOT, "functions", "debian_funcs.jsonl")):
        d = json.loads(line)
        if d["key"] in keys:
            bags[d["key"]] = Counter(" ".join(i) for i in d["insns"])
    folds = json.load(open(os.path.join(ROOT, "labels", "folds.json")))
    out = os.path.join(RUNS, f"overlap_{ref}")
    os.makedirs(out, exist_ok=True)
    for f in folds:
        split = {c: s for s in ("calib", "test") for c in f[s]}
        with open(os.path.join(out, f"fold{f['fold']}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["split", "cve", "project", "function", "arch", "version", "kind", "size", "label", "score"])
            for t in trip:
                if t["cve"] not in split:
                    continue
                s = (cos(bags[t["target"]], bags[t["pref"]]) - cos(bags[t["target"]], bags[t["vref"]]) + 1) / 2
                w.writerow([split[t["cve"]], t["cve"], t["project"], t["function"], t["arch"], t["version"],
                            t["kind"], t["size"], int(t["label"] == "patched"), f"{s:.6f}"])
    print("done", ref)


if __name__ == "__main__":
    main()
