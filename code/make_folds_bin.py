"""Binary-level folds (the split used by most earlier patch presence studies):
all targets built from the same Debian source version fall into the same
fold, while CVEs appear in both training and test data. 15 % of the training
versions form the calibration split. Output: data/labels/folds_bin.json"""
import csv
import json
import os
import random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
trip = list(csv.DictReader(open(os.path.join(ROOT, "labels", "triplets_same.csv"))))
vers = sorted({(t["project"], t["version"]) for t in trip})
rnd = random.Random(0)
rnd.shuffle(vers)
fold_of = {v: i % 5 for i, v in enumerate(vers)}
folds = []
for f in range(5):
    test = sorted(f"{p}|{v}" for (p, v), ff in fold_of.items() if ff == f)
    rest = sorted(f"{p}|{v}" for (p, v), ff in fold_of.items() if ff != f)
    random.Random(100 + f).shuffle(rest)
    n_cal = round(0.15 * len(rest))
    folds.append({"fold": f, "test": test, "calib": sorted(rest[:n_cal]), "train": sorted(rest[n_cal:])})
json.dump(folds, open(os.path.join(ROOT, "labels", "folds_bin.json"), "w"), indent=1)
print({f["fold"]: (len(f["train"]), len(f["calib"]), len(f["test"])) for f in folds})
