"""Turn labelled pairs into (target, vulnerable reference, patched reference)
triplets and CVE-grouped folds.

References are always amd64 builds of the same function for the same CVE:
the last vulnerable and first patched build of the target's upstream release
series (the fix pair), independent of the target's Debian revision. This mirrors an auditor who compiles
reference builds on an x86 workstation and checks ARM/MIPS firmware.

Folds: 5 CVE-grouped folds stratified by project; in fold k the test set is
fold k, the calibration set is ~15 % of the remaining CVEs, the rest trains.
Output: data/labels/triplets.csv, data/labels/folds.json
"""
import csv
import json
import os
import random
from collections import defaultdict

from debian.debian_support import Version

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LAB = os.path.join(ROOT, "labels")


def build(ref_arch):
    """ref_arch = "amd64": x86-64 reference builds for every target;
    ref_arch = "same": reference builds of the target's own architecture."""
    with open(os.path.join(LAB, "debian_pairs.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["key"] = f"{r['elf']}|{r['addr']}"
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cve"], r["function"])].append(r)
    trip, dropped = [], 0
    for (cve, func), rs in groups.items():
      for arch in sorted({r["arch"] for r in rs}) if ref_arch == "same" else ["amd64"]:
        refs = sorted([r for r in rs if r["arch"] == arch], key=lambda r: Version(r["version"]))
        vul = [r for r in refs if r["label"] == "vuln"]
        pat = [r for r in refs if r["label"] == "patched"]

        def fix_pair(series):
            """Last vulnerable and first patched reference build of a release series:
            the builds an analyst obtains around the fix commit. Chosen without
            looking at the target's Debian revision, which firmware does not expose."""
            v = [r for r in vul if r["series"] == series] or vul
            p = [r for r in pat if r["series"] == series] or pat
            if not v or not p:
                return None, None
            vv = v[-1]
            later = [r for r in p if Version(r["version"]) > Version(vv["version"])]
            return vv, (later[0] if later else p[0])

        pairs = {s: fix_pair(s) for s in {r["series"] for r in rs}}
        for t in (rs if ref_arch == "amd64" else [r for r in rs if r["arch"] == arch]):
            v, p = pairs[t["series"]]
            if v is not None and t["key"] in (v["key"], p["key"]):
                v = p = None  # a reference build is never its own target
            if v is None or p is None:
                dropped += 1
                continue
            trip.append([cve, t["project"], func, t["arch"], t["version"], t["suite"], t["series"],
                         t["label"], t["kind"], t["size"], t["key"], v["key"], v["version"], p["key"], p["version"]])
    with open(os.path.join(LAB, f"triplets_{'x86' if ref_arch == 'amd64' else 'same'}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cve", "project", "function", "arch", "version", "suite", "series", "label", "kind",
                    "size", "target", "vref", "vref_version", "pref", "pref_version"])
        w.writerows(trip)
    print(ref_arch, "triplets", len(trip), "dropped (no reference pair)", dropped, "CVEs", len({t[0] for t in trip}))
    return trip


def main():
    trip = build("amd64")
    same = build("same")
    # CVE-grouped, project-stratified folds
    rnd = random.Random(0)
    by_proj = defaultdict(list)
    for c in sorted({t[0] for t in trip}):
        by_proj[next(t[1] for t in trip if t[0] == c)].append(c)
    fold_of = {}
    k = 0
    for proj, cs in sorted(by_proj.items()):
        rnd.shuffle(cs)
        for c in cs:
            fold_of[c] = k % 5
            k += 1
    folds = []
    for f in range(5):
        test = sorted(c for c, ff in fold_of.items() if ff == f)
        rest = sorted(c for c, ff in fold_of.items() if ff != f)
        rnd2 = random.Random(100 + f)
        rnd2.shuffle(rest)
        n_cal = max(3, round(0.15 * len(rest)))
        folds.append({"fold": f, "test": test, "calib": sorted(rest[:n_cal]), "train": sorted(rest[n_cal:])})
    json.dump(folds, open(os.path.join(LAB, "folds.json"), "w"), indent=1)
    print("folds over", len(fold_of), "CVEs; same-arch CVEs outside the folds:",
          sorted({t[0] for t in same} - set(fold_of)))
    for f in folds:
        print("fold", f["fold"], {k2: len(f[k2]) for k2 in ("train", "calib", "test")})


if __name__ == "__main__":
    main()
