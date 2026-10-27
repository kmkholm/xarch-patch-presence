"""Locate the target functions in the stripped (sstrip) OpenWrt builds.

OpenWrt removes all section headers, but the dynamic symbol table is still
reachable through the PT_DYNAMIC segment, so exported functions can be found
by name. Pairs whose function is not exported are written to a list for
function recovery with angr (code/recover_functions.py) and later located by
embedding search. Output: data/labels/openwrt_located.csv, logs/recover_openwrt.txt
"""
import csv
import os
from collections import Counter, defaultdict

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
EXT = os.path.join(DATA, "extracted", "openwrt")


def dyn_funcs(p):
    with open(p, "rb") as fh:
        e = ELFFile(fh)
        for seg in e.iter_segments():
            if isinstance(seg, DynamicSegment):
                return {s.name: (int(s["st_value"]), int(s["st_size"])) for s in seg.iter_symbols()
                        if s["st_info"]["type"] == "STT_FUNC" and s["st_value"]}
    return {}


def main():
    deb = list(csv.DictReader(open(os.path.join(DATA, "labels", "debian_pairs.csv"))))
    lib = defaultdict(Counter)
    for r in deb:
        lib[r["function"]][os.path.basename(r["elf"]).split(".so")[0]] += 1
    flib = {f: c.most_common(1)[0][0] for f, c in lib.items()}
    pairs = list(csv.DictReader(open(os.path.join(DATA, "labels", "openwrt_pairs.csv"))))
    cache, out, missing = {}, [], set()
    for r in pairs:
        for f in r["functions"].split(";"):
            for e in r["elfs"].split(";"):
                if os.path.basename(e).split(".so")[0] != flib.get(f):
                    continue
                if e not in cache:
                    cache[e] = dyn_funcs(os.path.join(EXT, e))
                if f in cache[e]:
                    a, sz = cache[e][f]
                    out.append([r["cve"], r["project"], r["version"], r["upstream"], r["series"], r["arch"], e, f,
                                a, sz, "dynsym", r["label"], r["kind"]])
                else:
                    out.append([r["cve"], r["project"], r["version"], r["upstream"], r["series"], r["arch"], e, f,
                                "", "", "search", r["label"], r["kind"]])
                    missing.add(e)
    with open(os.path.join(DATA, "labels", "openwrt_located.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cve", "project", "version", "upstream", "series", "arch", "elf", "function", "addr", "size",
                    "how", "label", "kind"])
        w.writerows(out)
    with open(os.path.join(HERE, "..", "logs", "recover_openwrt.txt"), "w") as fh:
        for e in sorted(missing):
            fh.write(f"openwrt/{e}\t{os.path.join(EXT, e)}\n")
    c = Counter((x[10], x[11]) for x in out)
    print("located", c, "ELFs needing recovery", len(missing))
    print(Counter(os.path.basename(e).split(".so")[0] for e in missing))


if __name__ == "__main__":
    main()
