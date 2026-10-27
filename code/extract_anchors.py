"""Anchor sequences (anchors.py) of every Debian benchmark function.
Imported callees are named through the PLT on x86-64; internal callees carry a
constant fingerprint on all architectures.
Output: data/functions/debian_anchors.json  {key: [[type, value], ...]}"""
import csv
import json
import os
from collections import defaultdict

from elftools.elf.elffile import ELFFile

from anchors import anchor_sequence, callee_fingerprint, plt_map
from extract_functions import disassembler, read_function

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EXT = os.path.join(ROOT, "extracted", "debian")


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "labels", "debian_pairs.csv"))))
    by_elf = defaultdict(set)
    for r in rows:
        by_elf[r["elf"]].add((int(r["addr"]), int(r["size"])))
    out = {}
    for n, (rel, funcs) in enumerate(sorted(by_elf.items())):
        with open(os.path.join(EXT, rel), "rb") as fh:
            elf = ELFFile(fh)
            m, bits, little = elf.get_machine_arch(), elf.elfclass, elf.little_endian
            plt = plt_map(elf) if m in ("x64", "x86") else {}
            for addr, size in funcs:
                thumb = m == "ARM" and addr & 1
                a = addr & ~1 if thumb else addr
                md = disassembler(m, bits, little, thumb)
                code = read_function(elf, fh, a, size)
                seq = anchor_sequence(md, code, a, plt=plt, callee=lambda t: callee_fingerprint(md, elf, fh, t & ~1)) if code else []
                out[f"{rel}|{addr}"] = [list(x) for x in seq]
        if n % 200 == 0:
            print(n, len(by_elf), flush=True)
    json.dump(out, open(os.path.join(ROOT, "functions", "debian_anchors.json"), "w"))
    print("functions", len(out))


if __name__ == "__main__":
    main()
