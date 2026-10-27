"""Constant anchors of every benchmark function: immediates and displacements
<= 0xFFFF outside call/jump instructions, as a multiset. Constants are much
more stable across compilers and optimisation levels than whole instructions.
Output: data/functions/debian_consts.json  {key: {value: count}}"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

from elftools.elf.elffile import ELFFile

from extract_functions import disassembler, read_function

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EXT = os.path.join(ROOT, "extracted", "debian")
NUM = re.compile(r"0x[0-9a-fA-F]+|\b\d+\b")


def const_bag(md, code, addr):
    c = Counter()
    for i in md.disasm(code, addr):
        if i.mnemonic.startswith(("call", "j", "b", "bl", "jal")) and not i.mnemonic.startswith(("bic", "bfi", "bfc")):
            continue
        for t in NUM.findall(i.op_str):
            v = int(t, 16) if t.startswith("0x") else int(t)
            if v <= 0xFFFF:
                c[str(v)] += 1
    return c


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "labels", "debian_pairs.csv"))))
    by_elf = defaultdict(set)
    for r in rows:
        by_elf[r["elf"]].add((int(r["addr"]), int(r["size"])))
    out = {}
    for rel, funcs in sorted(by_elf.items()):
        with open(os.path.join(EXT, rel), "rb") as fh:
            elf = ELFFile(fh)
            m, bits, little = elf.get_machine_arch(), elf.elfclass, elf.little_endian
            for addr, size in funcs:
                thumb = m == "ARM" and addr & 1
                a = addr & ~1 if thumb else addr
                code = read_function(elf, fh, a, size)
                out[f"{rel}|{addr}"] = dict(const_bag(disassembler(m, bits, little, thumb), code, a)) if code else {}
    json.dump(out, open(os.path.join(ROOT, "functions", "debian_consts.json"), "w"))
    print("functions", len(out))


if __name__ == "__main__":
    main()
