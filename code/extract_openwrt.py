"""Tokenise the OpenWrt target functions located through the dynamic symbol
table (data/labels/openwrt_located.csv, how == dynsym).
Output: data/functions/openwrt_funcs.jsonl"""
import csv
import json
import os
from collections import defaultdict

from elftools.elf.elffile import ELFFile

from anchors import anchor_sequence, callee_fingerprint
from extract_constants import const_bag
from extract_functions import disassembler, read_function, tokens

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
EXT = os.path.join(DATA, "extracted", "openwrt")


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(DATA, "labels", "openwrt_located.csv"))) if r["how"] == "dynsym"]
    by_elf = defaultdict(set)
    for r in rows:
        by_elf[r["elf"]].add((int(r["addr"]), int(r["size"])))
    n_ok = n_bad = 0
    with open(os.path.join(DATA, "functions", "openwrt_funcs.jsonl"), "w") as out:
        for rel, funcs in sorted(by_elf.items()):
            with open(os.path.join(EXT, rel), "rb") as fh:
                elf = ELFFile(fh)
                m, bits, little = elf.get_machine_arch(), elf.elfclass, elf.little_endian
                for addr, size in sorted(funcs):
                    thumb = m == "ARM" and addr & 1
                    a = addr & ~1 if thumb else addr
                    code = read_function(elf, fh, a, size) if size else None
                    ins = tokens(disassembler(m, bits, little, thumb), code, a) if code else []
                    if not ins:
                        n_bad += 1
                        continue
                    md = disassembler(m, bits, little, thumb)
                    cb = dict(const_bag(md, code, a))
                    # sstrip removes section headers, so the PLT cannot be named here
                    anc = anchor_sequence(md, code, a, callee=lambda t: callee_fingerprint(md, elf, fh, t & ~1))
                    out.write(json.dumps({"key": f"{rel}|{addr}", "arch": rel.split("/")[0], "n_insn": len(ins),
                                          "insns": ins, "consts": cb, "anchors": [list(x) for x in anc]}) + "\n")
                    n_ok += 1
    print("functions", n_ok, "failed", n_bad)


if __name__ == "__main__":
    main()
