"""Recover function boundaries in stripped ELF files with angr CFGFast.

Runs inside the separate venv .venv_angr (angr needs a newer protobuf than
the main environment). For each input ELF, writes a JSON list of
[start, size] for every recovered function (Thumb starts keep the low bit).
Usage: .venv_angr/Scripts/python recover_functions.py <list.txt> <out_dir>
"""
import json
import logging
import os
import sys
import time

import angr

logging.getLogger("angr").setLevel(logging.ERROR)
logging.getLogger("cle").setLevel(logging.ERROR)
logging.getLogger("pyvex").setLevel(logging.ERROR)


def recover(path):
    proj = angr.Project(path, auto_load_libs=False, load_options={"main_opts": {"base_addr": 0}})
    cfg = proj.analyses.CFGFast(normalize=True, data_references=False, resolve_indirect_jumps=True,
                                force_complete_scan=True, show_progressbar=False)
    out = []
    for f in cfg.kb.functions.values():
        if f.is_plt or f.is_simprocedure or f.is_syscall or not f.block_addrs_set:
            continue
        blocks = [b for b in f.blocks]
        lo = min(b.addr for b in blocks)
        hi = max(b.addr + b.size for b in blocks)
        out.append([int(f.addr), int(hi - (f.addr & ~1))])
    return out


def main():
    lst, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for line in open(lst):
        rel, path = line.rstrip("\n").split("\t")
        dst = os.path.join(out_dir, rel.replace("/", "__") + ".json")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        try:
            res = recover(path)
        except Exception as e:
            print("fail", rel, repr(e)[:200], flush=True)
            continue
        json.dump(res, open(dst, "w"))
        print("ok", rel, len(res), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
