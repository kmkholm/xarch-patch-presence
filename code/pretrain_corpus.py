"""Build the cross-architecture pre-training corpus from Cisco-Talos Dataset-1.

Only patch-label-free function identity is used: functions that share a
symbol name across builds (architecture x compiler x optimisation) of the
same project. OpenSSL builds are left out so that no target library enters
pre-training.

Pass 1 reads symbol tables; names present in >=3 builds over >=2
architectures with 32..8000 bytes are kept (at most MAX_NAMES per project).
Pass 2 disassembles up to MAX_VARIANTS builds per kept name.
Output: data/functions/pretrain_funcs.jsonl {"key", "name", "arch", "insns"}
"""
import io
import json
import os
import random
import zipfile
from multiprocessing import Pool

import numpy as np
from collections import defaultdict

from elftools.elf.elffile import ELFFile

from extract_functions import disassembler, read_function, tokens

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ZIP = os.path.join(ROOT, "cisco", "Dataset-1.zip")
MAX_NAMES, MAX_VARIANTS = 6000, 6
WORKERS = 5
random.seed(0)


def func_symbols(elf):
    """Vectorised .symtab parse (pyelftools' per-symbol iteration is too slow
    for binaries with >100k symbols)."""
    sec = elf.get_section_by_name(".symtab")
    out = {}
    if sec is None:
        return out
    strtab = elf.get_section(sec["sh_link"]).data()
    e = "<" if elf.little_endian else ">"
    if elf.elfclass == 64:
        dt = np.dtype([("name", e + "u4"), ("info", "u1"), ("other", "u1"), ("shndx", e + "u2"),
                       ("value", e + "u8"), ("size", e + "u8")])
    else:
        dt = np.dtype([("name", e + "u4"), ("value", e + "u4"), ("size", e + "u4"), ("info", "u1"),
                       ("other", "u1"), ("shndx", e + "u2")])
    s = np.frombuffer(sec.data(), dtype=dt)
    sel = s[((s["info"] & 0xF) == 2) & (s["size"] >= 32) & (s["size"] <= 8000) & (s["value"] != 0)]
    for nm, v, sz in zip(sel["name"], sel["value"], sel["size"]):
        end = strtab.find(b"\0", nm)
        out.setdefault(strtab[nm:end].decode("latin-1"), (int(v), int(sz)))
    return out


def pass1(n):
    z = zipfile.ZipFile(ZIP)
    return n, func_symbols(ELFFile(io.BytesIO(z.read(n))))


def pass2(job):
    n, items = job
    z = zipfile.ZipFile(ZIP)
    fh = io.BytesIO(z.read(n))
    elf = ELFFile(fh)
    m, bits, little = elf.get_machine_arch(), elf.elfclass, elf.little_endian
    arch = n.split("/")[-1].split("-")[0]
    out = []
    for name, a, sz in items:
        thumb = m == "ARM" and a & 1
        aa = a & ~1 if thumb else a
        code = read_function(elf, fh, aa, sz)
        insns = tokens(disassembler(m, bits, little, thumb), code, aa) if code else []
        if len(insns) >= 8:
            out.append(json.dumps({"key": f"{n}|{a}", "name": name, "arch": arch, "insns": insns}))
    return out


def main():
    z = zipfile.ZipFile(ZIP)
    files = [n for n in z.namelist() if not n.endswith("/") and "/openssl/" not in n]
    where = defaultdict(list)  # (project, name) -> [(file, arch, addr, size)]
    with Pool(WORKERS) as pool:
        for i, (n, syms) in enumerate(pool.imap_unordered(pass1, files, chunksize=4)):
            proj, arch = n.split("/")[1], n.split("/")[-1].split("-")[0]
            for name, (a, sz) in syms.items():
                where[(proj, name)].append((n, arch, a, sz))
            if i % 200 == 0:
                print("pass1", i, len(files), flush=True)
    keep = defaultdict(list)
    for (proj, name), occ in where.items():
        if len(occ) >= 3 and len({o[1] for o in occ}) >= 2:
            keep[proj].append(name)
    plan = defaultdict(list)  # file -> [(name, addr, size)]
    for proj, names in keep.items():
        random.shuffle(names)
        for name in names[:MAX_NAMES]:
            occ = where[(proj, name)]
            random.shuffle(occ)
            for n, arch, a, sz in occ[:MAX_VARIANTS]:
                plan[n].append((f"{proj}:{name}", a, sz))
    print("names kept", {p: min(len(v), MAX_NAMES) for p, v in keep.items()}, flush=True)
    out_path = os.path.join(ROOT, "functions", "pretrain_funcs.jsonl")
    n_out = 0
    with open(out_path, "w") as out, Pool(WORKERS) as pool:
        for i, lines in enumerate(pool.imap_unordered(pass2, sorted(plan.items()), chunksize=2)):
            for line in lines:
                out.write(line + "\n")
            n_out += len(lines)
            if i % 200 == 0:
                print("pass2", i, len(plan), n_out, flush=True)
    print("ALL DONE", n_out, flush=True)


if __name__ == "__main__":
    main()
