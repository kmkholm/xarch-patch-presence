"""Cut every labelled target function out of its stripped ELF and turn it into
normalised instruction tokens.

Each function becomes a list of instructions, each instruction a short list
of sub-tokens: the mnemonic, then operand tokens with registers kept, small
constants (0..16) kept and every other immediate/address replaced by IMM.
ARM functions with an odd symbol value are decoded in Thumb mode.

Output: data/functions/debian_funcs.jsonl  {"key", "arch", "n_insn", "insns"}
"""
import csv
import json
import os
import re
from collections import defaultdict

import capstone as cs
from elftools.elf.elffile import ELFFile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EXT = os.path.join(ROOT, "extracted", "debian")
TOK = re.compile(r"0x[0-9a-fA-F]+|[A-Za-z_.$][\w.$]*|\d+|[\[\]()#!{}\-+*:^]")


def disassembler(machine, bits, little, thumb):
    if machine == "x64":
        return cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    if machine == "x86":
        return cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_32)
    if machine == "AArch64":
        return cs.Cs(cs.CS_ARCH_ARM64, cs.CS_MODE_ARM)
    if machine == "ARM":
        return cs.Cs(cs.CS_ARCH_ARM, cs.CS_MODE_THUMB if thumb else cs.CS_MODE_ARM)
    if machine == "MIPS":
        mode = cs.CS_MODE_MIPS64 if bits == 64 else cs.CS_MODE_MIPS32
        mode |= cs.CS_MODE_LITTLE_ENDIAN if little else cs.CS_MODE_BIG_ENDIAN
        return cs.Cs(cs.CS_ARCH_MIPS, mode)
    raise ValueError(machine)


def norm_operand_tokens(op_str):
    out = []
    for t in TOK.findall(op_str):
        if t[0].isdigit():
            v = int(t, 16) if t.lower().startswith("0x") else int(t)
            out.append(str(v) if v <= 16 else "IMM")
        else:
            out.append(t.lower())
    return out


def read_function(elf, fh, addr, size):
    for seg in elf.iter_segments():
        if seg["p_type"] != "PT_LOAD":
            continue
        lo = seg["p_vaddr"]
        if lo <= addr < lo + seg["p_filesz"]:
            fh.seek(addr - lo + seg["p_offset"])
            return fh.read(size)
    return None


def tokens(md, code, addr):
    insns = []
    for ins in md.disasm(code, addr):
        insns.append([ins.mnemonic.lower()] + norm_operand_tokens(ins.op_str))
    return insns


def main():
    os.makedirs(os.path.join(ROOT, "functions"), exist_ok=True)
    with open(os.path.join(ROOT, "labels", "debian_pairs.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_elf = defaultdict(set)
    for r in rows:
        by_elf[(r["elf"], r["arch"])].add((int(r["addr"]), int(r["size"])))
    out_path = os.path.join(ROOT, "functions", "debian_funcs.jsonl")
    n_ok = n_bad = 0
    with open(out_path, "w") as out:
        for (rel, arch), funcs in sorted(by_elf.items()):
            with open(os.path.join(EXT, rel), "rb") as fh:
                elf = ELFFile(fh)
                machine, bits, little = elf.get_machine_arch(), elf.elfclass, elf.little_endian
                for addr, size in sorted(funcs):
                    thumb = machine == "ARM" and addr & 1
                    a = addr & ~1 if thumb else addr
                    code = read_function(elf, fh, a, size)
                    insns = tokens(disassembler(machine, bits, little, thumb), code, a) if code else []
                    if not insns:
                        n_bad += 1
                        continue
                    out.write(json.dumps({"key": f"{rel}|{addr}", "arch": arch,
                                          "n_insn": len(insns), "insns": insns}) + "\n")
                    n_ok += 1
    print("functions", n_ok, "failed", n_bad)


if __name__ == "__main__":
    main()
