"""Patch code mapping (after PLocator, Sec. 4.2.1): locate the instructions of
the two reference builds that implement the lines deleted and added by the
security patch, using the patch file, the source files at the vulnerable and
fixed tags, and the DWARF line table of the reference binaries.

Returns, for each reference build, the ordered anchor path of the patched hunk
(context and changed lines) and the anchors of the changed lines only, in the
format expected by anchors.anchor_features.
"""
import os
import re
from collections import defaultdict

from elftools.elf.elffile import ELFFile

from anchors import anchor_sequence, callee_fingerprint, plt_map
from extract_functions import disassembler, read_function

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def norm(line):
    line = re.sub(r"/\*.*?\*/|//.*$", "", line)
    return re.sub(r"[\s{}]", "", line)


def parse_patch(path):
    """Hunks of a unified diff: list of (old_start, new_start, [(tag, text)])."""
    hunks, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = HUNK.match(line)
        if m:
            cur = (int(m.group(1)), int(m.group(3)), [])
            hunks.append(cur)
        elif cur is not None and line[:1] in (" ", "-", "+") and not line.startswith(("---", "+++")):
            cur[2].append((line[0], line[1:].rstrip("\n")))
    return hunks


def locate(src_lines, side_lines, guess):
    """Line numbers (1-based) of side_lines in the source, searched near guess."""
    target = [norm(x) for x in side_lines]
    src = [norm(x) for x in src_lines]
    best, best_off = -1, None
    lo, hi = max(0, guess - 400), min(len(src), guess + 400)
    for off in range(lo, hi):
        score = sum(1 for k, t in enumerate(target) if t and off + k < len(src) and src[off + k] == t)
        if score > best:
            best, best_off = score, off
    return best_off


def line_table(path):
    """(basename, line) -> set of addresses from the DWARF line programs."""
    table = defaultdict(set)
    with open(path, "rb") as fh:
        elf = ELFFile(fh)
        if not elf.has_dwarf_info():
            return table
        dw = elf.get_dwarf_info()
        for cu in dw.iter_CUs():
            lp = dw.line_program_for_CU(cu)
            if lp is None:
                continue
            files = lp["file_entry"]
            ver5 = lp.header["version"] >= 5
            rows = [e.state for e in lp.get_entries() if e.state is not None]
            # each row covers the addresses up to the next row of the same sequence
            for cur, nxt in zip(rows, rows[1:]):
                if cur.end_sequence:
                    continue
                fi = cur.file if ver5 else cur.file - 1
                if 0 <= fi < len(files) and nxt.address > cur.address:
                    name = files[fi].name
                    name = name.decode() if isinstance(name, bytes) else name
                    table[(os.path.basename(name), cur.line)].add((cur.address, nxt.address))
    return table


_LT = {}


def changed_lines(patch_dir, src_root, vul_tag, fix_tag):
    """For each patched file: deleted/context line numbers in the vulnerable
    source and added/context line numbers in the fixed source."""
    out = {}
    for f in os.listdir(os.path.join(patch_dir, "src")):
        rel = f.replace("#", "/")
        vs, fs = os.path.join(src_root, vul_tag, rel), os.path.join(src_root, fix_tag, rel)
        if not (os.path.exists(vs) and os.path.exists(fs)):
            continue
        vsrc = open(vs, encoding="utf-8", errors="replace").read().splitlines()
        fsrc = open(fs, encoding="utf-8", errors="replace").read().splitlines()
        dele, add, vctx, fctx = set(), set(), set(), set()
        for old_start, new_start, body in parse_patch(os.path.join(patch_dir, "src", f)):
            old_side = [(t, x) for t, x in body if t in " -"]
            new_side = [(t, x) for t, x in body if t in " +"]
            vo = locate(vsrc, [x for _, x in old_side], old_start - 1)
            fo = locate(fsrc, [x for _, x in new_side], new_start - 1)
            if vo is not None:
                for k, (t, _) in enumerate(old_side):
                    (dele if t == "-" else vctx).add(vo + k + 1)
            if fo is not None:
                for k, (t, _) in enumerate(new_side):
                    (add if t == "+" else fctx).add(fo + k + 1)
        out[os.path.basename(rel)] = (dele, vctx, add, fctx)
    return out


def region(bin_path, func_addr, func_size, lines, ctx_lines, base):
    """Anchor path of the hunk and anchors of the changed lines in one build."""
    if bin_path not in _LT:
        _LT[bin_path] = line_table(bin_path)
    lt = _LT[bin_path]
    lo, hi = func_addr, func_addr + func_size
    ch = [r for ln in lines for r in lt.get((base, ln), ()) if lo <= r[0] < hi]
    cx = [r for ln in ctx_lines for r in lt.get((base, ln), ()) if lo <= r[0] < hi]
    inside = lambda x, rs: any(a <= x < b for a, b in rs)
    with open(bin_path, "rb") as fh:
        elf = ELFFile(fh)
        md = disassembler(elf.get_machine_arch(), elf.elfclass, elf.little_endian, False)
        code = read_function(elf, fh, func_addr, func_size)
        addrs = [i.address for i in md.disasm(code, func_addr)]
        anc, idx = anchor_sequence(md, code, func_addr, with_index=True, plt=plt_map(elf),
                                   callee=lambda t: callee_fingerprint(md, elf, fh, t))
    path = [a for a, i in zip(anc, idx) if inside(addrs[i], ch) or inside(addrs[i], cx)]
    patch = [a for a, i in zip(anc, idx) if inside(addrs[i], ch)]
    return path, patch, sum(1 for x in addrs if inside(x, ch))
