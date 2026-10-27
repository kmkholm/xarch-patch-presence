"""Optimisation-stable anchors (after PLocator, TOSEM 2026, Sec. 4.2.2) and
patch-region anchor paths obtained from the two reference builds.

Anchors are taken only from key instructions that steer control flow:
  CMP  constant compared by cmp/test (test r,r and cbz/cbnz count as 0)
  AUX  immediate of the arithmetic instruction that feeds the comparison
  CALL a call site (callee identity is not used)
Stack offsets, addresses and other immediates are ignored, because they
change with register allocation and optimisation level.

The patch region is found without source code: the vulnerable and the patched
reference builds are aligned instruction by instruction, and the anchors of
the changed instructions plus a few neighbouring context anchors form the
reference anchor path of each build. The target is compared with both paths
in order (longest common subsequence).
"""
import difflib
import re
from collections import Counter

NUM = re.compile(r"0x[0-9a-fA-F]+|\b\d+\b")
CMP = ("cmp", "test", "cmn", "tst", "teq", "cbz", "cbnz", "tbz", "tbnz")
MIPS_BR = ("beq", "bne", "bgez", "bltz", "blez", "bgtz", "beqz", "bnez", "slti", "sltiu")
ARITH = ("add", "sub", "and", "or", "xor", "shl", "shr", "sar", "lsl", "lsr", "asr", "orr", "eor", "bic", "addiu",
         "andi", "ori", "xori", "sll", "srl", "sra")
CONTEXT = 3


def _imm(op_str):
    """Immediate operands, ignoring memory displacements inside brackets."""
    vals = []
    for t in NUM.findall(re.sub(r"\[.*?\]|\(.*?\)", " ", op_str)):
        v = int(t, 16) if t.startswith("0x") else int(t)
        if v <= 0x7FFFFFFF:
            vals.append(v)
    return vals


def plt_map(elf):
    """Address of each PLT stub -> imported function name (x86 and x86-64,
    classic lazy-binding layout). Works on stripped binaries, because the
    dynamic relocations and the dynamic symbol table are kept."""
    out = {}
    plt = elf.get_section_by_name(".plt")
    rel = elf.get_section_by_name(".rel.plt") or elf.get_section_by_name(".rela.plt")
    if plt is None or rel is None:
        return out
    symtab = elf.get_section(rel["sh_link"])
    entry = 16
    for i, r in enumerate(rel.iter_relocations()):
        sym = symtab.get_symbol(r["r_info_sym"])
        if sym is not None and sym.name:
            out[plt["sh_addr"] + entry * (i + 1)] = sym.name.split("@")[0]
    return out


def callee_fingerprint(md, elf, fh, target, max_bytes=1500):
    """Optimisation-tolerant identity of an internal callee: the set of its
    comparison constants other than 0 and 1, read from its entry up to the
    first return (function boundaries are not needed)."""
    from extract_functions import read_function
    code = read_function(elf, fh, target, max_bytes) or b""
    vals = set()
    for ins in md.disasm(code, target):
        m = ins.mnemonic.lower()
        if m.startswith("ret"):
            break
        if m.split(".")[0] in CMP:
            vals.update(v for v in _imm(ins.op_str) if v > 1)
    return "F:" + ",".join(str(v) for v in sorted(vals)[:6]) if len(vals) >= 2 else 0


def anchor_sequence(md, code, addr, with_index=False, plt=None, callee=None):
    """Ordered anchors of one function, optionally with instruction indices.
    With a PLT map, calls to imported functions carry the callee name; with a
    callee(target) function, internal calls carry a callee fingerprint."""
    seq, idx = [], []
    plt = plt or {}
    prev_arith = None
    for k, ins in enumerate(md.disasm(code, addr)):
        base = ins.mnemonic.lower().split(".")[0]
        ops = ins.op_str
        new = []
        if base in CMP or base in MIPS_BR:
            regs = [x.strip() for x in ops.split(",")]
            if (base == "test" and len(regs) == 2 and regs[0] == regs[1]) or base in ("cbz", "cbnz", "beqz", "bnez",
                                                                                     "bgez", "bltz", "blez", "bgtz"):
                vals = [0]
            elif base in ("beq", "bne"):
                vals = [0] if "$zero" in ops else []
            else:
                vals = _imm(ops)
            if vals:
                new.append(("CMP", vals[-1]))
            if prev_arith is not None:
                new.append(("AUX", prev_arith))
            prev_arith = None
        elif base.startswith(("call", "jal")) or base in ("bl", "blx", "blr"):
            tgt = _imm(ops)
            val = plt.get(tgt[0], 0) if tgt else 0
            if val == 0 and tgt and callee is not None:
                val = callee(tgt[0])
            new.append(("CALL", val))
            prev_arith = None
        elif base in ARITH and "sp" not in ops and "bp" not in ops and "gp" not in ops and "pc" not in ops:
            im = _imm(ops)
            if im and im[-1] < 0x10000:
                prev_arith = im[-1]
        seq += new
        idx += [k] * len(new)
    return (seq, idx) if with_index else seq


def lcs_ratio(ref, tgt):
    """Order-preserving match ratio of a reference anchor sequence in the target."""
    if not ref:
        return 0.0
    m = len(tgt)
    prev = [0] * (m + 1)
    for ri in ref:
        cur = [0] * (m + 1)
        for j in range(m):
            cur[j + 1] = prev[j] + 1 if ri == tgt[j] else max(prev[j + 1], cur[j])
        prev = cur
    return prev[m] / len(ref)


def _cos(a, b):
    num = sum(a[k] * b[k] for k in a if k in b)
    da = sum(x * x for x in a.values()) ** 0.5
    db = sum(x * x for x in b.values()) ** 0.5
    return num / (da * db + 1e-9)


def region_paths(v_tok, p_tok, v_anc, v_idx, p_anc, p_idx):
    """Reference anchor paths around the code changed by the fix in each build."""
    sm = difflib.SequenceMatcher(a=[" ".join(i) for i in v_tok], b=[" ".join(i) for i in p_tok], autojunk=False)
    cv, cp = set(), set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            cv.update(range(i1, i2))
            cp.update(range(j1, j2))

    def path(anc, idx, changed):
        hit = [k for k, i in enumerate(idx) if i in changed]
        if not hit:
            return [], []
        lo, hi = max(0, hit[0] - CONTEXT), min(len(anc), hit[-1] + 1 + CONTEXT)
        return anc[lo:hi], [anc[k] for k in hit]
    return path(v_anc, v_idx, cv), path(p_anc, p_idx, cp)


def anchor_features(t_anc, v_anc, p_anc, v_region=None, p_region=None, cap=(300, 400)):
    """Whole-function anchor evidence plus, when available, patch-region evidence."""
    T, V, P = Counter(t_anc), Counter(v_anc), Counter(p_anc)
    dp, dm = P - V, V - P
    n = sum(dp.values()) + sum(dm.values())
    delta = 0.0 if n == 0 else (sum(min(T[k], c) for k, c in dp.items()) - sum(min(T[k], c) for k, c in dm.items())) / n
    r, t = cap
    f = [delta, float(n == 0), lcs_ratio(p_anc[:r], t_anc[:t]) - lcs_ratio(v_anc[:r], t_anc[:t]), _cos(T, P) - _cos(T, V)]
    if v_region is not None:
        (vpath, vpatch), (ppath, ppatch) = v_region, p_region
        lp, lv = lcs_ratio(ppath, t_anc[:600]), lcs_ratio(vpath, t_anc[:600])
        # presence of the anchors that only the fix adds / only the vulnerable code has
        # for the anchor values touched by the patch, is the target's count closer
        # to the patched or to the vulnerable reference? (+1: patched, -1: vulnerable)
        keys = set(ppatch) | set(vpatch)
        den = sum(abs(P[k] - V[k]) for k in keys)
        pres = 0.0 if den == 0 else sum(abs(T[k] - V[k]) - abs(T[k] - P[k]) for k in keys) / den
        f += [lp - lv, lp, lv, pres, float(not vpath and not ppath)]
    return f
