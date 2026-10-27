"""Real-firmware test set from the Cisco-Talos Dataset-Vulnerability.

Targets: libcrypto of two commercial firmware images (NETGEAR R7000, ARM32,
OpenSSL 1.0.2h; TP-Link Deco M4, MIPS32, OpenSSL 1.0.2d) and Cisco's own
OpenSSL 1.0.2d builds for ARM32 and MIPS32. Function boundaries come from the
dataset's IDA export (start_ea/end_ea). Labels follow the OpenSSL version of
each image and the fixed-in versions stated in NVD (the vendors' patch
status beyond the version string is not known, see the README).
References: Debian amd64 builds of the 1.0.2 series around each fix.
Output: data/functions/cisco_funcs.jsonl, data/labels/cisco_triplets_{x86,same}.csv
"""
import csv
import io
import json
import os
import zipfile

from elftools.elf.elffile import ELFFile

from anchors import anchor_sequence, callee_fingerprint
from extract_constants import const_bag
from extract_functions import disassembler, read_function, tokens
from pretrain_corpus import func_symbols

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
# function -> (CVE, last upstream version without the fix, first upstream version with the fix)
CVES = {
    "BN_dec2bn": ("CVE-2016-0797", "1.0.2f", "1.0.2g"),
    "BN_hex2bn": ("CVE-2016-0797", "1.0.2f", "1.0.2g"),
    "SRP_VBASE_get_by_user": ("CVE-2016-0798", "1.0.2f", "1.0.2g"),
    "EVP_EncodeUpdate": ("CVE-2016-2105", "1.0.2g", "1.0.2h"),
    "EVP_EncryptUpdate": ("CVE-2016-2106", "1.0.2g", "1.0.2h"),
    "X509_NAME_oneline": ("CVE-2016-2176", "1.0.2g", "1.0.2h"),
    "BN_bn2dec": ("CVE-2016-2182", "1.0.2h", "1.0.2i"),
    "MDC2_Update": ("CVE-2016-6303", "1.0.2h", "1.0.2i"),
}
# Debian amd64 reference builds (first unstable upload of the upstream version)
REF_VULN = {"1.0.2f": "1.0.2f-2", "1.0.2g": "1.0.2f-2", "1.0.2h": "1.0.2h-1"}
REF_PATCH = {"1.0.2g": "1.0.2h-1", "1.0.2h": "1.0.2h-1", "1.0.2i": "1.0.2i-1"}
TARGETS = {  # file in the zip -> (name, arch tag, upstream version)
    "libcrypto.so.1.0.0_NETGEAR_R7000_1.0.2h_arm32": ("NETGEAR R7000", "arm32", "1.0.2h"),
    "libcrypto.so.1.0.0_TP-Link_Deco-M4_1.0.2d_mips32": ("TP-Link Deco M4", "mips32", "1.0.2d"),
    "libcrypto.so.1.0.0_openssl_1.0.2d_arm32": ("OpenSSL build", "arm32", "1.0.2d"),
    "libcrypto.so.1.0.0_openssl_1.0.2d_mips32": ("OpenSSL build", "mips32", "1.0.2d"),
}


def vkey(v):
    return (v[:-1], v[-1])


REF_ARCH = {"arm32": "armel", "mips32": "mipsel"}  # same-architecture reference ports


def debian_ref(ver, arch="amd64"):
    """Tokens of the target functions in the stripped libcrypto of a Debian version and port."""
    base = os.path.join(DATA, "extracted", "debian", "openssl", ver, arch)
    lib = dbg = None
    for dp, _, fns in os.walk(base):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.startswith("libcrypto.so"):
                if "debug" in dp.replace("\\", "/"):
                    dbg = p
                else:
                    lib = p
    if dbg is None:  # build-id layout: pick the debug file that holds the BN functions
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(".debug"):
                    with open(os.path.join(dp, fn), "rb") as fh:
                        if "BN_bn2dec" in func_symbols(ELFFile(fh)):
                            dbg = os.path.join(dp, fn)
    with open(dbg, "rb") as fh:
        syms = func_symbols(ELFFile(fh))
    out = {}
    with open(lib, "rb") as fh:
        elf = ELFFile(fh)
        for name in CVES:
            if name in syms:
                a, sz = syms[name]
                md = disassembler(elf.get_machine_arch(), elf.elfclass, elf.little_endian, bool(a & 1))
                code = read_function(elf, fh, a & ~1, sz)
                anc = anchor_sequence(md, code, a & ~1, callee=lambda t: callee_fingerprint(md, elf, fh, t & ~1))
                out[name] = (tokens(md, code, a & ~1), dict(const_bag(md, code, a & ~1)), [list(x) for x in anc])
    return out


def main():
    funcs, trip = [], {"x86": [], "same": []}
    refs = {}
    for ver in sorted(set(REF_VULN.values()) | set(REF_PATCH.values())):
        for ra in ("amd64", "armel", "mipsel"):
            refs[(ver, ra)] = debian_ref(ver, ra)
            for name, (ins, cb, anc) in refs[(ver, ra)].items():
                funcs.append({"key": f"debian/{ver}/{ra}|{name}", "arch": ra, "n_insn": len(ins), "insns": ins,
                              "consts": cb, "anchors": anc})
    zb = zipfile.ZipFile(os.path.join(DATA, "cisco", "Dataset-Vulnerability.zip"))
    zf = zipfile.ZipFile(os.path.join(DATA, "cisco", "Dataset-Vulnerability-features.zip"))
    table = list(csv.DictReader(io.TextIOWrapper(zf.open("testing_Dataset-Vulnerability.csv"))))
    for fname, (dev, arch, uver) in TARGETS.items():
        fh = io.BytesIO(zb.read(f"Dataset-Vulnerability/{fname}"))
        elf = ELFFile(fh)
        md = disassembler(elf.get_machine_arch(), elf.elfclass, elf.little_endian, False)
        rows = [r for r in table if fname in r["idb_path"] and r["func_name"] in CVES]
        for r in rows:
            name = r["func_name"]
            a, b = int(r["start_ea"], 16), int(r["end_ea"], 16)
            code = read_function(elf, fh, a, b - a)
            ins, cb = tokens(md, code, a), dict(const_bag(md, code, a))
            anc = [list(x) for x in anchor_sequence(md, code, a, callee=lambda t: callee_fingerprint(md, elf, fh, t & ~1))]
            cve, last_vuln, first_fix = CVES[name]
            label = "patched" if vkey(uver) >= vkey(first_fix) else "vuln"
            vref, pref = REF_VULN[last_vuln], REF_PATCH[first_fix]
            key = f"cisco/{fname}|{name}"
            funcs.append({"key": key, "arch": arch, "n_insn": len(ins), "insns": ins, "consts": cb, "anchors": anc})
            for mode, ra in (("x86", "amd64"), ("same", REF_ARCH[arch])):
                if name not in refs[(vref, ra)] or name not in refs[(pref, ra)]:
                    print("no reference", name, ra)
                    continue
                trip[mode].append([cve, name, dev, arch, uver, label, key, f"debian/{vref}/{ra}|{name}", vref,
                                   f"debian/{pref}/{ra}|{name}", pref])
    with open(os.path.join(DATA, "functions", "cisco_funcs.jsonl"), "w") as fh:
        for f in funcs:
            fh.write(json.dumps(f) + "\n")
    for mode, tr in trip.items():
        with open(os.path.join(DATA, "labels", f"cisco_triplets_{mode}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cve", "function", "device", "arch", "upstream", "label", "target", "vref", "vref_version",
                        "pref", "pref_version"])
            w.writerows(tr)
        print(mode, "targets", len(tr), "patched", sum(t[5] == "patched" for t in tr))


if __name__ == "__main__":
    main()
