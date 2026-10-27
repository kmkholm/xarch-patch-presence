"""Index the extracted Debian binaries.

For each (project, version, arch) record:
  * stripped ELF files of the package (path, machine, build-id)
  * the matching debug file (by build-id, or by path for old -dbg packages)
  * address/size of every target function (from the CVE table) found in the
    debug .symtab, so later stages can cut the exact function out of the
    stripped binary
  * the set of CVE ids mentioned in changelog.Debian.gz
Output: data/labels/debian_index.json
"""
import gzip
import json
import os
import re
from collections import defaultdict

from elftools.elf.elffile import ELFFile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EXT = os.path.join(ROOT, "extracted", "debian")


def build_id(elf):
    for sec in elf.iter_sections():
        if sec.header.sh_type == "SHT_NOTE":
            for note in sec.iter_notes():
                if note["n_type"] == "NT_GNU_BUILD_ID":
                    return note["n_desc"]
    return None


def func_syms(elf, wanted):
    out, total = {}, 0
    for name in (".symtab", ".dynsym"):
        sec = elf.get_section_by_name(name)
        if sec is None or sec.header.sh_type == "SHT_NOBITS":
            continue
        for s in sec.iter_symbols():
            if s["st_info"]["type"] != "STT_FUNC" or s["st_value"] == 0:
                continue
            total += 1
            if s.name in wanted and s.name not in out:
                out[s.name] = [s["st_value"], s["st_size"]]
    return out, total


def main():
    from build_cve_table import cve_functions
    wanted = defaultdict(set)
    for (proj, _), funcs in cve_functions().items():
        wanted[proj].update(funcs)
    index = {}
    for proj in sorted(os.listdir(EXT)):
        for ver in sorted(os.listdir(os.path.join(EXT, proj))):
            for arch in sorted(os.listdir(os.path.join(EXT, proj, ver))):
                base = os.path.join(EXT, proj, ver, arch)
                elfs, dbgs, changelog = [], {}, set()
                for dp, _, fns in os.walk(base):
                    for fn in fns:
                        p = os.path.join(dp, fn)
                        rel = os.path.relpath(p, EXT).replace("\\", "/")
                        if fn == "changelog.Debian.gz":
                            changelog |= set(re.findall(r"CVE-\d{4}-\d{4,}",
                                                        gzip.open(p).read().decode("utf-8", "replace")))
                            continue
                        try:
                            with open(p, "rb") as fh:
                                if fh.read(4) != b"\x7fELF":
                                    continue
                                fh.seek(0)
                                elf = ELFFile(fh)
                                bid = build_id(elf)
                                if "/usr/lib/debug/" in "/" + rel:
                                    syms, total = func_syms(elf, wanted[proj])
                                    key = bid or os.path.basename(p)
                                    dbgs[key] = {"path": rel, "syms": syms, "n_funcs": total}
                                else:
                                    elfs.append({"path": rel, "name": fn, "build_id": bid,
                                                 "machine": elf.get_machine_arch(),
                                                 "bits": elf.elfclass})
                        except Exception as e:
                            print("bad elf", rel, e, flush=True)
                for e in elfs:
                    d = dbgs.get(e["build_id"]) or dbgs.get(e["name"])
                    e["debug"] = d["path"] if d else None
                    e["syms"] = d["syms"] if d else {}
                    e["n_funcs"] = d["n_funcs"] if d else 0
                index[f"{proj}|{ver}|{arch}"] = {"project": proj, "version": ver, "arch": arch,
                                                 "elfs": elfs, "changelog_cves": sorted(changelog)}
            print(proj, ver, "ok", flush=True)
    json.dump(index, open(os.path.join(ROOT, "labels", "debian_index.json"), "w"))
    print("ALL DONE", len(index), flush=True)


if __name__ == "__main__":
    main()
