"""Label every (CVE, binary) pair as vulnerable or patched.

Debian rule, for upstream version u of Debian version v and CVE c:
  * c listed in changelog.Debian of v           -> patched
        (kind 'backport' when u itself is inside an NVD affected range;
         otherwise kept only if u's series contains an affected version,
         since changelogs are cumulative)
  * else u inside an NVD affected range          -> vuln
  * else an affected version of the same series
    lies below u                                -> patched (kind 'upstream')
  * else                                        -> not affected (dropped)
A pair is kept only when the CVE's vulnerable function is present in the
debug symbol table, so its exact address in the stripped ELF is known.

OpenWrt rule: the same, without the changelog step (builds track upstream
releases); the function location is left to the matcher.

Outputs: data/labels/debian_pairs.csv, data/labels/openwrt_pairs.csv
"""
import csv
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LAB = os.path.join(ROOT, "labels")


def upstream(debver):
    v = debver.split(":")[-1]
    if "-" in v:
        v = v.rsplit("-", 1)[0]
    return re.split(r"[+~]", v)[0]


def vkey(v):
    return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|[a-z]+", v.lower())]


def series(proj, u):
    nums = re.findall(r"\d+", u)
    if proj == "openssl" and nums and nums[0] in ("0", "1"):
        return ".".join(nums[:3])
    return ".".join(nums[:2])


def in_range(u, r):
    k = vkey(u)
    if r.get("exact"):
        return k == vkey(r["exact"])
    lo_i, lo_e = r.get("versionStartIncluding"), r.get("versionStartExcluding")
    hi_i, hi_e = r.get("versionEndIncluding"), r.get("versionEndExcluding")
    if not any((lo_i, lo_e, hi_i, hi_e)):
        return False
    if lo_i and k < vkey(lo_i):
        return False
    if lo_e and k <= vkey(lo_e):
        return False
    if hi_i and k > vkey(hi_i):
        return False
    if hi_e and k >= vkey(hi_e):
        return False
    return True


def range_points(r):
    return [x for x in (r.get("exact"), r.get("versionStartIncluding"), r.get("versionStartExcluding"),
                        r.get("versionEndIncluding")) if x]


def status(proj, u, cve, changelog=None):
    ranges = cve["ranges"] or []
    aff = any(in_range(u, r) for r in ranges)
    s = series(proj, u)
    same = [p for r in ranges for p in range_points(r) + [r.get("versionEndExcluding") or ""]
            if p and series(proj, p) == s]
    if changelog is not None and cve["id"] in changelog:
        if aff:
            return "patched", "backport"
        # changelogs are cumulative: keep the mention only inside an affected series
        return ("patched", "changelog") if same else (None, None)
    if aff:
        return "vuln", "vuln"
    lower = [p for p in same if vkey(p) < vkey(u)]
    if lower:
        return "patched", "upstream"
    return None, None


def suite(ver):
    m = re.search(r"deb(\d+)u\d+", ver)
    return f"deb{m.group(1)}" if m else "unstable"


def label_debian(cves):
    idx = json.load(open(os.path.join(LAB, "debian_index.json")))
    rows, dropped = [], {"not_affected": 0, "no_symbol": 0}
    for ent in idx.values():
        proj, ver, arch = ent["project"], ent["version"], ent["arch"]
        u = upstream(ver)
        cl = set(ent["changelog_cves"])
        for cid, c in cves.items():
            if c["project"] != proj:
                continue
            lab, kind = status(proj, u, c, cl)
            if lab is None:
                dropped["not_affected"] += 1
                continue
            hit = False
            for f in c["functions"]:
                for e in ent["elfs"]:
                    if f in e["syms"]:
                        a, sz = e["syms"][f]
                        rows.append([cid, proj, ver, u, series(proj, u), suite(ver), arch, e["machine"],
                                     e["path"], f, a, sz, lab, kind])
                        hit = True
                        break
            if not hit:
                dropped["no_symbol"] += 1
    with open(os.path.join(LAB, "debian_pairs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cve", "project", "version", "upstream", "series", "suite", "arch", "machine",
                    "elf", "function", "addr", "size", "label", "kind"])
        w.writerows(rows)
    return rows, dropped


OW_PROJ = {"libopenssl": "openssl", "libopenssl1.1": "openssl", "libopenssl3": "openssl",
           "libxml2": "libxml2", "tcpdump": "tcpdump", "libfreetype": "freetype"}


def label_openwrt(cves):
    ext = os.path.join(ROOT, "extracted", "openwrt")
    rows, seen = [], set()
    with open(os.path.join(ROOT, "openwrt", "manifest.csv"), newline="") as fh:
        man = list(csv.DictReader(fh))
    for r in man:
        proj = OW_PROJ.get(r["package"])
        if proj is None or (r["arch"], r["file"]) in seen:
            continue
        seen.add((r["arch"], r["file"]))
        ver = r["file"].split("_")[1]
        u = re.split(r"-r?\d+$", ver)[0]
        d = os.path.join(ext, r["arch"], r["file"] + ".d")
        elfs = [os.path.relpath(os.path.join(dp, f), ext).replace("\\", "/")
                for dp, _, fs in os.walk(d) for f in fs]
        if not elfs:
            continue
        for cid, c in cves.items():
            if c["project"] != proj:
                continue
            lab, kind = status(proj, u, c)
            if lab is None:
                continue
            rows.append([cid, proj, ver, u, series(proj, u), r["release"], r["arch"], ";".join(elfs),
                         ";".join(c["functions"]), lab, kind])
    with open(os.path.join(LAB, "openwrt_pairs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cve", "project", "version", "upstream", "series", "first_release", "arch", "elfs",
                    "functions", "label", "kind"])
        w.writerows(rows)
    return rows


def main():
    cves = json.load(open(os.path.join(LAB, "cve_table.json")))
    for k, v in cves.items():
        v["id"] = k
    rows, dropped = label_debian(cves)
    ow = label_openwrt(cves)
    print("debian pairs", len(rows), "dropped", dropped)
    print("openwrt pairs", len(ow))


if __name__ == "__main__":
    main()
