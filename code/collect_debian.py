"""Collect cross-architecture Debian binaries (plus debug symbols) for the
patch-presence benchmark.

For each source package we keep every stable/security upload (+debNuM or
~debNuM, where back-ported fixes live) and the first unstable upload of each
upstream release. For every kept version we fetch the library/tool package and
its debug-symbol package for six architectures from snapshot.debian.org.

The manifest (data/debian/manifest.csv) records one row per downloaded file;
re-running skips files already on disk with the right size.
"""
import csv
import json
import os
import re
import time
import urllib.request

S = "https://snapshot.debian.org"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "debian")
ARCHS = ["armel", "armhf", "arm64", "mipsel", "mips64el", "amd64"]

PROJECTS = {
    "openssl": (r"^(1\.0\.1|1\.0\.2|1\.1\.0|1\.1\.1|3\.0\.)", r"^libssl(1\.0\.0|1\.0\.2|1\.1|3|3t64)$"),
    "libxml2": (r"^2\.(9|10)\.", r"^libxml2$"),
    "tcpdump": (r"^4\.9\.", r"^tcpdump$"),
    "freetype": (r"^2\.(5|6|8|9|10|11|12)\.", r"^libfreetype6$"),
}
SKIP = ("~exp", "hurd", "~bpo", "beta", "~pre", "~rc")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "patch-presence-research/1.0"}), timeout=120) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def jget(url):
    time.sleep(0.6)
    return json.loads(get(url))


def upstream(v):
    return re.split(r"[-+~]", v.split(":")[-1])[0]


def pick_versions(src, rx):
    vers = [r["version"] for r in jget(f"{S}/mr/package/{src}/")["result"]]
    vers = [v for v in vers if re.match(rx, v) and not any(s in v for s in SKIP)]
    keep, seen_up = [], set()
    for v in reversed(vers):  # oldest first
        if re.search(r"deb\d+u\d+", v):
            keep.append(v)
        elif upstream(v) not in seen_up:
            keep.append(v)
        seen_up.add(upstream(v))
    return keep


def main():
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest.csv")
    done = set()
    if os.path.exists(man_path):
        with open(man_path, newline="") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    fresh = not os.path.exists(man_path)
    man = open(man_path, "a", newline="")
    w = csv.writer(man)
    if fresh:
        w.writerow(["project", "version", "arch", "binpkg", "kind", "file", "size"])
    for src, (vrx, brx) in PROJECTS.items():
        versions = pick_versions(src, vrx)
        print(f"{src}: {len(versions)} versions", flush=True)
        for ver in versions:
            try:
                bins = jget(f"{S}/mr/package/{src}/{ver}/binpackages")["result"]
            except Exception as e:
                print("  binpackages fail", ver, e, flush=True)
                continue
            names = {b["name"]: b["version"] for b in bins}
            mains = [n for n in names if re.match(brx, n)]
            for m in mains:
                dbg = next((d for d in (m + "-dbgsym", m + "-dbg") if d in names), None)
                for pkg, kind in [(m, "bin"), (dbg, "dbg")]:
                    if pkg is None:
                        continue
                    try:
                        info = jget(f"{S}/mr/binary/{pkg}/{names[pkg]}/binfiles?fileinfo=1")
                    except Exception as e:
                        print("  binfiles fail", pkg, ver, e, flush=True)
                        continue
                    for r in info["result"]:
                        if r["architecture"] not in ARCHS:
                            continue
                        fi = info["fileinfo"][r["hash"]][0]
                        rel = os.path.join(src, ver.replace(":", "_"), r["architecture"], fi["name"])
                        if rel in done:
                            continue
                        dst = os.path.join(OUT, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if not (os.path.exists(dst) and os.path.getsize(dst) == fi["size"]):
                            try:
                                data = get(f"{S}/file/{r['hash']}")
                            except Exception as e:
                                print("  file fail", rel, e, flush=True)
                                continue
                            with open(dst, "wb") as f:
                                f.write(data)
                            time.sleep(0.3)
                        w.writerow([src, ver, r["architecture"], pkg, kind, rel, fi["size"]])
                        man.flush()
                        done.add(rel)
            print(f"  {src} {ver} ok ({len(done)} files)", flush=True)
    man.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
