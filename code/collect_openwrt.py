"""Collect stripped OpenWrt router builds of the target libraries/tools.

Walks every release in archive.openwrt.org (17.01 -> 25.12) for five router
architectures and downloads each distinct .ipk once. These builds ship without
debug symbols, so they serve as the realistic, binary-level external test.
"""
import csv
import os
import re
import time
import urllib.request

ARCHIVE = "https://archive.openwrt.org/releases/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "openwrt")
ARCHS = ["mips_24kc", "mipsel_24kc", "arm_cortex-a7_neon-vfpv4", "aarch64_cortex-a53", "x86_64"]
PKG = re.compile(r'href="((libopenssl(1\.1|3)?|libxml2|tcpdump|libfreetype|busybox|dnsmasq)_[^"/]+\.ipk)"')


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "patch-presence-research/1.0"}), timeout=120) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    html = get(ARCHIVE).decode()
    rels = sorted(set(re.findall(r'href="(\d\d\.\d\d\.\d+)/"', html)))
    man_path = os.path.join(OUT, "manifest.csv")
    seen = set()
    if os.path.exists(man_path):
        with open(man_path, newline="") as f:
            seen = {(r["arch"], r["file"]) for r in csv.DictReader(f)}
    fresh = not os.path.exists(man_path)
    man = open(man_path, "a", newline="")
    w = csv.writer(man)
    if fresh:
        w.writerow(["release", "arch", "feed", "package", "file"])
    for rel in rels:
        for arch in ARCHS:
            for feed in ("base", "packages"):
                url = f"{ARCHIVE}{rel}/packages/{arch}/{feed}/"
                try:
                    idx = get(url).decode()
                except Exception:
                    continue
                for m in PKG.finditer(idx):
                    fn = m.group(1)
                    if (arch, fn) in seen:
                        continue
                    dst = os.path.join(OUT, arch, fn)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        try:
                            data = get(url + fn)
                        except Exception as e:
                            print("fail", url + fn, e, flush=True)
                            continue
                        with open(dst, "wb") as f:
                            f.write(data)
                        time.sleep(0.2)
                    w.writerow([rel, arch, feed, m.group(2), fn])
                    man.flush()
                    seen.add((arch, fn))
                time.sleep(0.3)
        print(f"{rel} ok ({len(seen)} files)", flush=True)
    man.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
