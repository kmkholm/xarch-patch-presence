"""Download the public datasets used by this benchmark.

Runs sequentially (one large transfer at a time) and skips files that are
already complete, so it can be re-run after an interruption.
"""
import os
import sys
import time
import urllib.request

import gdown

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

HTTP = [
    ("plocator/plocator-binaries.tar.gz",
     "https://ndownloader.figshare.com/files/59157296", 592124217),
    ("debian/debian_security_tracker.json",
     "https://security-tracker.debian.org/tracker/data/json", None),
]

GDRIVE = [
    ("cisco/Dataset-Vulnerability.zip", "1i9CEJ7IGwyFF_3VlQWVsJRVW1XMXy96Z"),
    ("cisco/Dataset-Vulnerability-features.zip", "1NOL5DBem1TbI2Lcc0R64L7JM1BnJMDSp"),
    ("cisco/Dataset-1.zip", "1QpRgVJZTM52bfB6PvCKwVnmkKdHenUai"),
    ("firmae/FirmAE_dataset.zip", "1hdm75NVKBvs-eVH9rKb5xfgryNSnsg_8"),
]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch_http(rel, url, size):
    out = os.path.join(ROOT, rel)
    if os.path.exists(out) and (size is None or os.path.getsize(out) == size):
        log(f"skip {rel} ({os.path.getsize(out)/1e6:.0f} MB)")
        return
    log(f"get  {rel}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(out + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(out + ".part", out)
    log(f"done {rel} ({os.path.getsize(out)/1e6:.0f} MB)")


def fetch_gdrive(rel, gid):
    out = os.path.join(ROOT, rel)
    if os.path.exists(out):
        log(f"skip {rel} ({os.path.getsize(out)/1e6:.0f} MB)")
        return
    log(f"get  {rel}")
    res = gdown.download(id=gid, output=out + ".part", quiet=True, resume=True)
    if res is None:
        log(f"FAIL {rel}")
        return
    os.replace(out + ".part", out)
    log(f"done {rel} ({os.path.getsize(out)/1e6:.0f} MB)")


if __name__ == "__main__":
    for rel, url, size in HTTP:
        try:
            fetch_http(rel, url, size)
        except Exception as e:
            log(f"FAIL {rel}: {e}")
    for rel, gid in GDRIVE:
        try:
            fetch_gdrive(rel, gid)
        except Exception as e:
            log(f"FAIL {rel}: {e}")
    log("ALL DONE")
    sys.exit(0)
