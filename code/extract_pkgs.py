"""Unpack downloaded .deb / .ipk packages into ELF binaries, debug files and
Debian changelogs.

Output mirrors the input tree under data/extracted/{debian,openwrt}/... and
keeps only what the benchmark needs:
  * ELF shared objects / executables of the target projects
  * separate debug files (usr/lib/debug/...)
  * usr/share/doc/<pkg>/changelog.Debian.gz
"""
import gzip
import io
import lzma
import os
import sys
import tarfile

import zstandard

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
WANT_DIRS = ("/lib/", "/usr/lib/", "/usr/sbin/", "/usr/bin/", "/bin/", "/sbin/")


def ar_members(blob):
    if not blob.startswith(b"!<arch>\n"):
        return {}
    out, i = {}, 8
    while i + 60 <= len(blob):
        hdr = blob[i:i + 60]
        name = hdr[:16].decode().strip().rstrip("/")
        size = int(hdr[48:58].decode().strip())
        out[name] = blob[i + 60:i + 60 + size]
        i += 60 + size + (size & 1)
    return out


def open_tar(name, data):
    if name.endswith(".zst"):
        data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
        return tarfile.open(fileobj=io.BytesIO(data))
    if name.endswith(".xz"):
        return tarfile.open(fileobj=io.BytesIO(lzma.decompress(data)))
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:*")


def data_tar(path):
    blob = open(path, "rb").read()
    mem = ar_members(blob)
    if not mem:  # newer .ipk files are a gzip'd tar of control/data tarballs
        outer = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
        for m in outer.getmembers():
            if m.name.lstrip("./").startswith("data.tar"):
                return open_tar(m.name, outer.extractfile(m).read())
        return None
    for k, v in mem.items():
        if k.startswith("data.tar"):
            return open_tar(k, v)
    return None


def keep(name, data):
    n = "/" + name.lstrip("./")
    if n.startswith("/usr/lib/debug/"):
        return True
    if n.endswith("changelog.Debian.gz"):
        return True
    if any(n.startswith(d) for d in WANT_DIRS) and data[:4] == b"\x7fELF":
        return True
    return False


def extract(src, dst_dir):
    tf = data_tar(src)
    if tf is None:
        return 0
    n = 0
    for m in tf.getmembers():
        if not m.isfile():
            continue
        data = tf.extractfile(m).read()
        if keep(m.name, data):
            out = os.path.join(dst_dir, m.name.lstrip("./"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(data)
            n += 1
    return n


def main(which):
    base = os.path.join(ROOT, which)
    dst_root = os.path.join(ROOT, "extracted", which)
    done = fail = 0
    for dp, _, fns in os.walk(base):
        for fn in fns:
            if not fn.endswith((".deb", ".ipk")):
                continue
            src = os.path.join(dp, fn)
            rel = os.path.relpath(src, base)
            dst = os.path.join(dst_root, rel + ".d")
            if os.path.isdir(dst):
                continue
            try:
                k = extract(src, dst)
                os.makedirs(dst, exist_ok=True)
                done += 1
                if k == 0:
                    print("empty", rel, flush=True)
            except Exception as e:
                fail += 1
                print("fail", rel, e, flush=True)
    print(f"{which}: extracted {done}, failed {fail}", flush=True)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["debian", "openwrt"]:
        main(w)
