"""Fetch the source files touched by each PLocator patch at the vulnerable and
the fixed release tags (GitHub raw), needed to map patch lines to line numbers
of the reference builds (PLocator, Sec. 4.2.1).
Output: data/plocator/src/<project>/<tag>/<path>"""
import json
import os
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "plocator")
PATCH = os.path.join(ROOT, "repo", "org_data", "patches")
REPO = {"openssl": "openssl/openssl", "tcpdump": "the-tcpdump-group/tcpdump", "libxml2": "GNOME/libxml2",
        "freetype": "freetype/freetype"}


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "patch-presence-research"}),
                                        timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def main():
    ok = miss = 0
    for d in sorted(os.listdir(PATCH)):
        proj = d.split("_")[1]
        tags = json.load(open(os.path.join(PATCH, d, "vul_fix_versions.json")))
        for f in os.listdir(os.path.join(PATCH, d, "src")):
            path = f.replace("#", "/")
            for pair in tags:
                for tag in pair:
                    dst = os.path.join(ROOT, "src", proj, tag, path)
                    if os.path.exists(dst):
                        continue
                    data = get(f"https://raw.githubusercontent.com/{REPO[proj]}/{tag}/{path}")
                    if data is None:
                        miss += 1
                        print("missing", proj, tag, path)
                        continue
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    open(dst, "wb").write(data)
                    ok += 1
                    time.sleep(0.2)
    print("fetched", ok, "missing", miss)


if __name__ == "__main__":
    main()
