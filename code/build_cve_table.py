"""Build the CVE table: CVE -> project, vulnerable functions, NVD affected
upstream ranges.

Functions come from the PLocator (TOSEM'26) and PS3 (ICSE'24) ground truth;
affected ranges come from NVD CPE configurations (API 2.0, one bulk
request per product). Output: data/labels/cve_table.json
"""
import collections
import csv
import json
import os
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CPE = {"openssl": "openssl:openssl", "libxml2": "xmlsoft:libxml2",
       "tcpdump": "tcpdump:tcpdump", "freetype": "freetype:freetype"}


def cve_functions():
    m = collections.defaultdict(set)
    with open(os.path.join(ROOT, "plocator/repo/org_data/dataset_wo_irr.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[(r["oss"], r["cve"])].add(r["vul_func"])
    with open(os.path.join(ROOT, "ps3/repo/dataset/CVE_info.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["project"] != "FFmpeg":
                m[(r["project"], r["CVE"])].add(r["func"])
    return m


def parse_cve(v, product):
    out = []
    for conf in v.get("configurations", []):
        for node in conf["nodes"]:
            for cm in node["cpeMatch"]:
                if not cm.get("vulnerable") or f":{product}:" not in cm["criteria"]:
                    continue
                parts = cm["criteria"].split(":")
                exact = parts[5] if parts[5] not in ("*", "-") else None
                if parts[6] not in ("*", "-") and exact:
                    exact = exact + parts[6]  # e.g. 1.0.1 + g
                out.append({k2: cm.get(k2) for k2 in ("versionStartIncluding", "versionStartExcluding",
                                                       "versionEndIncluding", "versionEndExcluding")}
                           | {"exact": exact})
    cwe = [x["description"][0]["value"] for x in v.get("weaknesses", [])]
    return out, cwe


def nvd_bulk(product):
    """All CVEs of one CPE product in a single call (bulk files are cached in data/labels)."""
    path = os.path.join(ROOT, "labels", f"nvd_{product.split(':')[1]}.json")
    if not os.path.exists(path):
        url = ("https://services.nvd.nist.gov/rest/json/cves/2.0?virtualMatchString=cpe:2.3:a:"
               f"{product}&resultsPerPage=2000")
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "patch-presence-research/1.0"}), timeout=120) as r:
            open(path, "wb").write(r.read())
        time.sleep(7)
    d = json.load(open(path, encoding="utf-8"))
    return {x["cve"]["id"]: x["cve"] for x in d["vulnerabilities"]}


def main():
    os.makedirs(os.path.join(ROOT, "labels"), exist_ok=True)
    bulk = {proj: nvd_bulk(cpe) for proj, cpe in CPE.items()}
    table = {}
    for (proj, cve), funcs in sorted(cve_functions().items()):
        v = bulk[proj].get(cve)
        ranges, cwe = parse_cve(v, CPE[proj]) if v else (None, None)
        table[cve] = {"project": proj, "functions": sorted(funcs), "ranges": ranges, "cwe": cwe}
    json.dump(table, open(os.path.join(ROOT, "labels", "cve_table.json"), "w"), indent=1)
    miss = [k for k, v in table.items() if not v["ranges"]]
    print("CVEs", len(table), "without NVD ranges", len(miss), miss)


if __name__ == "__main__":
    main()
