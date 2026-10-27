# Cross-Architecture Patch Presence Benchmark for IoT Binaries

This repository contains a labelled benchmark and the full pipeline for **binary patch presence testing** across
instruction sets: given a stripped function from an IoT firmware binary and the fix for a CVE, decide whether
the fix is present. It targets the case that version-based scanners cannot handle: a security fix that was
**back-ported** without changing the upstream version string.

## What is in the benchmark

| | |
|---|---|
| Source | Debian security updates (snapshot.debian.org), binaries with debug symbols |
| Libraries | OpenSSL, libxml2, tcpdump, FreeType |
| Architectures | amd64, armel, armhf (Thumb-2), arm64, mipsel, mips64el |
| Labelled pairs | 19,403 (CVE, binary, function) pairs over 97 CVEs and 1,531 binaries |
| Back-ported fixes | 2,864 pairs whose upstream version lies inside the affected range |
| Evaluation triplets | 17,288 (target, vulnerable reference, patched reference) over 85 CVEs |
| External set | 4,405 version-labelled pairs from stripped OpenWrt builds (5 router architectures) |

Labels come from the Debian changelogs (back-ported fixes) and from the NVD affected version ranges
(CPE match API). Function addresses come from the distribution's debug symbols, and the CVE-to-function
mapping follows the ground truth of earlier patch presence studies.

## Repository layout

```
code/                      pipeline (Python)
data/labels/               benchmark labels, triplets and folds (CSV / JSON)
data/functions/            extracted function features
```

Files in `data/labels/`:

| File | Content |
|---|---|
| `debian_pairs.csv` | all labelled pairs: CVE, library, function, architecture, version, label, label kind, ELF path, address, size |
| `triplets_same.csv` | evaluation triplets with references of the same architecture |
| `triplets_x86.csv` | the same targets with x86-64 references |
| `folds.json` | five CVE-grouped folds (no CVE appears in two folds) |
| `folds_bin.json` | five binary-level folds (for comparison only; CVEs are shared between folds) |
| `openwrt_pairs.csv`, `openwrt_located.csv` | OpenWrt pairs and the functions located in the stripped builds |
| `cisco_triplets_*.csv` | triplets for the firmware functions of the Cisco-Talos vulnerability set |
| `cve_table.json`, `nvd_*.json` | CVE table and the NVD affected ranges used for labelling |
| `debian_index.json` | index of the Debian versions and architectures used |

Files in `data/functions/`: normalised instruction sequences (`debian_funcs.jsonl.gz`, decompress with
`gunzip -k`), constant multisets, control-flow anchor sequences, the OpenWrt and Cisco-Talos functions, and the
token vocabulary.

## Data size and what is not included

The raw binaries are **not** stored here. The complete working set is about 45 GB (Debian packages 3.9 GB,
extracted files 7 GB, OpenWrt 0.35 GB, PLocator 2.1 GB, Cisco-Talos 3.9 GB, FirmAE 12 GB), the packages carry
their own licences, and all of them are publicly available. The scripts below download them again from the
original sources:

| Script | Source |
|---|---|
| `collect_debian.py`, `extract_pkgs.py` | snapshot.debian.org (packages and dbgsym packages) |
| `collect_openwrt.py` | archive.openwrt.org |
| `download_all.py` | PLocator benchmark (Figshare), Debian security tracker, other public sets |
| `fetch_sources.py` | upstream source files at the release tags (GitHub) |

The pre-training corpus (functions from the Cisco-Talos binary similarity dataset, OpenSSL excluded) is rebuilt
by `pretrain_corpus.py`.

## Pipeline

```bash
pip install -r requirements.txt
cd code
# 1. data and labels
python collect_debian.py && python extract_pkgs.py && python index_debian.py
python build_cve_table.py && python label_pairs.py && python build_triplets.py
# 2. function features
python extract_functions.py && python extract_constants.py && python extract_anchors.py
# 3. learned stream: cross-architecture pre-training, then CVE-grouped fine-tuning
python pretrain_corpus.py && bash run_grid.sh
# 4. evidence fusion (gradient-boosted trees) and conformal risk control
python fuse.py full x86ref && python evaluate.py full+fused x86ref+fused
# 5. external sets
python collect_openwrt.py && python openwrt_locate.py && python extract_openwrt.py
python external_cisco.py && python eval_external.py
```

Main components:

- `anchors.py` – control-flow anchors (comparison constants, feeding immediates, callee identities) and their
  order-preserving match
- `patchmap.py` – mapping of patched source lines to instructions through DWARF line tables
- `model.py`, `pretrain.py`, `finetune.py` – gated attention function encoder and reference-contrast head
- `fuse.py` – evidence fusion, cross-fitted over the CVE folds
- `evaluate.py` – metrics, CVE-level bootstrap and conformal risk control with CVEs as the exchangeable units
- `plocator_eval.py`, `plocator_cv.py`, `plocator_fuse.py` – evaluation on the PLocator benchmark protocol

All results are computed with CVE-grouped folds, so that no CVE of a test fold is seen during training.

## Citation

If you use this benchmark, the labels or the code, please cite the following paper:

```bibtex

```

## Contact

If you use this data, please contact me at **kmkhol01@gmail.com**.

## Licence

Code: MIT licence (see `LICENSE`). Benchmark labels and derived features in `data/`: CC BY 4.0.
Third-party datasets keep their own licences and are not redistributed here.

## Author

Mohammed Tawfik, Department of Cyber Security, Ajloun National University, Jordan
(kmkhol01@gmail.com, ORCID 0000-0002-1227-387X)
