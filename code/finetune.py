"""Fine-tune the encoder on CVE-grouped folds and write calibration/test scores.

Configurations (fixed grid):
  full        gate=sdpa, pre-trained, both references of the target's architecture
  xarch       as full, but training targets are amd64 only (test on ARM/MIPS)
  x86ref      as full, but the references are always amd64 builds (cross-architecture references)
  nogate / valuegate / sharedgate   gate ablations (each with its own pre-training)
  vulnonly    vulnerable reference only
  nopretrain  encoder trained from scratch
  fullbin     as full, but folds split by binary version (CVEs seen in training)
Usage: python finetune.py --config full --fold 0
Output: runs/<config>/fold<k>.csv  (split, cve, arch, kind, label, score)
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, build_vocab, load_functions, pad_batch
from model import FunctionEncoder, PatchHead

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
CONFIGS = {  # ref: "same" = reference builds of the target's architecture, "x86" = amd64 references
    "full": dict(gate="sdpa", pre=True, refs="both", train_arch=None, ref="same"),
    "xarch": dict(gate="sdpa", pre=True, refs="both", train_arch="amd64", ref="same"),
    "x86ref": dict(gate="sdpa", pre=True, refs="both", train_arch=None, ref="x86"),
    "nogate": dict(gate="none", pre=True, refs="both", train_arch=None, ref="same"),
    "valuegate": dict(gate="value", pre=True, refs="both", train_arch=None, ref="same"),
    "sharedgate": dict(gate="shared", pre=True, refs="both", train_arch=None, ref="same"),
    "vulnonly": dict(gate="sdpa", pre=True, refs="vuln", train_arch=None, ref="same"),
    "nopretrain": dict(gate="sdpa", pre=False, refs="both", train_arch=None, ref="same"),
    # binary-level split (CVEs seen in training), for comparison with earlier studies
    "fullbin": dict(gate="sdpa", pre=True, refs="both", train_arch=None, ref="same", split="bin"),
}


def batches(rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    seed = 1000 + a.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = os.path.join(RUNS, a.config)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(ROOT, "labels", f"triplets_{cfg['ref']}.csv"), newline="") as fh:
        trip = list(csv.DictReader(fh))
    by_bin = cfg.get("split") == "bin"
    fold = json.load(open(os.path.join(ROOT, "labels", "folds_bin.json" if by_bin else "folds.json")))[a.fold]
    split_of = {c: s for s in ("train", "calib", "test") for c in fold[s]}
    for t in trip:
        t["split"] = split_of[f"{t['project']}|{t['version']}" if by_bin else t["cve"]]
    train = [t for t in trip if t["split"] == "train" and (cfg["train_arch"] is None or t["arch"] == cfg["train_arch"])]
    evalset = [t for t in trip if t["split"] in ("calib", "test")]
    keys = {t[k] for t in trip for k in ("target", "vref", "pref")}
    vocab = build_vocab()
    funcs, _ = load_functions("debian_funcs.jsonl", vocab, keep=keys)

    dev = "cuda"
    enc = FunctionEncoder(len(vocab), gate=cfg["gate"]).to(dev)
    if cfg["pre"]:
        ck = torch.load(os.path.join(RUNS, f"pretrain_{cfg['gate']}.pt"), map_location=dev)
        enc.load_state_dict(ck["state"])
    head = PatchHead(refs=cfg["refs"]).to(dev)
    lr_enc = 1e-4 if cfg["pre"] else 3e-4
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": lr_enc},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [lr_enc, 1e-3], total_steps=steps, pct_start=0.1)
    pos = sum(t["label"] == "patched" for t in train)
    pw = torch.tensor((len(train) - pos) / max(pos, 1), device=dev)

    def score(rows, train_mode):
        t = enc(pad_batch([funcs[r["target"]] for r in rows]).to(dev))
        v = enc(pad_batch([funcs[r["vref"]] for r in rows]).to(dev))
        p = enc(pad_batch([funcs[r["pref"]] for r in rows]).to(dev)) if cfg["refs"] == "both" else None
        return head(t, v, p)

    t0 = time.time()
    for ep in range(a.epochs):
        enc.train()
        head.train()
        tot = n = 0
        for b in batches(train, a.batch, True):
            y = torch.tensor([r["label"] == "patched" for r in b], dtype=torch.float, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = score(b, True)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
        print(f"{a.config} fold{a.fold} epoch {ep} loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)

    enc.eval()
    head.eval()
    with open(os.path.join(out_dir, f"fold{a.fold}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "cve", "project", "function", "arch", "version", "kind", "size", "label", "score"])
        with torch.no_grad():
            for b in batches(evalset, 128, False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = torch.sigmoid(score(b, False).float()).cpu().numpy()
                for r, sc in zip(b, s):
                    w.writerow([r["split"], r["cve"], r["project"], r["function"], r["arch"], r["version"],
                                r["kind"], r["size"], int(r["label"] == "patched"), f"{sc:.6f}"])
    if a.config in ("full", "xarch", "x86ref"):  # kept for the external tests and the figures
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "gate": cfg["gate"], "refs": cfg["refs"]},
                   os.path.join(out_dir, f"fold{a.fold}.pt"))
    print("done", a.config, a.fold, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
