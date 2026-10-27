"""The model trained on the PLocator benchmark with CVE-grouped 5-fold
cross-validation (test CVEs never seen in training), evaluated with PLocator's
Same / XO / XC settings and its accuracy metric.

The encoder starts from the cross-architecture pre-training checkpoint. For
each fold the evidence-fusion regression is fitted on out-of-fold predictions
of an inner split of the training CVEs.
Output: results/plocator_cv_{rows.csv,summary.json}
"""
import csv
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import build_vocab, pad_batch, tensorise
from fuse import evidence, fit
from common import MAX_INSN
from model import FunctionEncoder, PatchAttentionHead, PatchHead, patch_masks

HEAD = os.environ.get("PS_HEAD", "pooled")  # "pooled" or "pa" (patch attention)


def trunc(insns):
    return insns if len(insns) <= MAX_INSN else insns[:MAX_INSN // 2] + insns[-MAX_INSN // 2:]


def mask_tensor(idx_lists, L):
    m = torch.zeros(len(idx_lists), L)
    for i, idx in enumerate(idx_lists):
        idx = [j for j in idx if j < L]
        m[i, idx] = 1.0
    return m

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, RES = os.path.join(HERE, "..", "runs"), os.path.join(HERE, "..", "results")


def load_items():
    import plocator_eval as pe
    rows = list(csv.DictReader(open(os.path.join(pe.PL, "repo", "org_data", "dataset_wo_irr.csv"), encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cve"], r["vul_func"])].append(r)
    items = []
    for (cve, func), rs in groups.items():
        ref = {lab: next((r["rel_bin_path"] for r in rs if r["label"] == lab and r["compiler"] == "gcc"
                          and r["optim"] == "O0"), None) for lab in ("vul", "fix")}
        if None in ref.values():
            continue
        rt = {}
        for lab, rel in ref.items():
            nm = pe.sym(rel)[0]
            if func in nm:
                rt[lab] = pe.func_tokens(rel, *nm[func])
        if len(rt) < 2:
            continue
        for r in rs:
            addr = int(r["tgt_func"].replace("sub_", ""), 16)
            size = pe.sym(r["rel_bin_path"])[1].get(addr)
            if not size:
                continue
            setting = "Same" if (r["compiler"], r["optim"]) == ("gcc", "O0") else "XC" if r["compiler"] == "clang" else "XO"
            items.append(dict(cve=cve, function=func, setting=setting, label=int(r["label"] == "fix"),
                              t=pe.func_tokens(r["rel_bin_path"], addr, size), v=rt["vul"], p=rt["fix"]))
    return items


def train_and_score(train, test, vocab, dev, seed, epochs=12):
    torch.manual_seed(seed)
    random.seed(seed)
    enc = FunctionEncoder(len(vocab)).to(dev)
    enc.load_state_dict(torch.load(os.path.join(RUNS, "pretrain_sdpa.pt"), map_location=dev)["state"])
    head = (PatchAttentionHead() if HEAD == "pa" else PatchHead()).to(dev)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 1e-4}, {"params": head.parameters(), "lr": 1e-3}],
                            weight_decay=0.01)
    T = {id(it): [tensorise(it[k], vocab) for k in ("t", "v", "p")] for it in train + test}
    Mk = {id(it): patch_masks(trunc(it["v"]), trunc(it["p"])) for it in train + test}

    def forward(bt):
        if HEAD != "pa":
            z = [enc(pad_batch([T[id(it)][j] for it in bt]).to(dev)) for j in range(3)]
            return head(*z)
        outs = [enc(pad_batch([T[id(it)][j] for it in bt]).to(dev), return_seq=True) for j in range(3)]
        (zt, ht, pt), (zv, hv, pv), (zp, hp, pp) = outs
        mv = mask_tensor([Mk[id(it)][0] for it in bt], hv.shape[1]).to(dev)
        mp = mask_tensor([Mk[id(it)][1] for it in bt], hp.shape[1]).to(dev)
        return head(zt, zv, zp, ht.float(), hv.float(), hp.float(), pt, pv, pp, mv, mp)
    for ep in range(epochs):
        enc.train()
        head.train()
        idx = list(range(len(train)))
        random.shuffle(idx)
        for b in range(0, len(idx), 32):
            bt = [train[i] for i in idx[b:b + 32]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = forward(bt)
            y = torch.tensor([it["label"] for it in bt], dtype=torch.float, device=dev)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    enc.eval()
    head.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(test), 64):
            bt = test[b:b + 64]
            out += torch.sigmoid(forward(bt).float()).cpu().tolist()
    return out


def feats(it, s):
    bag = lambda ins: Counter(" ".join(x) for x in ins)
    s = min(max(s, 1e-5), 1 - 1e-5)
    return [np.log(s / (1 - s))] + evidence(bag(it["t"]), bag(it["v"]), bag(it["p"]))


def main():
    items = load_items()
    vocab = build_vocab()
    dev = "cuda"
    cves = sorted({it["cve"] for it in items})
    random.Random(0).shuffle(cves)
    fold_of = {c: i % 5 for i, c in enumerate(cves)}
    learned, fused = {}, {}
    for k in range(5):
        tr = [it for it in items if fold_of[it["cve"]] != k]
        te = [it for it in items if fold_of[it["cve"]] == k]
        # inner out-of-fold predictions on the training CVEs for the fusion weights
        inner = {c: i % 3 for i, c in enumerate(sorted({it["cve"] for it in tr}))}
        Xf, yf = [], []
        for j in range(3):
            a = [it for it in tr if inner[it["cve"]] != j]
            b = [it for it in tr if inner[it["cve"]] == j]
            for it, s in zip(b, train_and_score(a, b, vocab, dev, seed=10 * k + j)):
                Xf.append(feats(it, s))
                yf.append(it["label"])
        lr = fit(np.array(Xf), np.array(yf))
        for it, s in zip(te, train_and_score(tr, te, vocab, dev, seed=100 + k)):
            learned[id(it)] = s
            fused[id(it)] = float(lr.predict_proba([feats(it, s)])[0, 1])
        print("fold", k, "done", flush=True)
    rows = [{"cve": it["cve"], "function": it["function"], "setting": it["setting"], "label": it["label"],
             "learned": learned[id(it)], "fused": fused[id(it)]} for it in items]
    os.makedirs(RES, exist_ok=True)
    tag = "" if HEAD == "pooled" else "_" + HEAD
    with open(os.path.join(RES, f"plocator_cv{tag}_rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for st in ("Same", "XO", "XC", "All"):
        rs = [r for r in rows if st == "All" or r["setting"] == st]
        y = np.array([r["label"] for r in rs])
        summ[st] = {"n": len(rs)}
        for key in ("learned", "fused"):
            s = np.array([r[key] for r in rs])
            p = s > 0.5
            tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
            summ[st][key] = {"accuracy": float((p == y).mean()), "precision": tp / max(tp + fp, 1),
                             "recall": tp / max(tp + fn, 1), "auroc": float(roc_auc_score(y, s))}
    json.dump(summ, open(os.path.join(RES, f"plocator_cv{tag}_summary.json"), "w"), indent=1)
    for st, v in summ.items():
        print(st, v["n"], {k: {m: round(x, 3) for m, x in v[k].items()} for k in ("learned", "fused")})


if __name__ == "__main__":
    main()
