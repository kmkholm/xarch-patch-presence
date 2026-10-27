"""Cross-architecture contrastive pre-training of the function encoder.

Positives are two builds of the same function (same project and symbol
name), preferably on different architectures; all other functions in the
batch are negatives (symmetric InfoNCE, temperature 0.07).
Usage: python pretrain.py --gate sdpa --epochs 8
Output: runs/pretrain_<gate>.pt
"""
import argparse
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from common import build_vocab, load_functions, pad_batch
from model import FunctionEncoder

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="sdpa")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(RUNS, exist_ok=True)
    vocab = build_vocab()
    funcs, meta = load_functions("pretrain_funcs.jsonl", vocab)
    by_name = defaultdict(list)
    for k, m in meta.items():
        by_name[m["name"]].append(k)
    names = [n for n, ks in by_name.items() if len(ks) >= 2]
    print("functions", len(funcs), "names", len(names), "vocab", len(vocab), flush=True)
    dev = "cuda"
    enc = FunctionEncoder(len(vocab), gate=a.gate).to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4, weight_decay=0.01)
    steps = a.epochs * (len(names) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(names)
        tot = 0.0
        for b in range(len(names) // a.batch):
            xa, xb = [], []
            for n in names[b * a.batch:(b + 1) * a.batch]:
                ks = by_name[n]
                k1 = random.choice(ks)
                other = [k for k in ks if meta[k]["arch"] != meta[k1]["arch"]] or [k for k in ks if k != k1]
                xa.append(funcs[k1])
                xb.append(funcs[random.choice(other)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                za, zb = enc(pad_batch(xa).to(dev)), enc(pad_batch(xb).to(dev))
                logits = za.float() @ zb.float().T / 0.07
                lab = torch.arange(len(xa), device=dev)
                loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            step += 1
        nb = len(names) // a.batch
        print(f"epoch {ep} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": enc.state_dict(), "gate": a.gate, "vocab": len(vocab)},
               os.path.join(RUNS, f"pretrain_{a.gate}.pt"))
    print("saved", flush=True)


if __name__ == "__main__":
    main()
