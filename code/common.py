"""Shared helpers: vocabulary, function tensorisation, batching."""
import json
import os
from collections import Counter

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FUNC = os.path.join(ROOT, "functions")
MAX_INSN, MAX_TOK = 512, 8


def build_vocab(min_count=2):
    path = os.path.join(FUNC, "vocab.json")
    if os.path.exists(path):
        return json.load(open(path))
    c = Counter()
    for fn in ("pretrain_funcs.jsonl", "debian_funcs.jsonl"):
        for line in open(os.path.join(FUNC, fn)):
            for ins in json.loads(line)["insns"]:
                c.update(ins[:MAX_TOK])
    vocab = {"<pad>": 0, "<unk>": 1}
    for t, n in c.most_common():
        if n >= min_count:
            vocab[t] = len(vocab)
    json.dump(vocab, open(path, "w"))
    return vocab


def tensorise(insns, vocab):
    if len(insns) > MAX_INSN:  # keep prologue and epilogue regions
        insns = insns[:MAX_INSN // 2] + insns[-MAX_INSN // 2:]
    x = torch.zeros(len(insns), MAX_TOK, dtype=torch.int16)
    for i, ins in enumerate(insns):
        ids = [vocab.get(t, 1) for t in ins[:MAX_TOK]]
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
    return x


def load_functions(fn, vocab, keep=None):
    out, meta = {}, {}
    for line in open(os.path.join(FUNC, fn)):
        d = json.loads(line)
        if keep is not None and d["key"] not in keep:
            continue
        out[d["key"]] = tensorise(d["insns"], vocab)
        meta[d["key"]] = {k: v for k, v in d.items() if k not in ("insns",)}
    return out, meta


def pad_batch(seqs):
    L = max(s.shape[0] for s in seqs)
    x = torch.zeros(len(seqs), L, MAX_TOK, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = s.long()
    return x
