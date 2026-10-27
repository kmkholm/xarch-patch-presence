"""Encoder: instruction embedding + Gated Attention Transformer.

Gated Attention follows Qiu et al. (NeurIPS 2025): a query-dependent,
head-specific sigmoid gate multiplies the scaled dot-product attention output
before the output projection,  Y = W_o [ SDPA(Q,K,V) * sigmoid(X W_g) ].
Variants for the ablation study:
  gate="sdpa"    head-specific elementwise gate on the SDPA output (default)
  gate="value"   gate applied to the value projection instead
  gate="shared"  one scalar gate per head (head-shared across channels)
  gate="none"    plain softmax attention (no gate)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, d, heads, gate="sdpa", dropout=0.1):
        super().__init__()
        self.h, self.dk, self.gate = heads, d // heads, gate
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        if gate in ("sdpa", "value"):
            self.g = nn.Linear(d, d)
        elif gate == "shared":
            self.g = nn.Linear(d, heads)
        self.drop = dropout

    def forward(self, x, pad):  # x: B,L,d ; pad: B,L (True = padding)
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        if self.gate == "value":
            v = v * torch.sigmoid(self.g(x)).view(B, L, self.h, self.dk).transpose(1, 2)
        mask = (~pad)[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.drop if self.training else 0.0)
        if self.gate == "sdpa":
            y = y * torch.sigmoid(self.g(x)).view(B, L, self.h, self.dk).transpose(1, 2)
        elif self.gate == "shared":
            y = y * torch.sigmoid(self.g(x)).transpose(1, 2)[..., None]
        return self.o(y.transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d, heads, gate, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = GatedAttention(d, heads, gate, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.dp = nn.Dropout(dropout)

    def forward(self, x, pad):
        x = x + self.dp(self.att(self.n1(x), pad))
        return x + self.dp(self.ff(self.n2(x)))


class FunctionEncoder(nn.Module):
    """Sub-tokens -> instruction vectors (mean of token + slot embeddings) ->
    Gated Attention layers over the instruction sequence -> pooled, L2-normalised
    function embedding."""

    def __init__(self, vocab, d=256, layers=4, heads=8, max_insn=512, max_tok=8, gate="sdpa", dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=0)
        self.slot = nn.Embedding(max_tok, d)
        self.pos = nn.Embedding(max_insn, d)
        self.blocks = nn.ModuleList([Block(d, heads, gate, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Linear(d, 1)
        self.d = d

    def forward(self, toks, return_seq=False):  # toks: B,L,T (0 = pad)
        B, L, T = toks.shape
        tmask = (toks > 0).float()
        e = self.tok(toks) + self.slot.weight[:T][None, None]
        e = (e * tmask[..., None]).sum(2) / tmask.sum(2, keepdim=True).clamp(min=1)
        pad = tmask.sum(2) == 0
        x = e + self.pos.weight[:L][None]
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)
        w = self.pool(x).squeeze(-1).masked_fill(pad, -1e4).softmax(-1)
        z = F.normalize((w[..., None] * x).sum(1), dim=-1)
        return (z, x, pad) if return_seq else z


def patch_masks(v_insns, p_insns):
    """Instructions removed by the fix (in the vulnerable reference) and added by
    the fix (in the patched reference), from an alignment of the two reference
    builds, which are compiled with the same toolchain and options."""
    import difflib
    sm = difflib.SequenceMatcher(a=[" ".join(i) for i in v_insns], b=[" ".join(i) for i in p_insns], autojunk=False)
    rv, ap = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            rv += range(i1, i2)
        if tag in ("replace", "insert"):
            ap += range(j1, j2)
    return rv, ap


class PatchAttentionHead(nn.Module):
    """Patch-aware head. Besides the pooled embeddings, it measures at the
    instruction level how well the instructions changed by the fix are present
    in the target: for every changed instruction of a reference, the best
    cosine match among the target instructions is taken, and the matches are
    averaged over the patch region (soft coverage). The same is done for the
    unchanged context and from the target side."""

    def __init__(self, d=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.mlp = nn.Sequential(nn.Linear(4 * d + 2 + 9, d), nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def _cov(S, rmask):
        """S: B,Lr,Lt similarities (target pads at -inf); rmask: B,Lr region mask."""
        best = S.max(-1).values.clamp(min=-1)
        n = rmask.sum(-1)
        return (best * rmask).sum(-1) / n.clamp(min=1), (n > 0).float()

    def forward(self, zt, zv, zp, ht, hv, hp, padt, padv, padp, mv, mp):
        q = lambda h: F.normalize(self.proj(h), dim=-1)
        qt, qv, qp = q(ht), q(hv), q(hp)
        Sv = (qv @ qt.transpose(1, 2)).masked_fill(padt[:, None, :], -2.0)
        Sp = (qp @ qt.transpose(1, 2)).masked_fill(padt[:, None, :], -2.0)
        vctx, pctx = (~padv).float() * (1 - mv), (~padp).float() * (1 - mp)
        cv_patch, hv_ = self._cov(Sv, mv)
        cp_patch, hp_ = self._cov(Sp, mp)
        cv_ctx, _ = self._cov(Sv, vctx)
        cp_ctx, _ = self._cov(Sp, pctx)
        # target side: does each target instruction look more like the patched or the vulnerable code?
        tv_ = Sv.masked_fill(padv[:, :, None], -2.0).max(1).values
        tp_ = Sp.masked_fill(padp[:, :, None], -2.0).max(1).values
        tval = (~padt).float()
        t_dir = ((tp_ - tv_) * tval).sum(-1) / tval.sum(-1).clamp(min=1)
        f = torch.cat([zt * zv, zt * zp, (zt - zv).abs(), (zt - zp).abs(),
                       (zt * zv).sum(-1, keepdim=True), (zt * zp).sum(-1, keepdim=True),
                       torch.stack([cp_patch, cv_patch, cp_patch - cv_patch, hp_, hv_, cp_ctx, cv_ctx,
                                    cp_ctx - cv_ctx, t_dir], -1)], -1)
        return self.mlp(f).squeeze(-1)


class PatchHead(nn.Module):
    """Scores 'target is closer to the patched reference than to the vulnerable one'.
    refs="both" uses both references; refs="vuln" is the vulnerable-only ablation."""

    def __init__(self, d=256, refs="both"):
        super().__init__()
        self.refs = refs
        n_in = 4 * d + 2 if refs == "both" else 2 * d + 1
        self.mlp = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, t, v, p=None):
        if self.refs == "both":
            f = torch.cat([t * v, t * p, (t - v).abs(), (t - p).abs(),
                           (t * v).sum(-1, keepdim=True), (t * p).sum(-1, keepdim=True)], -1)
        else:
            f = torch.cat([t * v, (t - v).abs(), (t * v).sum(-1, keepdim=True)], -1)
        return self.mlp(f).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    enc = FunctionEncoder(1100)
    x = torch.randint(1, 1100, (2, 100, 8))
    x[1, 50:] = 0
    print(enc(x).shape, n_params(enc), math.isfinite(enc(x).sum().item()))
    for g in ("none", "value", "shared"):
        print(g, FunctionEncoder(1100, gate=g)(x).shape)
