"""Tier 1 model: a JEPA-style text encoder with mixture-of-experts risk heads.

Needs torch and transformers, so the gateway imports this lazily and only when a
trained checkpoint exists (see monitor/encoder.py). Training lives in training/.

How it works (longer version in docs/models.md):

1. A pretrained text encoder (ModernBERT by default) turns two texts into vectors:
   the CONTEXT (user goal + earlier steps + which channel is next) and the
   TARGET (the new reasoning, tool call, tool output, request or answer).
2. A small predictor guesses the target vector from the context vector. That is the
   JEPA idea: predict in embedding space, not word by word. Trained on normal runs,
   the predictor learns what a sensible next step looks like for a given goal, so a
   large prediction error ("surprise") means the step does not fit the goal.
3. SIGReg (from LeJEPA) keeps the embeddings spread out like an isotropic Gaussian,
   which stops the trivial solution where every text maps to the same vector.
4. Mixture-of-experts heads read [context, target, prediction error, surprise] and output
   one logit per risk category. A router picks a soft mix of experts per event, so
   experts can specialize (for example one for shell commands, one for reasoning).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import CATEGORIES, Channel, Event

CHANNELS = [c.value for c in Channel]


def context_text(event: Event) -> str:
    history = " | ".join(event.history[-6:]) or "none"
    return f"goal: {event.goal}\nearlier: {history}\nnext: {event.channel.value}"


def target_text(event: Event) -> str:
    return f"{event.channel.value}: {event.content}"


def sigreg(z: torch.Tensor, n_proj: int = 128, n_t: int = 17) -> torch.Tensor:
    """Sketched isotropic Gaussian regularizer, a small re-implementation of LeJEPA's SIGReg.

    Projects the batch onto random unit directions and compares each 1D projection's
    empirical characteristic function with a standard normal's (an Epps-Pulley style test).
    Zero when every direction looks N(0, 1). The reference implementation is the
    `lejepa` package from the paper's authors (github.com/rbalestr-lab/lejepa); swap it in
    if you want the exact published loss.

    The statistic is integrated over t and scaled by the batch size, as Epps-Pulley is.
    Without that scaling it was roughly 50x too weak at batch 8, and the first overnight run
    collapsed: every embedding shrank toward one point, prediction MSE went to about 0, and
    "surprise" carried no signal.

    z: (batch, dim) embeddings. Returns a scalar loss.
    """
    dirs = torch.randn(z.shape[1], n_proj, device=z.device, dtype=z.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    proj = z @ dirs                                    # (batch, n_proj)
    t = torch.linspace(-3.0, 3.0, n_t, device=z.device, dtype=z.dtype)
    x = proj.unsqueeze(-1) * t                         # (batch, n_proj, n_t)
    ecf_re, ecf_im = torch.cos(x).mean(0), torch.sin(x).mean(0)
    gauss = torch.exp(-0.5 * t ** 2)                   # characteristic function of N(0, 1)
    err = (ecf_re - gauss) ** 2 + ecf_im ** 2
    stat = torch.trapezoid(err * gauss, t, dim=-1)     # (n_proj,)
    return z.shape[0] * stat.mean()


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MonitorModel(nn.Module):
    def __init__(self, base: str = "answerdotai/ModernBERT-base", dim: int = 256, n_experts: int = 4):
        super().__init__()
        from transformers import AutoModel

        self.base_name = base
        self.backbone = AutoModel.from_pretrained(base)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, dim)
        self.chan = nn.Embedding(len(CHANNELS), dim)
        self.predictor = _mlp(2 * dim, 2 * dim, dim)
        feat = 3 * dim + 1
        self.router = nn.Linear(2 * dim, n_experts)
        self.experts = nn.ModuleList([_mlp(feat, dim, len(CATEGORIES)) for _ in range(n_experts)])

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.proj(pooled)

    def forward(self, ctx: dict, tgt: dict, channel_idx: torch.Tensor) -> dict:
        zc = self.embed(ctx["input_ids"], ctx["attention_mask"])
        zt = self.embed(tgt["input_ids"], tgt["attention_mask"])
        c = self.chan(channel_idx)
        zhat = self.predictor(torch.cat([zc, c], -1))
        surprise = ((zhat - zt) ** 2).mean(-1, keepdim=True)
        feats = torch.cat([zc, zt, zhat - zt, surprise], -1)
        gate = F.softmax(self.router(torch.cat([zt, c], -1)), -1)            # (batch, n_experts)
        expert_logits = torch.stack([e(feats) for e in self.experts], 1)   # (batch, n_experts, n_cat)
        logits = (gate.unsqueeze(-1) * expert_logits).sum(1)
        return {"zc": zc, "zt": zt, "zhat": zhat, "surprise": surprise.squeeze(-1), "logits": logits, "gate": gate}


def jepa_loss(out: dict, lam: float = 0.5) -> torch.Tensor:
    """Prediction loss in embedding space plus SIGReg on both views (no EMA teacher, no stop-grad)."""
    pred = F.mse_loss(out["zhat"], out["zt"])
    return pred + lam * (sigreg(out["zc"]) + sigreg(out["zt"]))
