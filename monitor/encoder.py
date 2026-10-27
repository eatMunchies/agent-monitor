"""Tier 1 runtime: load the trained encoder checkpoint and score events.

If torch is not installed in the gateway image, or models/encoder/ has no checkpoint
yet, the scorer reports not-ready and the cascade skips straight from rules to judge.
That is the expected state until you have trained one (docs/finetuning.md).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .types import CATEGORIES, Decision, Event, Verdict


class EncoderScorer:
    def __init__(self, model_dir: str | Path, policy: dict):
        self.ready = False
        self.why_not = ""
        self.esc = policy["thresholds"]["encoder_escalate"]
        self.block = policy["thresholds"]["encoder_block"]
        model_dir = Path(model_dir)
        ckpt, cfg_path = model_dir / "model.pt", model_dir / "config.json"
        if not ckpt.exists():
            self.why_not = f"no checkpoint at {ckpt}"
            return
        try:
            import torch
            from transformers import AutoTokenizer

            from .encoder_model import CHANNELS, MonitorModel, context_text, target_text
        except ImportError as exc:
            self.why_not = f"torch/transformers not installed ({exc.name})"
            return
        cfg = json.loads(cfg_path.read_text())
        self.torch = torch
        self.ctx_text, self.tgt_text, self.channels = context_text, target_text, CHANNELS
        self.tok = AutoTokenizer.from_pretrained(cfg["base"])
        self.model = MonitorModel(cfg["base"], cfg["dim"], cfg["n_experts"])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.max_len = cfg.get("max_len", 256)
        # Surprise is reported as a z-score against benign training data.
        self.s_mean, self.s_std = cfg.get("surprise_mean", 0.0), cfg.get("surprise_std", 1.0)
        self.ready = True

    def score(self, event: Event) -> Verdict | None:
        if not self.ready:
            return None
        t0 = time.perf_counter()
        torch = self.torch
        enc = lambda s: self.tok([s], truncation=True, max_length=self.max_len, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self.model(enc(self.ctx_text(event)), enc(self.tgt_text(event)),
                             torch.tensor([self.channels.index(event.channel.value)]))
            probs = torch.softmax(out["logits"], -1)[0]
        risk = float(1.0 - probs[0])  # index 0 is "none"
        cat = CATEGORIES[int(probs[1:].argmax()) + 1]
        z = (float(out["surprise"][0]) - self.s_mean) / max(self.s_std, 1e-6)
        reasons = [f"encoder risk {risk:.2f} ({cat}), surprise z={z:.1f}"]
        decision = Decision.BLOCK if risk >= self.block else Decision.ESCALATE if risk >= self.esc else Decision.ALLOW
        return Verdict(decision, "encoder", risk, reasons, cat if decision != Decision.ALLOW else "none",
                       (time.perf_counter() - t0) * 1000)
