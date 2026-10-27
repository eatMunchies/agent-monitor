"""Tier 2: an LLM alignment judge served by Ollama.

Out of the box this is a stock instruct model (llama3.2:3b) prompted with
monitor/judge_prompt.py. After fine-tuning (docs/finetuning.md) it is your own
"agent-judge" model, and nothing here changes except JUDGE_MODEL.
"""

from __future__ import annotations

import json
import time

import httpx

from . import judge_prompt
from .types import Decision, Event, Verdict


class Judge:
    def __init__(self, ollama_url: str, model: str, policy: dict, timeout_s: float = 30.0):
        self.url = ollama_url.rstrip("/") + "/api/chat"
        self.model = model
        self.policy_summary = policy.get("summary", "")
        self.min_conf = policy["thresholds"]["judge_block_confidence"]
        self.fail_closed = policy.get("fail_closed", True)
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def check(self, event: Event, flags: list[str]) -> Verdict:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "stream": False,
            "format": judge_prompt.SCHEMA,  # Ollama constrains decoding to this JSON schema
            "options": {"temperature": 0},
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": judge_prompt.SYSTEM},
                {"role": "user", "content": judge_prompt.user_message(event, self.policy_summary, flags)},
            ],
        }
        try:
            r = await self.client.post(self.url, json=body)
            r.raise_for_status()
            out = json.loads(r.json()["message"]["content"])
            decision = out["decision"]
            conf = float(out.get("confidence", 0.5))
            cat = out.get("category", "policy")
            reason = str(out.get("reason", "")).strip() or "no reason given"
        except Exception as exc:  # network error, timeout, bad JSON
            ms = (time.perf_counter() - t0) * 1000
            d = Decision.BLOCK if self.fail_closed else Decision.ALLOW
            return Verdict(d, "judge", 1.0 if self.fail_closed else 0.0,
                           flags + [f"judge unavailable ({type(exc).__name__}), failing {'closed' if self.fail_closed else 'open'}"],
                           "policy", ms)

        ms = (time.perf_counter() - t0) * 1000
        if decision == "block" and conf >= self.min_conf:
            return Verdict(Decision.BLOCK, "judge", conf, flags + [reason], cat, ms)
        if decision == "block":
            # A hesitant block stays an escalation; the cascade decides by channel what that means.
            return Verdict(Decision.ESCALATE, "judge", conf, flags + [f"low-confidence block: {reason}"], cat, ms)
        return Verdict(Decision.ALLOW, "judge", 1.0 - conf, flags + [reason], "none", ms)

    async def close(self) -> None:
        await self.client.aclose()
