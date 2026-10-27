"""The checker: runs each event through rules, then the encoder, then the judge.

Cheap tiers go first and settle most events. A tier hands an event on only when it is
unsure, so the slow LLM judge sees a small fraction of traffic. See docs/checker.md.

Modes (MONITOR_MODE env var):
  cascade     rules -> encoder -> judge only when flagged (default)
  judge_all   rules -> judge on every event (the "Llama baseline" to compare against)
  rules_only  rules alone, no models
  log_only    full cascade, but only network blocks are enforced; everything else is
              logged as "would have blocked". Use it to collect traces for training.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .encoder import EncoderScorer
from .judge import Judge
from .rules import Rules
from .types import Channel, Decision, Event, Verdict

# Channels where an action is about to happen, so an unresolved doubt must stop it.
ENFORCED = {Channel.TOOL_CALL, Channel.CONNECTION, Channel.OUTPUT}
MODES = {"cascade", "judge_all", "rules_only", "log_only"}


class AuditLog:
    """Append-only JSONL record of every event and verdict, plus a readable stdout line."""

    def __init__(self, path: str | Path, max_content: int = 4000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_content = max_content

    def write(self, event: Event, verdict: Verdict, enforced: Verdict) -> None:
        e = event.to_dict()
        e["content"] = e["content"][: self.max_content]
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": e,
            "verdict": verdict.to_dict(),
            "enforced": enforced.decision.value,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{event.session_id[:8]} {enforced.line(event)}", flush=True)


class Cascade:
    def __init__(self, policy: dict, rules: Rules, encoder: EncoderScorer, judge: Judge,
                 audit: AuditLog, mode: str = "cascade"):
        if mode not in MODES:
            raise ValueError(f"MONITOR_MODE must be one of {sorted(MODES)}, got {mode!r}")
        self.policy, self.rules, self.encoder, self.judge = policy, rules, encoder, judge
        self.audit, self.mode = audit, mode
        self.fail_closed = policy.get("fail_closed", True)

    async def check(self, event: Event, extra_flags: list[str] | None = None) -> Verdict:
        """Judge one event, write it to the audit log, and return the verdict to enforce.

        extra_flags are reasons from outside the tiers (for example a cross-channel
        mismatch). Any flag forces the event up to the judge.
        """
        t0 = time.perf_counter()
        try:
            verdict = await self._decide(event, list(extra_flags or []))
        except Exception as exc:  # a bug in a tier must not let an action through
            d = Decision.BLOCK if self.fail_closed else Decision.ALLOW
            verdict = Verdict(d, "error", 1.0, [f"monitor error {type(exc).__name__}: {exc}"], "policy")
        verdict.latency_ms = (time.perf_counter() - t0) * 1000

        enforced = verdict
        if self.mode == "log_only" and event.channel != Channel.CONNECTION and verdict.decision != Decision.ALLOW:
            enforced = replace(verdict, decision=Decision.ALLOW,
                               reasons=[f"log_only, would have been {verdict.decision.value}"] + verdict.reasons)
        self.audit.write(event, verdict, enforced)
        return enforced

    async def _decide(self, event: Event, flags: list[str]) -> Verdict:
        v0 = self.rules.check(event)
        if v0.decision == Decision.BLOCK:
            return v0
        if v0.decision == Decision.ESCALATE:
            flags += v0.reasons
        if self.mode == "rules_only":
            if not flags:
                return v0
            return self._resolve(event, Verdict(Decision.ESCALATE, "rules", 0.5, flags, v0.category))

        v1 = None
        if self.mode != "judge_all":
            v1 = self.encoder.score(event)
            if v1 and v1.decision == Decision.BLOCK:
                return v1
            if v1 and v1.decision == Decision.ESCALATE:
                flags += v1.reasons

        if self.mode != "judge_all" and not flags:
            return Verdict(Decision.ALLOW, v1.tier if v1 else "rules", v1.score if v1 else 0.0,
                           (v1.reasons if v1 else []) or v0.reasons)
        return self._resolve(event, await self.judge.check(event, flags))

    def _resolve(self, event: Event, v: Verdict) -> Verdict:
        """Turn a leftover ESCALATE into a final decision for this channel."""
        if v.decision != Decision.ESCALATE:
            return v
        if event.channel in ENFORCED and self.fail_closed:
            return replace(v, decision=Decision.BLOCK, reasons=v.reasons + ["unresolved doubt on an action, failing closed"])
        # Reasoning and tool output cannot be "blocked" as actions. They stay flagged,
        # and the agent loop redacts a flagged tool output before the model reads it.
        return v
