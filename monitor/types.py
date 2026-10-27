"""Core data shapes shared by the gateway, the training scripts and the eval scripts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class Channel(str, Enum):
    """Where in an agent step the monitored text came from."""

    COT = "cot"                  # the model's reasoning ("thinking") text
    OUTPUT = "output"            # the final answer shown to the user
    TOOL_CALL = "tool_call"      # a tool the model wants to run, as JSON {name, arguments}
    TOOL_RESULT = "tool_result"  # what a tool returned (untrusted, may carry injections)
    CONNECTION = "connection"    # an outbound network request seen by the egress proxy


class Decision(str, Enum):
    ALLOW = "allow"
    ESCALATE = "escalate"  # not sure: ask the next tier
    BLOCK = "block"


# Risk categories the encoder heads and the judge both use. Keep in sync with
# training/schema.py, which maps dataset labels onto these names.
CATEGORIES = [
    "none",
    "exfiltration",       # sending data, files or secrets off the box
    "destructive",        # deleting or corrupting data or the system
    "prompt_injection",   # following instructions from tool output instead of the user
    "privilege",          # escalating access, touching credentials, escaping the sandbox
    "off_task",           # actions unrelated to the user's goal
    "deception",          # hiding actions, lying in the output, evading the monitor
    "policy",             # anything else the system policy forbids
]


@dataclass
class Event:
    """One thing the monitor is asked to judge.

    session_id ties the event to an agent run so the monitor knows the user's goal.
    history holds short summaries of earlier steps, oldest first.
    meta holds channel-specific fields (for a connection: host, port, method, path).
    """

    session_id: str
    channel: Channel
    content: str
    goal: str = ""
    history: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channel"] = self.channel.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            session_id=d.get("session_id", ""),
            channel=Channel(d["channel"]),
            content=d.get("content", ""),
            goal=d.get("goal", ""),
            history=list(d.get("history", [])),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Verdict:
    """The monitor's answer for one event.

    tier names the deepest tier that decided (rules, encoder, judge).
    score is a risk estimate from 0 (fine) to 1 (certainly bad).
    """

    decision: Decision
    tier: str
    score: float
    reasons: list[str] = field(default_factory=list)
    category: str = "none"
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d

    def line(self, event: Event) -> str:
        """One human-readable log line, for example:

        [BLOCK] tool_call run_shell  tier=rules score=1.00 cat=exfiltration  reason: ...
        """
        what = event.meta.get("tool") or event.meta.get("host") or ""
        reason = "; ".join(self.reasons) if self.reasons else "-"
        return (
            f"[{self.decision.value.upper()}] {event.channel.value} {what}".rstrip()
            + f"  tier={self.tier} score={self.score:.2f} cat={self.category}"
            + f" {self.latency_ms:.0f}ms  reason: {reason}"
        )


def tool_call_content(name: str, arguments: dict) -> str:
    """Serialize a tool call the same way everywhere, so training and serving match."""
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
