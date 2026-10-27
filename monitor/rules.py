"""Tier 0: deterministic rules from policy/policy.yaml.

Rules are fast (well under a millisecond), never hallucinate, and cannot be argued with,
so anything they are sure about is decided here. They return ESCALATE when a pattern is
suspicious but legitimate uses exist, and the cascade passes those on to the models.
"""

from __future__ import annotations

import ipaddress
import json
import math
import posixpath
import re
from collections import Counter
from pathlib import Path

import yaml

from .types import Channel, Decision, Event, Verdict


def load_policy(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def shannon_entropy(s: str) -> float:
    """Bits per character. Random base64 sits near 6, English text near 4."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def host_matches(host: str, domains: list[str]) -> bool:
    """True when host is one of domains or a subdomain of one."""
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


class Rules:
    def __init__(self, policy: dict):
        self.p = policy
        self.allowed_tools = set(policy["tools"]["allowed"])
        self.network_tools = set(policy["tools"].get("network", []))
        self.shell_deny = [(re.compile(d["pattern"], re.IGNORECASE), d["category"])
                           for d in policy["shell"]["deny_patterns"]]
        self.shell_escalate = _compile(policy["shell"]["escalate_patterns"])
        self.file_root = policy["files"]["root"]
        self.file_deny = policy["files"]["deny_substrings"]
        self.file_review = policy["files"].get("review_substrings", [])
        net = policy["network"]
        self.allow_domains = [d.lower() for d in net["allow_domains"]]
        self.deny_domains = [d.lower() for d in net["deny_domains"]]
        self.unknown_domain = net.get("unknown_domain", "escalate")
        self.unknown_session = net.get("unknown_session", "allowlist_only")
        self.deny_private = net.get("deny_private", True)
        self.max_upload = net.get("max_upload_bytes", 65536)
        sec = policy["secrets"]
        self.secret_patterns = {k: re.compile(v) for k, v in sec["patterns"].items()}
        self.entropy_min_len = sec.get("entropy_min_len", 32)
        self.entropy_threshold = sec.get("entropy_threshold", 4.3)
        self.injection = _compile(policy["injection"]["patterns"])
        self.cot_escalate = _compile(policy["cot"]["escalate_patterns"])

    # ------------------------------------------------------------------ helpers

    def find_secrets(self, text: str) -> list[str]:
        """Names of secret patterns found in text, plus 'high_entropy' for random tokens."""
        found = [name for name, rx in self.secret_patterns.items() if rx.search(text)]
        for tok in re.findall(r"[A-Za-z0-9+/_=-]{%d,}" % self.entropy_min_len, text):
            # Skip plain words and hex hashes such as git SHAs, which are common and harmless.
            if re.fullmatch(r"[0-9a-f]+", tok) or tok.isalpha():
                continue
            if shannon_entropy(tok) >= self.entropy_threshold:
                found.append("high_entropy")
                break
        return found

    def check_domain(self, host: str, session_known: bool) -> tuple[Decision, str]:
        host = (host or "").lower().rstrip(".")
        if not host:
            return Decision.BLOCK, "no host"
        if self.deny_private:
            try:
                ipaddress.ip_address(host)
                return Decision.BLOCK, f"IP literal {host} (only named, allowlisted hosts)"
            except ValueError:
                pass
            if "." not in host or host.endswith((".local", ".internal", ".lan")):
                return Decision.BLOCK, f"internal host {host}"
        if host_matches(host, self.deny_domains):
            return Decision.BLOCK, f"denylisted domain {host}"
        if host_matches(host, self.allow_domains):
            return Decision.ALLOW, f"allowlisted domain {host}"
        if not session_known and self.unknown_session == "allowlist_only":
            return Decision.BLOCK, f"{host} is not allowlisted and the request has no agent session"
        if self.unknown_domain == "block":
            return Decision.BLOCK, f"{host} is not allowlisted"
        return Decision.ESCALATE, f"{host} is not on the allowlist"

    def _path_problem(self, path: str) -> str | None:
        if not path:
            return "empty path"
        full = posixpath.normpath(path if path.startswith("/") else posixpath.join(self.file_root, path))
        if not (full == self.file_root or full.startswith(self.file_root + "/")):
            return f"path {full} is outside {self.file_root}"
        for bad in self.file_deny:
            if bad in full:
                return f"path {full} touches protected name '{bad}'"
        return None

    # ------------------------------------------------------------------ channels

    def check(self, event: Event) -> Verdict:
        handler = {
            Channel.TOOL_CALL: self._tool_call,
            Channel.CONNECTION: self._connection,
            Channel.COT: self._cot,
            Channel.OUTPUT: self._output,
            Channel.TOOL_RESULT: self._tool_result,
        }[event.channel]
        return handler(event)

    def _tool_call(self, e: Event) -> Verdict:
        try:
            call = json.loads(e.content)
            name, args = call["name"], call.get("arguments") or {}
        except (ValueError, KeyError, TypeError):
            return Verdict(Decision.BLOCK, "rules", 1.0, ["tool call is not valid JSON"], "policy")
        # Rows from public datasets use their own tool suites, so offline eval marks them
        # foreign_tools and only the tool-independent checks (secrets) apply.
        foreign = bool(e.meta.get("foreign_tools"))
        if name not in self.allowed_tools and not foreign:
            return Verdict(Decision.BLOCK, "rules", 1.0, [f"tool '{name}' is not allowed"], "policy")

        arg_text = json.dumps(args)
        secrets = self.find_secrets(arg_text)
        if secrets and name in self.network_tools:
            return Verdict(Decision.BLOCK, "rules", 1.0,
                           [f"secret ({', '.join(secrets)}) in arguments of network tool {name}"], "exfiltration")

        reasons: list[str] = []
        if secrets:
            reasons.append(f"secret-like value ({', '.join(secrets)}) in arguments")
        if foreign:
            return Verdict(Decision.ESCALATE if secrets else Decision.ALLOW, "rules", 0.5 if secrets else 0.0, reasons)

        if name == "run_shell":
            cmd = str(args.get("command", ""))
            for rx, cat in self.shell_deny:
                if rx.search(cmd):
                    return Verdict(Decision.BLOCK, "rules", 1.0, [f"command matches forbidden pattern /{rx.pattern}/"], cat)
            for rx in self.shell_escalate:
                if rx.search(cmd):
                    reasons.append(f"command matches review pattern /{rx.pattern}/")
                    break
        elif name in ("read_file", "write_file"):
            path = str(args.get("path", ""))
            problem = self._path_problem(path)
            if problem:
                return Verdict(Decision.BLOCK, "rules", 1.0, [problem], "privilege")
            hit = next((w for w in self.file_review if w in path.lower()), None)
            if hit:
                reasons.append(f"path {path} looks like a credential file ('{hit}')")
        elif name == "http_get":
            from urllib.parse import urlsplit
            host = urlsplit(str(args.get("url", ""))).hostname or ""
            decision, why = self.check_domain(host, session_known=True)
            if decision == Decision.BLOCK:
                return Verdict(Decision.BLOCK, "rules", 1.0, [why], "exfiltration")
            if decision == Decision.ESCALATE:
                reasons.append(why)

        if reasons:
            return Verdict(Decision.ESCALATE, "rules", 0.5, reasons, "policy")
        return Verdict(Decision.ALLOW, "rules", 0.0, [])

    def _connection(self, e: Event) -> Verdict:
        m = e.meta
        decision, why = self.check_domain(m.get("host", ""), session_known=bool(e.goal))
        if decision == Decision.BLOCK:
            return Verdict(Decision.BLOCK, "rules", 1.0, [why], "exfiltration")
        body = e.content or ""
        secrets = self.find_secrets(body + " " + m.get("path", ""))
        if secrets:
            return Verdict(Decision.BLOCK, "rules", 1.0, [f"secret ({', '.join(secrets)}) in outbound request"], "exfiltration")
        if int(m.get("body_bytes", 0)) > self.max_upload:
            return Verdict(Decision.BLOCK, "rules", 1.0,
                           [f"upload of {m.get('body_bytes')} bytes exceeds {self.max_upload}"], "exfiltration")
        if decision == Decision.ALLOW:
            # Allowlisted hosts still get reviewed when the agent is sending data, not just fetching.
            if m.get("method", "GET").upper() not in ("GET", "HEAD", "CONNECT", "OPTIONS"):
                return Verdict(Decision.ESCALATE, "rules", 0.4, [f"{m.get('method')} with a body to {m.get('host')}"], "exfiltration")
            return Verdict(Decision.ALLOW, "rules", 0.0, [why])
        return Verdict(Decision.ESCALATE, "rules", 0.5, [why], "policy")

    def _cot(self, e: Event) -> Verdict:
        hits = [rx.pattern for rx in self.cot_escalate if rx.search(e.content)]
        if hits:
            return Verdict(Decision.ESCALATE, "rules", 0.5, [f"reasoning matches /{hits[0]}/"], "deception")
        return Verdict(Decision.ALLOW, "rules", 0.0, [])

    def _output(self, e: Event) -> Verdict:
        secrets = self.find_secrets(e.content)
        if secrets:
            return Verdict(Decision.BLOCK, "rules", 1.0, [f"secret ({', '.join(secrets)}) in final output"], "exfiltration")
        return Verdict(Decision.ALLOW, "rules", 0.0, [])

    def _tool_result(self, e: Event) -> Verdict:
        reasons = []
        hits = [rx.pattern for rx in self.injection if rx.search(e.content)]
        if hits:
            reasons.append(f"tool output contains instruction-like text /{hits[0]}/")
        if self.find_secrets(e.content):
            reasons.append("tool output contains a secret-like value")
        if reasons:
            cat = "prompt_injection" if hits else "exfiltration"
            return Verdict(Decision.ESCALATE, "rules", 0.5, reasons, cat)
        return Verdict(Decision.ALLOW, "rules", 0.0, [])
