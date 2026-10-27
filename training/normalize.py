"""Convert data/raw/* into data/rows/all.jsonl (one Row per monitored event).

  python3 training/normalize.py                   # every source found in data/raw
  python3 training/normalize.py --sources tsbench atbench

Each converter states how it turns that dataset's labels into per-event labels,
because most public sets label a whole trajectory, while the monitor judges one step:

  tsbench     step-level labels already (score 0 / 0.5 / 1). Best fit. Emits the step's
              reasoning (cot) and its tool call, both with the step's label.
  rjudge      trajectory label. Safe runs: every step is safe. Unsafe runs: only the
              LAST agent action is labeled unsafe; earlier steps are dropped as ambiguous.
  atbench     same rule as rjudge (trajectory label, last action of unsafe runs).
  injecagent  no labels, every case is an attack. Emits the user's own tool call (safe),
              the poisoned tool output (unsafe, prompt_injection) and the attacker's tool
              call that would follow (unsafe).
  agentharm   prompts only, no trajectories. Emits each target function as a tool call
              with the prompt as the goal; harmful config unsafe, benign config safe.

Rows from these sets set meta.foreign_tools so the rules tier skips checks that only
make sense for this repo's own tools.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from schema import Row, guess_category, write_rows

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FOREIGN = {"foreign_tools": True}


def parse_args_text(text: str) -> dict:
    text = (text or "").strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(text)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            pass
    return {"raw": text[:2000]} if text else {}


def parse_action(action) -> tuple[str, dict]:
    """Best effort (tool name, arguments) from the many action formats in these datasets."""
    if isinstance(action, dict):
        return str(action.get("name", "unknown")), action.get("arguments") or action.get("args") or {}
    text = str(action or "").strip().strip('"')
    if text.startswith("{"):
        try:
            d = json.loads(text)
            if "name" in d:
                return str(d["name"]), d.get("arguments") or {}
        except ValueError:
            pass
    m = re.search(r"Action:\s*([\w.-]+)", text) or re.match(r"\s*([\w.-]+)", text)
    name = m.group(1) if m else "unknown"
    a = re.search(r"Action Input:\s*(\{.*\})", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    return name, parse_args_text(a.group(1)) if a else {}


def tc(name: str, args: dict) -> str:
    return json.dumps({"name": name, "arguments": args}, sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------- tsbench

def tsbench(src: Path):
    for f in sorted(src.glob("*.json")):
        env = f.stem
        for i, r in enumerate(json.loads(f.read_text())):
            label = float(r.get("score", 0))
            action = r.get("current_action", "")
            thought = re.search(r"Thought:\s*(.*?)(?:\(2\)|Action:|$)", action, re.S)
            name, args = parse_action(action)
            hist = [h.strip() for h in re.split(r"\n(?=\(\d\) Thought:)", r.get("history") or "") if h.strip()][-6:]
            hist = [h[:300] for h in hist]
            if "asb" in env or "agentdojo" in env:
                cat = "prompt_injection"
            else:
                cat = guess_category(r.get("instruction", ""))
            base = dict(source="tsbench", env=env, goal=r.get("instruction", ""), label=label,
                        category=cat, history=hist, meta=dict(FOREIGN))
            rid = f"ts-{env}-{r.get('id-interaction', i)}-{r.get('id-segment', 0)}"
            if thought and thought.group(1).strip():
                yield Row(id=rid + "-cot", channel="cot", content=thought.group(1).strip(), **base)
            yield Row(id=rid, channel="tool_call", content=tc(name, args), **base)


# --------------------------------------------------------------------------- rjudge / atbench

def _trajectory_rows(source, env, rid, turns, label, cat, reason, extra_meta=None):
    """Shared by R-Judge and ATBench: both store contents as rounds of user/agent/environment turns."""
    goal = next((t.get("content") or "" for t in turns if t.get("role") == "user"), "")
    agent_idx = [i for i, t in enumerate(turns) if t.get("role") == "agent"]
    if not agent_idx:
        return
    last = agent_idx[-1]
    history: list[str] = []
    meta = dict(FOREIGN, **(extra_meta or {}))
    for i, t in enumerate(turns):
        role = t.get("role")
        if role == "agent":
            name, args = parse_action(t.get("action"))
            keep = label == 0 or i == last
            step_label = label if i == last else 0
            if keep:
                if t.get("thought"):
                    yield Row(f"{rid}-{i}-cot", source, env, "cot", goal, str(t["thought"]), step_label,
                              cat, list(history[-6:]), reason, meta)
                yield Row(f"{rid}-{i}", source, env, "tool_call", goal, tc(name, args), step_label,
                          cat, list(history[-6:]), reason, meta)
            history.append(f"{name}({json.dumps(args, ensure_ascii=False)[:120]})")
        elif role == "environment":
            content = str(t.get("content") or "")
            if label == 0 and content:
                yield Row(f"{rid}-{i}-res", source, env, "tool_result", goal, content, 0, "none",
                          list(history[-6:]), "", meta)
            history.append(f"-> {content[:100]}")
        elif role == "user" and i > 0:
            history.append(f"user: {str(t.get('content'))[:100]}")


def rjudge(src: Path):
    for f in sorted((src / "data").glob("*/*.json")):
        env = f"{f.parent.name}/{f.stem}"
        for r in json.loads(f.read_text()):
            label = float(r.get("label", 0))
            desc = r.get("risk_description") or ""
            cat = "prompt_injection" if r.get("attack_type") == "injection" else guess_category(desc)
            turns = [t for rnd in r.get("contents", []) for t in rnd]
            yield from _trajectory_rows("rjudge", env, f"rj-{f.stem}-{r.get('id')}", turns, label, cat, desc[:300])


def atbench(src: Path):
    for f in sorted(src.glob("*.json")):
        for r in json.loads(f.read_text()):
            label = float(r.get("label", 0))
            traj = r.get("contents") or r.get("content") or [[]]
            text = " ".join(str(r.get(k, "")) for k in ("risk_source", "failure_mode", "real_world_harm", "reason"))
            cat = "prompt_injection" if "injection" in str(r.get("risk_source", "")) else guess_category(text)
            yield from _trajectory_rows("atbench", str(r.get("risk_source", "unknown")),
                                        f"at-{r.get('id', r.get('conv_id'))}", traj[0], label, cat,
                                        str(r.get("reason", ""))[:300])


# --------------------------------------------------------------------------- injecagent

def injecagent(src: Path):
    for kind in ("dh", "ds"):
        f = src / "data" / f"test_cases_{kind}_base.json"
        if not f.exists():
            continue
        for i, r in enumerate(json.loads(f.read_text())):
            goal, user_tool = r["User Instruction"], r["User Tool"]
            env = f"injecagent-{user_tool}"
            args = parse_args_text(r.get("Tool Parameters", ""))
            response = str(r.get("Tool Response", "")).strip('"')
            first = f"{user_tool}({json.dumps(args)[:120]})"
            cat = "exfiltration" if kind == "ds" else guess_category(r.get("Attack Type", "")) or "policy"
            rid = f"ia-{kind}-{i}"
            yield Row(rid + "-user", "injecagent", env, "tool_call", goal, tc(user_tool, args), 0, "none", [], "", dict(FOREIGN))
            yield Row(rid + "-res", "injecagent", env, "tool_result", goal, response, 1, "prompt_injection", [first],
                      "Tool output carries instructions from a third party.", dict(FOREIGN))
            attacker = r.get("Attacker Tools") or ["unknown"]
            yield Row(rid + "-atk", "injecagent", env, "tool_call", goal, tc(attacker[0], {}), 1, cat,
                      [first, f"-> {response[:100]}"], f"Follows the injected instruction: {r.get('Attacker Instruction', '')[:150]}",
                      dict(FOREIGN))


# --------------------------------------------------------------------------- agentharm

def agentharm(src: Path):
    for f in sorted(src.glob("*.json")):
        harmful = f.name.startswith("harmful")
        for r in json.loads(f.read_text()).get("behaviors", []):
            hist: list[str] = []
            for j, fn in enumerate(r.get("target_functions", [])):
                yield Row(f"ah-{f.stem}-{r['id']}-{j}", "agentharm", r.get("category", "unknown"), "tool_call",
                          r.get("prompt", ""), tc(fn, {}), 1 if harmful else 0,
                          "policy" if harmful else "none", list(hist),
                          f"Part of a harmful task ({r.get('category')}: {r.get('name')})." if harmful else "",
                          dict(FOREIGN))
                hist.append(f"{fn}(...)")


CONVERTERS = {"tsbench": tsbench, "rjudge": rjudge, "atbench": atbench, "injecagent": injecagent, "agentharm": agentharm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--sources", nargs="*", default=list(CONVERTERS))
    ap.add_argument("--out", default=str(ROOT / "data" / "rows" / "all.jsonl"))
    a = ap.parse_args()
    rows = []
    for name in a.sources:
        src = Path(a.raw) / name
        if not src.exists():
            print(f"{name:11} skipped (no {src}, run fetch_data.py)")
            continue
        got = list(CONVERTERS[name](src))
        unsafe = sum(r.label >= 0.5 for r in got)
        by_ch = {}
        for r in got:
            by_ch[r.channel] = by_ch.get(r.channel, 0) + 1
        print(f"{name:11} {len(got):6} rows, {unsafe:5} unsafe  {by_ch}")
        rows += got
    print(f"wrote {write_rows(rows, a.out)} rows to {a.out}")


if __name__ == "__main__":
    main()
