"""Turn your own gateway runs into labeled training rows.

Your own traces are the most valuable data: they match your agent model, your tools and
your policy exactly. Collect them by running eval/tasks through the gateway with
MONITOR_MODE=log_only (nothing but network traffic is blocked, every verdict is logged),
then label each event here. The monitor's own verdict is shown as a hint, never trusted.

  python training/label_traces.py logs/audit.jsonl --out data/rows/own.jsonl

Keys: s safe, u unsafe, then a category number for unsafe; k skip; q save and quit.
Already-labeled event ids are skipped on the next run, so you can stop any time.
"""

import argparse
import json
from pathlib import Path

from schema import Row, write_rows, read_rows
from monitor.types import CATEGORIES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_log")
    ap.add_argument("--out", default="data/rows/own.jsonl")
    ap.add_argument("--env", default="own", help="environment name for splitting, e.g. the task suite")
    a = ap.parse_args()

    out = Path(a.out)
    rows = read_rows(out) if out.exists() else []
    done = {r.id for r in rows}
    recs = [json.loads(l) for l in open(a.audit_log, encoding="utf-8") if l.strip()]
    print(f"{len(recs)} events, {len(done)} already labeled")
    cats = " ".join(f"{i}={c}" for i, c in enumerate(CATEGORIES) if i)

    try:
        for i, rec in enumerate(recs):
            ev, v = rec["event"], rec["verdict"]
            rid = f"own-{ev['session_id'][:12]}-{i}"
            if rid in done or ev["session_id"].startswith("eval"):
                continue
            print("\n" + "=" * 80)
            print(f"GOAL: {ev['goal']}\nCHANNEL: {ev['channel']}  monitor said: {v['decision']} ({'; '.join(v['reasons'])[:160]})")
            print("-" * 80 + "\n" + ev["content"][:1500])
            key = input("\n[s]afe [u]nsafe [k]skip [q]uit > ").strip().lower()
            if key == "q":
                break
            if key not in ("s", "u"):
                continue
            cat, reason = "none", ""
            if key == "u":
                c = input(f"category ({cats}) > ").strip()
                cat = CATEGORIES[int(c)] if c.isdigit() and 0 < int(c) < len(CATEGORIES) else "policy"
                reason = input("one-line reason (optional) > ").strip()
            rows.append(Row(rid, "own", a.env, ev["channel"], ev["goal"], ev["content"],
                            1 if key == "u" else 0, cat, ev.get("history", []), reason, ev.get("meta", {})))
    finally:
        print(f"saved {write_rows(rows, out)} rows to {out}")


if __name__ == "__main__":
    main()
