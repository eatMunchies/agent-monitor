"""Download the public datasets into data/raw/. Standard library plus git.

  python3 training/fetch_data.py            # all sources
  python3 training/fetch_data.py rjudge ts  # some

Formats and licenses were checked against the real files on 2026-10-15; see
docs/data.md. Downloaded files are data: normalize.py parses them as JSON and never
executes anything from them.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
HF = "https://huggingface.co/datasets"
TS = "https://media.githubusercontent.com/media/MurrayTom/ToolSafe/main/TS-Bench"  # Git LFS files

SOURCES = {
    # name: list of (url, local path) or ("git", repo url)
    "rjudge": ("git", "https://github.com/Lordog/R-Judge"),
    "injecagent": ("git", "https://github.com/uiuc-kang-lab/InjecAgent"),
    "agentharm": [(f"{HF}/ai-safety-institute/AgentHarm/resolve/main/benchmark/{f}", f) for f in (
        "harmful_behaviors_test_public.json", "harmful_behaviors_validation.json",
        "benign_behaviors_test_public.json", "benign_behaviors_validation.json")],
    "atbench": [(f"{HF}/AI45Research/ATBench/resolve/main/ATBench/test.json", "ATBench_test.json")],
    "tsbench": [(f"{TS}/{d}/{f}.json", f"{d}_{f}.json") for d, fs in {
        "agentdojo-traj": ["banking", "slack", "travel", "workspace"],
        "agentharm-traj": ["benign_steps", "harmful_steps"],
        "asb-traj/test": ["DPI_attack_success", "OPI_attack_success", "atttack_failure"],  # sic
    }.items() for f in fs],
}


def fetch(name: str) -> None:
    dest = RAW / name
    spec = SOURCES[name]
    if spec[0] == "git":
        if (dest / ".git").exists():
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "-q"], check=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", "-q", spec[1], str(dest)], check=True)
        print(f"{name}: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    for url, local in spec:
        path = dest / local.replace("/", "_")
        if path.exists():
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "agent-monitor-fetch"})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    print(f"{name}: {dest} ({len(spec)} files)")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    for n in names:
        fetch(n)
