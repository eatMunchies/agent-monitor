"""Tool server that runs inside the sandbox container. Standard library only.

The gateway calls POST /tool after the monitor approves a tool call. This process has
no network route except the egress proxy, runs as an unprivileged user on a read-only
root filesystem, and can only write to /work and /tmp.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORK = Path("/work").resolve()
MAX_OUT = 8000


def _inside_work(path: str) -> Path:
    p = (WORK / path).resolve() if not path.startswith("/") else Path(path).resolve()
    if p != WORK and WORK not in p.parents:
        raise ValueError(f"{p} is outside /work")
    return p


def run_shell(command: str, session_id: str) -> str:
    env = dict(os.environ, AGENT_SESSION=session_id)
    r = subprocess.run(["bash", "-c", command], cwd=WORK, env=env, capture_output=True,
                       text=True, timeout=30)
    out = (r.stdout + r.stderr)[-MAX_OUT:]
    return f"exit {r.returncode}\n{out}"


def read_file(path: str, **_) -> str:
    return _inside_work(path).read_text(errors="replace")[:MAX_OUT * 2]


def write_file(path: str, content: str, **_) -> str:
    p = _inside_work(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


def http_get(url: str, session_id: str) -> str:
    # urllib reads HTTP(S)_PROXY from the environment, so this goes through the egress proxy.
    # The session header lets the proxy ask the gateway with the right user goal.
    req = urllib.request.Request(url, headers={"X-Agent-Session": session_id, "User-Agent": "agent-sandbox"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"HTTP {r.status}\n" + r.read(MAX_OUT * 2).decode(errors="replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}\n" + e.read(2000).decode(errors="replace")


TOOLS = {"run_shell": run_shell, "read_file": read_file, "write_file": write_file, "http_get": http_get}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/tool":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            fn = TOOLS[req["name"]]
            args = dict(req.get("arguments") or {})
            if fn in (run_shell, http_get):
                args["session_id"] = req.get("session_id", "")
            res = {"ok": True, "output": fn(**args)}
        except subprocess.TimeoutExpired:
            res = {"ok": False, "output": "command timed out after 30s"}
        except Exception as exc:
            res = {"ok": False, "output": f"{type(exc).__name__}: {exc}"}
        body = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass  # the gateway's audit log is the record; keep sandbox logs quiet


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9000), Handler).serve_forever()
