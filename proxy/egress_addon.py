"""mitmproxy addon: ask the gateway before any request leaves the sandbox.

This is the only way out of the sandbox. Every HTTP request, and every HTTPS request
after mitmproxy decrypts it, is described to the gateway's /check/connection endpoint,
which runs the same rules/encoder/judge cascade as tool calls. A block becomes a 403
here and never reaches the internet.

Run by docker-compose as:
  mitmdump -s /addon/egress_addon.py --set connection_strategy=lazy ...
connection_strategy=lazy means mitmproxy opens no upstream connection until the
request hook has allowed the request, so even the TCP handshake waits for a verdict.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from mitmproxy import http, tcp

GATEWAY = os.environ.get("GATEWAY_CHECK_URL", "http://gateway:8080/check/connection")
TOKEN = os.environ.get("CHECK_TOKEN", "")
SESSION_HEADER = "X-Agent-Session"
BODY_EXCERPT = 4000


def _ask(payload: dict) -> dict:
    req = urllib.request.Request(GATEWAY, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Check-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


class EgressGuard:
    async def request(self, flow: http.HTTPFlow) -> None:
        req = flow.request
        session = req.headers.pop(SESSION_HEADER, "")  # never forward it upstream
        body = req.get_content(strict=False) or b""
        payload = {
            "session_id": session,
            "host": req.pretty_host,
            "port": req.port,
            "method": req.method,
            "path": req.path,
            "body_excerpt": body[:BODY_EXCERPT].decode(errors="replace"),
            "body_bytes": len(body),
        }
        try:
            verdict = await asyncio.to_thread(_ask, payload)
        except Exception as exc:  # gateway down: fail closed
            verdict = {"allow": False, "line": f"[BLOCK] connection {req.pretty_host}  gateway unreachable ({type(exc).__name__})"}
        print(verdict.get("line", ""), flush=True)
        if not verdict.get("allow"):
            reasons = "; ".join(verdict.get("reasons", [])) or "blocked by policy"
            flow.response = http.Response.make(403, f"Blocked by agent monitor: {reasons}\n".encode(),
                                               {"Content-Type": "text/plain"})

    def http_connect(self, flow: http.HTTPFlow) -> None:
        # CONNECT opens a tunnel. Only web ports may be tunneled; the request inside the
        # tunnel is then checked by request() above once mitmproxy decrypts it.
        if flow.request.port not in (80, 443):
            print(f"[BLOCK] connection CONNECT {flow.request.pretty_host}:{flow.request.port} (only ports 80 and 443)", flush=True)
            flow.response = http.Response.make(403, b"Blocked by agent monitor: only ports 80 and 443\n")

    def tcp_message(self, flow: tcp.TCPFlow) -> None:
        # Raw TCP inside an allowed tunnel (for example ssh on port 443) is never HTTP, so
        # it never passed request(). Drop the bytes and kill the flow before they go out.
        flow.messages[-1].content = b""
        if flow.killable:
            flow.kill()
        print(f"[BLOCK] connection raw TCP to {flow.server_conn.address}", flush=True)


addons = [EgressGuard()]
