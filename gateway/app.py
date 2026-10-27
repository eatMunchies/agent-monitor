"""Gateway HTTP API.

  POST /v1/agent/run          run a monitored agent task (the main entry point)
  POST /v1/chat/completions   OpenAI-compatible chat with reasoning and output monitoring
  POST /v1/monitor/check      judge one event directly (used by eval/offline_eval.py)
  POST /v1/sandbox/seed       write files into the sandbox before a test run
  POST /check/connection      asked by the egress proxy before each outbound request
  GET  /health                which tiers are live

/v1/* needs "Authorization: Bearer $GATEWAY_TOKEN"; /check/connection needs
"X-Check-Token: $CHECK_TOKEN". The sandbox has neither, so the agent cannot call
the gateway to run itself or approve its own traffic.
"""

from __future__ import annotations

import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from monitor.cascade import AuditLog, Cascade
from monitor.encoder import EncoderScorer
from monitor.judge import Judge
from monitor.rules import Rules, load_policy
from monitor.types import Channel, Decision, Event, tool_call_content

from . import agent
from .clients import Ollama, Sandbox
from .config import settings

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not settings.gateway_token or not settings.check_token:
        raise RuntimeError("GATEWAY_TOKEN and CHECK_TOKEN must be set (run scripts/init-env.sh)")
    policy = load_policy(settings.policy_path)
    encoder = EncoderScorer(settings.encoder_dir, policy)
    judge = Judge(settings.ollama_url, settings.judge_model, policy, settings.judge_timeout)
    state.update(
        policy=policy,
        encoder=encoder,
        judge=judge,
        monitor=Cascade(policy, Rules(policy), encoder, judge, AuditLog(settings.audit_log), settings.mode),
        ollama=Ollama(settings.ollama_url, settings.ollama_timeout, settings.agent_num_predict),
        sandbox=Sandbox(settings.sandbox_url),
    )
    print(f"gateway up: mode={settings.mode} agent={settings.agent_model} judge={settings.judge_model} "
          f"encoder={'ready' if encoder.ready else 'off (' + encoder.why_not + ')'}", flush=True)
    yield
    await judge.close()


app = FastAPI(title="agent-monitor gateway", lifespan=lifespan)


def _same(a: str, b: str) -> bool:
    return bool(a) and hmac.compare_digest(a.encode(), b.encode())


def require_token(authorization: str = Header("")) -> None:
    if not _same(authorization.removeprefix("Bearer ").strip(), settings.gateway_token):
        raise HTTPException(401, "bad or missing bearer token")


def require_check_token(x_check_token: str = Header("")) -> None:
    if not _same(x_check_token, settings.check_token):
        raise HTTPException(401, "bad or missing check token")


# --------------------------------------------------------------------------- agent

class RunRequest(BaseModel):
    goal: str
    model: str | None = None
    max_steps: int | None = Field(None, ge=1, le=50)


@app.post("/v1/agent/run", dependencies=[Depends(require_token)])
async def run(req: RunRequest) -> dict:
    return await agent.run_agent(
        req.goal, state["ollama"], state["sandbox"], state["monitor"],
        req.model or settings.agent_model, req.max_steps or settings.max_steps,
        state["policy"].get("max_blocks_per_run", 3),
        state["policy"].get("cot_block", "flag"),
    )


# --------------------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    model: str = "agent"
    messages: list[dict]
    tools: list[dict] | None = None


@app.post("/v1/chat/completions", dependencies=[Depends(require_token)])
async def chat(req: ChatRequest) -> dict:
    """OpenAI-shaped chat for clients such as Open WebUI.

    The client runs its own tools here, so this endpoint can only veto tool calls before
    the client sees them; it cannot watch what the client does with them. Use
    /v1/agent/run for full monitoring.
    """
    model = settings.agent_model if req.model in ("agent", "") else req.model
    goal = next((m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"), "")
    sid = "chat-" + uuid.uuid4().hex[:12]
    mon: Cascade = state["monitor"]
    verdicts = []

    msg = await state["ollama"].chat(model, req.messages, req.tools, think=True)
    thinking, content = msg.get("thinking") or "", msg.get("content") or ""

    def ev(ch: Channel, text: str, **meta) -> Event:
        return Event(sid, ch, text, goal, [], meta)

    if thinking:
        e = ev(Channel.COT, thinking)
        v = await mon.check(e)
        verdicts.append(v.line(e))
        if v.decision == Decision.BLOCK:
            content, msg["tool_calls"] = "[response withheld by monitor]", []

    kept = []
    for i, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function", {})
        e = ev(Channel.TOOL_CALL, tool_call_content(fn.get("name", ""), fn.get("arguments") or {}), tool=fn.get("name"))
        v = await mon.check(e)
        verdicts.append(v.line(e))
        if v.decision != Decision.BLOCK:
            kept.append({"id": f"call_{i}", "type": "function",
                         "function": {"name": fn.get("name"), "arguments": json.dumps(fn.get("arguments") or {})}})

    if content:
        e = ev(Channel.OUTPUT, content)
        v = await mon.check(e)
        verdicts.append(v.line(e))
        if v.decision == Decision.BLOCK:
            content = "[answer withheld by monitor: " + "; ".join(v.reasons) + "]"

    message = {"role": "assistant", "content": content}
    if kept:
        message["tool_calls"] = kept
    return {
        "id": sid, "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if kept else "stop"}],
        "monitor": {"verdicts": verdicts},
    }


# --------------------------------------------------------------------------- direct checks

@app.post("/v1/monitor/check", dependencies=[Depends(require_token)])
async def monitor_check(event: dict) -> dict:
    e = Event.from_dict(event)
    v = await state["monitor"].check(e)
    return {"verdict": v.to_dict(), "line": v.line(e)}


class SeedRequest(BaseModel):
    files: dict[str, str]


@app.post("/v1/sandbox/seed", dependencies=[Depends(require_token)])
async def seed(req: SeedRequest) -> dict:
    """Plant files (for example a document with a hidden prompt injection) for a test run."""
    out = {}
    for path, content in req.files.items():
        out[path] = await state["sandbox"].call("write_file", {"path": path, "content": content}, "seed")
    return {"written": out}


class ConnectionCheck(BaseModel):
    session_id: str = ""
    host: str
    port: int = 443
    method: str = "GET"
    path: str = "/"
    body_excerpt: str = ""
    body_bytes: int = 0


@app.post("/check/connection", dependencies=[Depends(require_check_token)])
async def check_connection(req: ConnectionCheck) -> dict:
    sess = agent.SESSIONS.get(req.session_id)
    e = Event(
        req.session_id or "no-session", Channel.CONNECTION, req.body_excerpt,
        sess.goal if sess else "", list(sess.history) if sess else [],
        {"host": req.host, "port": req.port, "method": req.method, "path": req.path[:300], "body_bytes": req.body_bytes},
    )
    v = await state["monitor"].check(e)
    return {"allow": v.decision != Decision.BLOCK, "line": v.line(e), "reasons": v.reasons}


@app.get("/health")
async def health() -> dict:
    try:
        models = await state["ollama"].tags()
    except Exception as exc:
        models = [f"ollama unreachable: {type(exc).__name__}"]
    enc: EncoderScorer = state["encoder"]
    return {"mode": settings.mode, "agent_model": settings.agent_model, "judge_model": settings.judge_model,
            "encoder": "ready" if enc.ready else f"off: {enc.why_not}", "ollama_models": models}
