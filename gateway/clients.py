"""HTTP clients for Ollama (the models) and the sandbox tool server."""

from __future__ import annotations

import httpx


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str, timeout_s: float = 600.0, num_predict: int = 4096):
        self.url = url.rstrip("/")
        self.num_predict = num_predict
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
                   think: bool = True) -> dict:
        """One non-streaming chat turn. Returns Ollama's message dict, which may hold
        "thinking" (the reasoning trace), "content" and "tool_calls".

        Models without a thinking mode reject think=True, so this retries once without it.
        """
        body = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0.2, "num_predict": self.num_predict}}
        if tools:
            body["tools"] = tools
        if think:
            body["think"] = True
        try:
            r = await self.client.post(self.url + "/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise OllamaError(f"ollama request failed: {type(exc).__name__}") from exc
        if r.status_code == 400 and think and "think" in r.text.lower():
            return await self.chat(model, messages, tools, think=False)
        if r.status_code != 200:
            raise OllamaError(f"ollama {r.status_code}: {r.text[:300]}")
        return r.json()["message"]

    async def tags(self) -> list[str]:
        r = await self.client.get(self.url + "/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


class Sandbox:
    def __init__(self, url: str, timeout_s: float = 60.0):
        self.url = url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def call(self, name: str, arguments: dict, session_id: str) -> str:
        try:
            r = await self.client.post(self.url + "/tool",
                                       json={"name": name, "arguments": arguments, "session_id": session_id})
            data = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            return f"tool error: sandbox unreachable ({type(exc).__name__})"
        return data.get("output", "") if data.get("ok") else f"tool error: {data.get('output', '')}"
