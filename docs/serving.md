# How the local AI is served

## Ollama

[Ollama](https://ollama.com) runs both the agent model and the judge model. It loads
quantized GGUF weights, keeps them in GPU memory, and serves an HTTP API on port 11434. The
compose file pins `ollama/ollama:0.34.4`, with these settings:

- `OLLAMA_MAX_LOADED_MODELS=2`: the agent and the judge stay resident together, so neither
  pays a reload on every escalation.
- `keep_alive: 30m` on each request: idle models unload after 30 minutes.
- **No published port.** Only the gateway talks to Ollama, over the `outside` network.
  Nothing on the sandbox network can reach it.

The gateway uses Ollama's native `/api/chat`:

- **Agent turns:** `think: true` plus the tool list. The response's `message.thinking` is
  the reasoning channel, `message.tool_calls` are the tool calls, and `message.content` is
  the answer.
- **Judge calls:** `format: <JSON schema>` and temperature 0. Ollama constrains decoding to
  the schema, so the verdict always parses.

## What you talk to: the gateway

You never call Ollama directly. The gateway (`localhost:8080`, bearer token from `.env`)
offers two ways in.

**1. `POST /v1/agent/run`: full monitoring. Use this.**

```sh
curl -s localhost:8080/v1/agent/run -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"goal": "Summarize /work/notes/meeting.txt"}'
```

- **Who does what:** the gateway runs the whole agent loop and executes the tools in the
  sandbox itself, so every channel is monitored.
- **What comes back:**
  - `status`: completed, halted, withheld, max_steps or error
  - `final`: the answer
  - `blocks`: how many actions were blocked
  - `verdicts`: one line per event
  - `steps`: the full transcript with each verdict

**2. `POST /v1/chat/completions`: OpenAI-compatible.** For chat UIs such as Open WebUI, or
anything that speaks the OpenAI API (`"model": "agent"` uses `AGENT_MODEL`).

- **What it checks:** reasoning, output and proposed tool calls.
- **Its limit:** the client runs the tools, so the gateway can veto a call but cannot watch
  what the client then does.

## Hardware budget

**RTX 3060, 12 GB** (the Linux server):

| Item | VRAM |
|---|---|
| `qwen3:8b` Q4_K_M agent | about 5.2 GB weights + 1 GB KV cache at 8K context |
| `llama3.2:3b` Q4 judge (or your fine-tuned `agent-judge`) | about 2 GB + 0.5 GB |
| Tier 1 encoder (ModernBERT-base) | about 0.6 GB if on GPU (in the gateway it runs on CPU, fine at ~50 ms) |
| **Total** | **about 9 to 10 GB**, leaving headroom |

An 8B judge alongside an 8B agent is too tight. Use a 3B judge with an 8B agent, or a 4B
agent (`qwen3:4b`) with an 8B judge.

**Mac (M4, 16 GB)**: Docker on macOS cannot use the GPU, so Ollama in the container runs
on the CPU. Measured with `qwen3:1.7b` and `llama3.2:1b`, a two-tool task took about
8 minutes. Two ways to make it usable:

- **Native Ollama (Metal, fast):** `brew install ollama && ollama serve`, then
  `ollama pull` the models and set `OLLAMA_URL=http://host.docker.internal:11434` in `.env`.
  The gateway then uses the Mac's GPU. The monitor and its guarantees are unchanged: the
  sandbox still cannot reach Ollama, because the gateway is the only client.
- **Patience:** keep the small models and raise `OLLAMA_TIMEOUT` and `JUDGE_TIMEOUT` in
  `.env`. On CPU, a judge timeout fails closed and shows up as a block.

## Switching models

Edit `AGENT_MODEL` or `JUDGE_MODEL` in `.env`, run `scripts/pull-models.sh`, then
`docker compose up -d gateway`. The agent must support tool calling in Ollama; qwen3,
llama3.1+, gpt-oss and mistral-small do. Reasoning traces come from qwen3, gpt-oss and the
deepseek-r1 distills.
