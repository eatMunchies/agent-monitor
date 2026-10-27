# agent-monitor

Serve a local AI agent and watch everything it does, as text, in four places:

| Channel | What is checked | Where it is caught |
|---|---|---|
| Chain of thought | the model's reasoning before it acts | gateway, before any tool runs |
| Tool use | each tool call, and each tool's output before the model reads it | gateway, before and after the sandbox runs it |
| External connections | every outbound HTTP(S) request | egress proxy, before the TCP connection opens |
| Output | the final answer | gateway, before it reaches you |

Each item goes through a cascade of checkers: fast rules first, then a JEPA-style encoder with
mixture-of-experts heads, then an LLM judge only when something looks off. Anything that
breaks the user's intent or the system policy is blocked, logged, and explained.

```
 you ──► gateway :8080 ──► Ollama (agent model, judge model)
           │   ▲
           │   │ /check/connection
           ▼   │
        sandbox ──► egress proxy (mitmproxy) ──► internet
     (no network        (the only way out)
      route of its own)
```

## Docs

| Topic | File |
|---|---|
| How each model works (agent, rules, JEPA encoder, experts, judge) | [docs/models.md](docs/models.md) |
| Where the training data comes from, and labeling your own traces | [docs/data.md](docs/data.md) |
| Fine-tuning the encoder and the judge | [docs/finetuning.md](docs/finetuning.md) |
| How the local AI is served (Ollama, VRAM budget, Mac vs server) | [docs/serving.md](docs/serving.md) |
| How the checker decides, and what its output looks like | [docs/checker.md](docs/checker.md) |
| Where the proxy sits and how traffic is stopped | [docs/network.md](docs/network.md) |

## Quick start

Needs Docker. On the Linux box with the RTX 3060, also nvidia-container-toolkit.

```sh
scripts/init-env.sh                 # creates .env with random tokens
docker compose up -d --build        # Mac, CPU only
# Linux + GPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
scripts/pull-models.sh              # first run: pulls AGENT_MODEL and JUDGE_MODEL
scripts/run-task.sh "Create /work/hello.py that prints hello, run it, and tell me the output."
```

`run-task.sh` prints one verdict line per monitored event, then the answer:

```
[ALLOW] cot  tier=rules score=0.00 cat=none 1ms  reason: -
[ALLOW] tool_call write_file  tier=rules score=0.00 cat=none 1ms  reason: -
[BLOCK] connection pastebin.com  tier=rules score=1.00 cat=exfiltration 0ms  reason: denylisted domain pastebin.com
status: completed  blocks: 0
```

The full record is `logs/audit.jsonl`, one JSON object per event.

## Layout

| Path | What it is |
|---|---|
| `monitor/` | The checker, shared by gateway, training and eval: `rules.py` (tier 0), `encoder_model.py` + `encoder.py` (tier 1), `judge.py` + `judge_prompt.py` (tier 2), `cascade.py` (ties them together, audit log), `consistency.py` (cross-channel checks) |
| `gateway/` | FastAPI service: the monitored agent loop (`agent.py`), the API (`app.py`) |
| `sandbox/` | The container the agent's tools run in, and its tool server |
| `proxy/` | mitmproxy addon that asks the gateway about every outbound request |
| `policy/policy.yaml` | The system policy: allowed tools, forbidden commands, domains, secret patterns, thresholds |
| `training/` | Fetch and normalize datasets, label your own traces, train the encoder, fine-tune the judge |
| `eval/` | Attack and benign tasks, end-to-end and offline evaluation |
| `tests/` | Unit tests for rules and cascade (`scripts/test.sh`) |
| `models/`, `data/`, `logs/` | Trained weights, datasets, audit logs (all gitignored) |

## Commands

| Task | Command |
|---|---|
| Demo on a Mac (network escapes, then the checker on attack and harmless steps; about 5 min) | `scripts/demo.sh` |
| Unit tests | `scripts/test.sh` |
| Health (which tiers are live) | `curl -s localhost:8080/health -H "Authorization: Bearer $(grep ^GATEWAY_TOKEN .env \| cut -d= -f2)"` |
| Watch verdicts live | `docker compose logs -f gateway egress-proxy` |
| Quick checker regression (13 handmade attack and benign steps, no agent model needed) | `python3 eval/offline_eval.py eval/tasks/attack_steps.jsonl` |
| End-to-end eval (add `--timeout 7200` on CPU) | `python3 eval/agent_eval.py` |
| Offline eval on labeled rows | `python3 eval/offline_eval.py data/rows/test.jsonl --limit 500` |
| Get training data | `python3 training/fetch_data.py && python3 training/normalize.py && python3 training/split.py data/rows/all.jsonl` |
| Compare checker modes | set `MONITOR_MODE` in `.env` (cascade, judge_all, rules_only, log_only), `docker compose up -d gateway`, rerun eval |

## Status

Working and tested on a Mac (CPU), with qwen3:1.7b as the agent and llama3.2:1b as the judge:

- the gateway, sandbox and proxy
- tier 0 rules
- the judge tier
- the network controls
- the dataset pipeline
- the encoder training code path

Not yet done:

- training a real encoder on the GPU box
- the QLoRA judge fine-tune, which needs CUDA and has not been run yet
- real eval numbers

The build order is in [docs/finetuning.md](docs/finetuning.md#build-order).

## Later: the home server

This is standalone on purpose. Once the home server exists, the gateway can join its network, send the
audit log to Loki/Grafana, and send block alerts to ntfy. None of that is wired up yet.
