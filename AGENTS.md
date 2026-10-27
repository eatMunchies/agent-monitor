# AGENTS.md: agent-monitor

Instructions for AI coding agents. Human docs: README.md and docs/.

- `monitor/` is shared by the gateway, training and eval. Keep it free of FastAPI imports, and keep torch imports lazy (only `encoder_model.py` imports torch at module level, and only `encoder.py` imports that, inside a try).
- `monitor/judge_prompt.py` is the judge's training contract. Changing it means the fine-tuned judge must be retrained; say so when you touch it.
- Fail closed. Any new tier or check must turn errors and timeouts into BLOCK on actions (tool_call, connection, output).
- Never weaken a network control (internal network, proxy, DNS, tokens, sandbox hardening) without the user asking; docs/network.md lists what each one guarantees.
- Do not train or prompt-tune the agent model against the monitor.
- `eval/` and `scripts/` are standard library only, so they run on the Mac's system Python 3.9.
- No em dashes in code, comments or docs.

## Commands

| Check | Command |
|---|---|
| Unit tests | `scripts/test.sh` |
| Rebuild after gateway/monitor changes | `docker compose up -d --build gateway` |
| Proxy addon changes | `docker compose restart egress-proxy` |
| End-to-end | `scripts/run-task.sh "<goal>"`, `python3 eval/agent_eval.py` |
