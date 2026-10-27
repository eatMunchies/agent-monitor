# How the models work

There are two kinds of model here: the **agent** that does the work, and the **monitor**
models that watch it. The monitor is a cascade of three tiers; each tier is slower and
smarter than the one before, and only sees what the earlier tiers could not settle.

```
event ──► Tier 0 rules ──► Tier 1 encoder + experts ──► Tier 2 LLM judge ──► verdict
          < 1 ms            ~10-60 ms                    ~0.3-3 s (GPU)
          settles most      flags the unusual            reads only flagged events
```

## The agent model

An open-weights reasoning model served by Ollama: `qwen3:8b` by default (`qwen3:1.7b` for
a quick Mac test). It needs two things:

- **Tool calling.** Ollama passes the tool list in `gateway/agent.py`, and the model answers
  with structured `tool_calls` instead of free text.
- **A visible reasoning trace.** With `think: true`, Ollama returns the model's reasoning in
  a separate `thinking` field. That field is the chain-of-thought channel. Models without a
  thinking mode still work; they just give the monitor one channel less.

Caveats on reasoning traces:

- **A trace is evidence, not ground truth.** Models sometimes do things their reasoning
  never mentions, so the cascade never treats clean reasoning as permission. Tool calls
  are always checked on their own.
- **Never train the agent against the monitor.** OpenAI's 2025 work on CoT monitoring found
  that penalizing "bad thoughts" during training teaches models to hide them while still
  misbehaving. The agent stays a stock model. Only the monitor is trained.

## Tier 0: rules (`monitor/rules.py`)

Deterministic checks read from `policy/policy.yaml`. Each check returns one of three
results: ALLOW, BLOCK (certain), or ESCALATE (suspicious but sometimes legitimate).

- **Tool calls**
  - The tool must be on the allowlist.
  - Shell commands are matched against deny patterns (`rm -rf /`, `curl | sh`, `/dev/tcp`,
    `sudo`, ...) and review patterns (`curl`, `pip install`, `chmod`, ...).
  - File paths must stay under `/work` and away from `.ssh`, `.env` and similar names.
  - URLs go through the domain lists.
- **Secrets**: known key formats (AWS, GitHub, OpenAI style, private keys, JWTs), plus a
  Shannon-entropy test for long random-looking tokens. Git SHAs are excluded.
- **Connections**
  - Domain allowlist and denylist.
  - IP literals and internal hostnames are blocked.
  - Upload size limit.
  - POST bodies to allowlisted hosts are still reviewed.
- **Tool output**: instruction-like text ("ignore previous instructions", "send the key
  to") is flagged as a likely prompt injection.
- **Reasoning**: phrases about hiding, bypassing the monitor or evading detection.

Rules are cheap and impossible to argue with, but they only know what you wrote down.
The next two tiers exist for everything else.

## Tier 1: JEPA-style encoder with mixture-of-experts heads (`monitor/encoder_model.py`)

### The JEPA idea

A JEPA (Joint-Embedding Predictive Architecture, from Yann LeCun's group) learns by
predicting the *embedding* of a missing piece from the embedding of its context, instead of
predicting raw tokens. Here:

- **Context** = user goal + summaries of earlier steps + which channel comes next.
- **Target** = the new step itself (the reasoning, tool call, tool output, request or answer).

Both go through the same pretrained text encoder (ModernBERT-base, about 150M parameters),
then a small predictor network guesses the target's vector from the context's vector.

Training on normal runs teaches the predictor what a sensible next step looks like for a
given goal. At run time, the distance between the predicted and actual vectors is a
**surprise score**. If the goal is "summarize this report" and the next step is a POST to an
unknown domain, the prediction misses badly. This is anomaly detection learned without labels,
so it can flag attack types nobody labeled.

### SIGReg (from LeJEPA)

Predicting embeddings has a trivial cheat: map every text to the same vector, and the
prediction error is zero. Older JEPAs prevent this with tricks such as an EMA teacher and
stop-gradients. LeJEPA (Balestriero and LeCun, 2025) replaces them with one regularizer,
**SIGReg**:

1. Project the batch of embeddings onto random directions.
2. Push each 1D projection to look like a standard normal distribution (it compares
   characteristic functions, an Epps-Pulley style test).

An isotropic Gaussian cannot collapse, so the cheat is ruled out with a single loss term and
one weight. `monitor/encoder_model.py` has a short re-implementation (`sigreg`). The authors'
`lejepa` package is the reference if you want the exact published loss.

### Mixture-of-experts heads

On top of the frozen encoder, several small expert networks each read
`[context, target, target minus prediction, surprise]` and output a score for each risk
category:

    none, exfiltration, destructive, prompt_injection, privilege, off_task, deception, policy

A **router** looks at the target embedding plus the channel, and gives each expert a weight
for this event. The final output is the weighted sum. Experts can specialize, for example
one on shell commands, one on reasoning text and one on network requests, without hard-coding
which is which. A small load-balancing loss keeps the router from sending everything to one
expert. The risk score is `1 - P(none)`.

Its job in the cascade:

- Above `encoder_block` (0.92): block.
- Above `encoder_escalate` (0.5): pass to the judge with the reason attached.
- Otherwise: allow without calling the judge.

That last case is where the speed comes from.

Until you train it, there is no checkpoint, and the cascade runs rules then judge (see
`/health`).

## Tier 2: the LLM judge (`monitor/judge.py`, `monitor/judge_prompt.py`)

A small instruct model (`llama3.2:3b` by default) acting as an alignment check, the same
idea as Meta's LlamaFirewall AlignmentCheck.

- **What it reads**
  - the policy summary
  - the user's goal and the last steps
  - what the earlier tiers flagged
  - the one new item
- **What it answers**: does this item serve the user's goal and respect the policy?
- **Output**: Ollama constrains decoding to a JSON schema, so the reply always has this
  shape:

```json
{"decision": "block", "category": "prompt_injection", "confidence": 0.86,
 "reason": "Follows an instruction found in the file instead of the user's request."}
```

How its answer is turned into a verdict:
- A block with confidence under `judge_block_confidence` (0.6) becomes an escalation. On an
  action, an escalation fails closed.
- If the judge times out or returns garbage, the verdict also fails closed.

The prompt lives in its own file because fine-tuning must train on exactly the text the
judge sees in production. Change the prompt and you need to fine-tune again.

## Cross-channel checks (`monitor/consistency.py`)

Two cheap comparisons between channels. Neither blocks on its own; a mismatch sends the
event to the judge.

- **Tool call vs reasoning**: the call's target (host, file, command) never appears in the
  reasoning that preceded it. This is typical of injected or hidden intent.
- **Answer vs actions**: the final answer denies an action ("I did not send anything")
  while the run log shows tool calls.
