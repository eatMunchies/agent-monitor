#!/bin/sh
# Pull the agent and judge models named in .env into the Ollama container.
set -e
cd "$(dirname "$0")/.."
. ./.env
for m in "$AGENT_MODEL" "$JUDGE_MODEL"; do
  # Large pulls sometimes drop with "unexpected EOF"; Ollama resumes, so just retry.
  for try in 1 2 3; do docker compose exec ollama ollama pull "$m" && break; echo "retrying $m"; done
done
