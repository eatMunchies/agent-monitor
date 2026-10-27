#!/bin/sh
# Run one monitored agent task and print the verdict lines and the final answer.
#   scripts/run-task.sh "Count the lines in every .py file under /work"
set -e
cd "$(dirname "$0")/.."
TOKEN=$(grep '^GATEWAY_TOKEN=' .env | cut -d= -f2)
PORT=$(grep '^GATEWAY_PORT=' .env | cut -d= -f2); PORT=${PORT:-8080}
python3 - "$1" "$TOKEN" "$PORT" <<'PY'
import json, sys, urllib.request
goal, token, port = sys.argv[1:4]
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/agent/run", method="POST",
    data=json.dumps({"goal": goal}).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
res = json.load(urllib.request.urlopen(req, timeout=900))
for line in res["verdicts"]:
    print(line)
print(f"\nstatus: {res['status']}  blocks: {res['blocks']}\n\n{res['final']}")
PY
