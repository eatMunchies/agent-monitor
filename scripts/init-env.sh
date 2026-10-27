#!/bin/sh
# Create .env from .env.example (if missing) and fill empty tokens with random values.
# Prints nothing secret.
set -e
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
for key in GATEWAY_TOKEN CHECK_TOKEN; do
  if grep -q "^$key=$" .env; then
    val=$(openssl rand -hex 24)
    sed -i.bak "s/^$key=$/$key=$val/" .env && rm -f .env.bak
    echo "set $key"
  fi
done
