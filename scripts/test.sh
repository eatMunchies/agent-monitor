#!/bin/sh
# Unit tests for the monitor package, in a throwaway container (no models needed).
set -e
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD":/src:ro -w /src -e PYTHONDONTWRITEBYTECODE=1 python:3.12-slim-bookworm sh -c \
  "pip install -q --root-user-action=ignore -r gateway/requirements.txt pytest==8.3.5 && python -m pytest -q -p no:cacheprovider tests"
