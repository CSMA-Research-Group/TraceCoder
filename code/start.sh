#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
runtime="${TRACECODER_PYTHON:-.venv/bin/python}"
if [ ! -x "$runtime" ] && [ -z "${TRACECODER_PYTHON:-}" ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
exec "$runtime" app.py "$@"
