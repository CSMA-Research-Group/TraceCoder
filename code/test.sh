#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${TRACECODER_PYTHON:-.venv/bin/python}" -m unittest discover -s tests -v
