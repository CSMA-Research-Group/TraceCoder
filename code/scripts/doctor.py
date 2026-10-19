#!/usr/bin/env python3
"""Inspect deployment prerequisites without sending a model request."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecoder.catalog import DATA
from tracecoder.readiness import Readiness
from tracecoder.storage import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--public-demo", action="store_true")
    args = parser.parse_args()
    directory = Path(os.environ.get("TRACECODER_STATE_DIR", str(DATA))) / "workspace"
    Store(directory)
    value = Readiness(directory, args.public_demo).inspect()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0 if not args.require_live or value["live_enabled"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
