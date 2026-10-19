#!/usr/bin/env python3
"""Verify a distributed source tree's manifest; runtime files may be added later."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, default=Path.cwd(), nargs="?")
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative)
    print(json.dumps({"version": manifest["version"], "verified_files": len(manifest["files"]), "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
