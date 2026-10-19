#!/usr/bin/env python3
"""Create a consistent metadata backup plus atomic run snapshots; no model keys."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecoder.catalog import DATA
from tracecoder.storage import Store, RUN_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=Path(os.environ.get("TRACECODER_STATE_DIR", str(DATA))))
    parser.add_argument("--output", type=Path, required=True, help="A new ZIP path; existing files are never overwritten")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "files": {},
                "note": "运行中的记录为备份时快照；恢复后中断任务不会自动重放。"}
    with tempfile.TemporaryDirectory(prefix="tracecoder-backup-") as directory:
        db = Path(directory) / "system.db"
        Store(args.state_dir / "workspace").backup(db)
        with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
            def add(name, data):
                archive.writestr(name, data)
                manifest["files"][name] = hashlib.sha256(data).hexdigest()
            add("workspace/system.db", db.read_bytes())
            for path in sorted((args.state_dir / "runs").glob("*/run.json")):
                if not RUN_ID.fullmatch(path.parent.name):
                    continue
                raw = path.read_bytes()
                run = json.loads(raw)
                add("runs/%s/run.json" % run["id"], raw)
                if run.get("final_code") is not None:
                    add("runs/%s/final.py" % run["id"], run["final_code"].encode())
            archive.writestr("backup-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"backup": str(args.output), "files": len(manifest["files"]),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
