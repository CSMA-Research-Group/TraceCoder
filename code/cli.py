#!/usr/bin/env python3
"""CLI entrypoint, using exactly the same engine and artifacts as the web app."""
import argparse
import json
import os
from pathlib import Path

from tracecoder.catalog import DATA, get_task
from tracecoder.engine import Config, Run
from tracecoder.providers import CompatibleProvider, FixtureProvider
from tracecoder.runner import Runner
from tracecoder.readiness import Readiness
from tracecoder.storage import Store


def main():
    parser = argparse.ArgumentParser(description="TraceCoder paper-method reproduction")
    parser.add_argument("mode", choices=["fixture", "live"])
    parser.add_argument("--dataset", default="demo", choices=["demo", "humaneval", "humanevalplus", "classeval", "bigcodebench"])
    parser.add_argument("--task", default="paper-sunday")
    parser.add_argument("--initial-code", type=Path, help="Frozen failing source; omit to generate initial code in live mode")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--no-instrumentation", action="store_true", help="Explicit ablation: disable diagnostic probes")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    state_dir = Path(os.environ.get("TRACECODER_STATE_DIR", str(DATA)))
    store = Store(state_dir / "workspace") if args.output_dir is None else None
    task = get_task(args.dataset, args.task)
    if args.mode == "fixture":
        if args.dataset != "demo" or args.initial_code:
            parser.error("Fixture mode only runs unchanged shipped examples")
        provider, code = FixtureProvider(task), task["initial_code"]
    else:
        if not Readiness(state_dir / "workspace").inspect()["live_enabled"]:
            parser.error("真实模式需配置模型接口并准备可用的 Docker 执行镜像；请先运行 doctor.py --require-live")
        provider = CompatibleProvider()
        code = args.initial_code.read_text() if args.initial_code else None
    config = Config(args.rounds, args.patience, args.timeout, not args.no_instrumentation)
    run = Run(task, provider, Runner("local" if args.mode == "fixture" else "docker", args.timeout,
              os.environ.get("TRACECODER_SANDBOX_IMAGE", "tracecoder-sandbox:1")), config, code,
              args.output_dir or state_dir / "runs", store.index_run if store else None)
    result = run.execute()
    evaluation = result.get("final_eval") or {}
    print(json.dumps({"id": result["id"], "status": result["status"], "evidence": result["evidence"],
        "stop_reason": result["stop_reason"], "passed": evaluation.get("passed_count"), "total": evaluation.get("total_count"),
        "rounds": len(result["rounds"]), "error": result.get("error"), "artifact": str(run.directory / "run.json")}, ensure_ascii=False, indent=2))
    return 0 if evaluation.get("passed") and result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
