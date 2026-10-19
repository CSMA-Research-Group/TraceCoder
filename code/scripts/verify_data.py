#!/usr/bin/env python3
"""Check provenance and run the original HumanEval reference implementations.

These are evaluator compatibility checks, NOT generated-code repair experiments.
"""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracecoder.catalog import DATA, ROOT
from tracecoder.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-originals", action="store_true", help="Require every original source/PDF in the research checkout")
    parser.add_argument("--skip-evaluator", action="store_true", help="Verify dataset integrity only")
    args = parser.parse_args()
    started = time.monotonic()
    original_count = 0
    missing = []
    sources = json.loads((DATA / "source_manifest.json").read_text())
    for source in sources:
        original = ROOT / source["source"]
        if original.exists():
            assert hashlib.sha256(original.read_bytes()).hexdigest() == source["source_sha256"]
        elif args.require_originals:
            raise FileNotFoundError(original)
        assert hashlib.sha256((ROOT / source["output"]).read_bytes()).hexdigest() == source["output_sha256"]
        rows = [json.loads(line) for line in (ROOT / source["output"]).read_text().splitlines()]
        assert len(rows) == source["rows"]
        assert len({row["task_id"] for row in rows}) == len(rows)
        for row in rows:
            ast.parse(row["test"])
    preservation = json.loads((DATA / "original_materials.sha256.json").read_text())
    for path, sha in preservation.items():
        original = ROOT / path
        if original.exists():
            assert hashlib.sha256(original.read_bytes()).hexdigest() == sha
            original_count += 1
        else:
            missing.append(path)
    if args.require_originals and missing:
        raise FileNotFoundError("Missing original materials: " + str(missing))
    results = []
    runner = Runner(timeout=10)
    for line in ([] if args.skip_evaluator else (DATA / "benchmarks/humaneval.jsonl").read_text().splitlines()):
        row = json.loads(line)
        result = runner.evaluate(row["prompt"] + row["canonical_solution"], row)
        results.append({"task_id": row["task_id"], "passed": result["passed"], "valid": result["valid"],
                        "passed_count": result["passed_count"], "total_count": result["total_count"],
                        "error": result.get("error"), "failing_cases": [c for c in result["cases"] if c["status"] != "passed"]})
    report = {"scope": "Supplied reference implementations used only to validate evaluator compatibility; no LLM calls, no repair benchmark result",
              "sources_verified": len(sources), "original_files_unchanged": original_count, "original_files_not_in_this_package": len(missing),
              "reference_programs_passed": sum(r["passed"] for r in results), "reference_programs_total": len(results),
              "elapsed_seconds": round(time.monotonic() - started, 3), "results": results}
    destination = DATA / "verification/integration-v2.1/data_and_evaluator.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
