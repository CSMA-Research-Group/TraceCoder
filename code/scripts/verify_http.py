#!/usr/bin/env python3
"""Exercise the real HTTP service, without model calls or custom code execution."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data/verification/integration-v2.1/http-fixtures.json")
    args = parser.parse_args()
    checks = []

    def request(path, data=None, expected=200, headers=None):
        header = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(args.url + path, data=None if data is None else json.dumps(data).encode(), headers=header)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                status, body, content_type = response.status, response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            status, body, content_type = exc.code, exc.read(), exc.headers.get("Content-Type", "")
        assert status == expected, (path, status, body[:400])
        checks.append({"path": path, "status": status, "expected": expected})
        return json.loads(body) if "json" in content_type else body.decode()

    assert "TraceCoder" in request("/")
    assert request("/healthz")["ok"]
    catalog = request("/api/tasks")
    assert len(catalog["demos"]) == 4
    assert {k: len(v) for k, v in catalog["benchmarks"].items()} == {"humaneval": 164, "humanevalplus": 164, "classeval": 100, "bigcodebench": 1140}
    request("/api/runs", {"mode": "unknown"}, 400)
    request("/api/runs", {"mode": "fixture", "initial_code": "print('untrusted')"}, 400)
    request("/api/runs", {"max_rounds": 6}, 400)
    request("/api/runs", {"mode": "fixture"}, 403, {"Origin": "https://another.invalid"})
    request("/api/config", expected=403, headers={"Host": "unexpected.invalid"})
    request("/.env", expected=404)
    request("/api/runs/not-a-run", expected=400)
    summaries = []
    for task in catalog["demos"]:
        run = request("/api/runs", {"mode": "fixture", "dataset": "demo", "task_id": task["id"]}, 202)
        deadline = time.monotonic() + 20
        while run["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(.1)
            run = request("/api/runs/" + run["id"])
        assert run["status"] == "completed", run.get("error")
        assert run["final_eval"]["passed"]
        export = request("/api/runs/" + run["id"] + "/export")
        assert export["hashes"] == run["hashes"]
        assert export["final_code"] == run["final_code"]
        summaries.append({"task": task["id"], "run_id": run["id"], "initial": run["initial_eval"]["passed_count"], "final": run["final_eval"]["passed_count"], "total": run["final_eval"]["total_count"], "rounds": len(run["rounds"])})
    history = request("/api/runs")
    assert all(any(entry["id"] == run["run_id"] for entry in history) for run in summaries)
    result = {"all_checks_passed": True, "scope": "actual local HTTP + fixture agents + actual Python execution; no model API call", "checks": checks, "runs": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"http_checks": len(checks), "runs": summaries, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
