#!/usr/bin/env python3
"""Submit a frozen task; return review artifacts, never apply patches to a repo.

Exit codes: 0 = all fixed tests passed (or prepare-only completed),
1 = incomplete/failed repair, 2 = client configuration, 3 = HTTP/transport failure.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--project", required=True, help="Existing p-... project ID")
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("tracecoder-artifacts"))
    parser.add_argument("--prepare-only", action="store_true", help="Save immutable inputs without any model/code execution")
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    base = args.server.rstrip("/")
    url = urllib.parse.urlparse(base)
    if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error("server 必须为不含凭据的 HTTP(S) URL")
    if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
        parser.error("远程服务必须通过 HTTPS 连接")
    if not args.project.startswith("p-") or any(c not in "p-0123456789abcdef" for c in args.project):
        parser.error("无效项目 ID")
    if args.wait_seconds < 1 or not 1 <= args.rounds <= 5:
        parser.error("等待时间必须为正数，修复轮数须为 1–5")
    try:
        if args.task_file.stat().st_size > 250000:
            parser.error("任务文件不能超过 250 KB")
        task = json.loads(args.task_file.read_text())
    except (OSError, ValueError) as exc:
        parser.error("无法读取任务 JSON: " + str(exc))
    token = os.environ.get("TRACECODER_ACCESS_TOKEN", "")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def request(path, body=None, binary=False):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-TraceCoder-Token"] = token
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        with opener.open(req, timeout=30) as response:
            data = response.read(40_000_001)
        if len(data) > 40_000_000:
            raise ValueError("服务响应超过 40 MB 客户端限制")
        return data if binary else json.loads(data)

    try:
        saved = request("/api/projects/%s/tasks" % args.project, task)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / (saved["id"] + "-input.json")).write_text(json.dumps(saved, ensure_ascii=False, indent=2))
        if args.prepare_only:
            print(json.dumps({"prepared": True, "task_id": saved["id"], "revision": saved["revision"],
                              "model_called": False, "code_executed": False}, ensure_ascii=False))
            return 0
        run = request("/api/runs", {"mode": "live", "dataset": "project", "task_id": saved["id"],
                                    "revision": saved["revision"], "max_rounds": args.rounds})
        run_id = run["id"]
        deadline = time.monotonic() + args.wait_seconds
        while run["status"] in ("queued", "running"):
            if time.monotonic() >= deadline:
                request("/api/runs/%s/cancel" % run_id, {})
                print("CI 等待超时，已请求取消；可从服务端运行中心继续读取证据。", file=sys.stderr)
                return 1
            time.sleep(.5)
            run = request("/api/runs/" + run_id)
        (args.output / (run_id + ".json")).write_text(json.dumps(run, ensure_ascii=False, indent=2))
        (args.output / (run_id + "-review.zip")).write_bytes(request("/api/runs/%s/bundle" % run_id, binary=True))
        print(json.dumps({"id": run_id, "status": run["status"], "stop_reason": run["stop_reason"],
                          "passed": bool((run.get("final_eval") or {}).get("passed")), "artifact_dir": str(args.output)}, ensure_ascii=False))
        return 0 if run["status"] == "completed" and (run.get("final_eval") or {}).get("passed") else 1
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read(10000)).get("error", "请求失败")
        except (ValueError, AttributeError):
            detail = "服务未返回有效错误信息"
        print("HTTP %s: %s" % (exc.code, detail), file=sys.stderr)
        return 3
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        print("CI 客户端失败: %s" % exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
