#!/usr/bin/env python3
"""Isolated end-to-end HTTP, access, persistence and CI prepare-only acceptance."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/verification/integration-v2.1/http-system.json")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Test an extracted standalone package instead of this checkout")
    args = parser.parse_args()
    root = args.source_root.resolve()
    checks, runs = [], []
    token = "tracecoder-isolated-acceptance-only"
    with tempfile.TemporaryDirectory(prefix="tracecoder-system-http-") as temporary:
        state_dir = Path(temporary) / "state"
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRACECODER_")}
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "TRACECODER_STATE_DIR": str(state_dir), "TRACECODER_ACCESS_TOKEN": token})
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = "http://127.0.0.1:%d" % port

        def request(path, value=None, status=200, protected=True, headers=None):
            h = {"Content-Type": "application/json"}
            if protected:
                h["X-TraceCoder-Token"] = token
            h.update(headers or {})
            req = urllib.request.Request(base + path, data=None if value is None else json.dumps(value).encode(), headers=h)
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    actual, data, content_type = r.status, r.read(), r.headers["Content-Type"]
            except urllib.error.HTTPError as exc:
                actual, data, content_type = exc.code, exc.read(), exc.headers["Content-Type"]
            assert actual == status, (path, actual, data[:400])
            checks.append({"path": path, "expected": status, "actual": actual})
            return json.loads(data) if "application/json" in content_type else data

        @contextmanager
        def server(public=False):
            command = [sys.executable, "-B", str(root / "code/app.py"), "--port", str(port)]
            if public:
                command.append("--public-demo")
            proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise AssertionError(proc.communicate()[1].decode())
                    try:
                        request("/healthz", protected=False)
                        break
                    except urllib.error.URLError:
                        time.sleep(.05)
                else:
                    raise AssertionError("HTTP service never became healthy")
                yield proc
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    proc.communicate(timeout=15)

        with server():
            assert b"TraceCoder" in request("/", protected=False)
            request("/api/projects", protected=False, status=401)
            request("/api/config", headers={"Host": "attacker.invalid"}, status=403)
            request("/api/projects", {"name": "blocked"}, headers={"Origin": "https://attacker.invalid"}, status=403)
            request("/api/projects", {"name": "blocked"}, headers={"Content-Type": "text/plain"}, status=415)
            request("/.env", status=404)
            request("/api/runs/not-a-run", status=400)
            config = request("/api/config")
            assert not config["live_enabled"] and config["access_protected"]
            project = request("/api/projects", {"name": "HTTP acceptance", "source_ref": "test://local"}, status=201)
            template = request("/api/templates")[0]
            task = request("/api/projects/%s/tasks" % project["id"], template, status=201)
            changed = {**template, "initial_code": template["initial_code"].replace("start <=", "start <")}
            revision = request("/api/projects/%s/tasks/%s" % (project["id"], task["id"]), changed, status=201)
            assert revision["revision"] == 2
            first = request("/api/project-tasks/%s?revision=1" % task["id"])
            assert first["initial_code"] == template["initial_code"]
            assert first["revision_sha256"] != revision["revision_sha256"]
            assert len(request("/api/project-tasks/%s/revisions" % task["id"])) == 2
            request("/api/projects/%s/tasks" % project["id"], {**template, "test": ""}, status=400)
            request("/api/runs", {"mode": "live", "dataset": "project", "task_id": task["id"], "revision": 1}, status=400)
            request("/api/runs", {"mode": "fixture", "dataset": "project", "task_id": task["id"], "revision": 1}, status=400)
            request("/api/runs", {"mode": "fixture", "initial_code": "print(1)"}, status=400)
            request("/api/runs", {"max_rounds": 2.5}, status=400)
            cli = subprocess.run([sys.executable, "-B", str(root / "code/integrations/ci_submit.py"),
                                  "--server", base, "--project", project["id"], "--task-file", str(root / "data/templates/api-window.task.json"),
                                  "--prepare-only", "--output", str(Path(temporary) / "ci")], env=env, capture_output=True, timeout=15)
            assert cli.returncode == 0, cli.stderr.decode()
            assert json.loads(cli.stdout)["model_called"] is False
            for demo in request("/api/tasks")["demos"]:
                value = request("/api/runs", {"mode": "fixture", "task_id": demo["id"]}, status=202)
                deadline = time.monotonic() + 30
                while value["status"] in ("queued", "running") and time.monotonic() < deadline:
                    time.sleep(.1)
                    value = request("/api/runs/" + value["id"])
                assert value["status"] == "completed" and value["final_eval"]["passed"], value.get("error")
                assert value == request("/api/runs/%s/export" % value["id"])
                bundle = request("/api/runs/%s/bundle" % value["id"])
                with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                    assert json.loads(archive.read("run.json")) == value
                    for line in archive.read("SHA256SUMS").decode().splitlines():
                        sha, path = line.split("  ", 1)
                        assert sha == hashlib.sha256(archive.read(path)).hexdigest()
                assert b"TraceCoder" in request("/api/runs/%s/report" % value["id"])
                evidence_dir = args.output.parent / "snapshots"
                evidence_dir.mkdir(parents=True, exist_ok=True)
                (evidence_dir / (demo["id"] + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2))
                runs.append({"id": value["id"], "task": demo["id"], "initial": value["initial_eval"]["passed_count"],
                             "final": value["final_eval"]["passed_count"], "total": value["final_eval"]["total_count"], "rounds": len(value["rounds"])})
            assert request("/api/overview")["runs"] == 4
            assert request("/api/runs/search?mode=fixture&limit=2")["total"] == 4
            assert len(request("/api/runs/search?mode=fixture&limit=2&offset=2")["items"]) == 2
            assert request("/api/runs/search?mode=live")["total"] == 0
            request("/api/runs/search?status=invalid", status=400)

        with server():
            assert request("/api/projects")[0]["id"] == project["id"]
            assert request("/api/runs/" + runs[0]["id"])["status"] == "completed"
            assert request("/api/project-tasks/%s?revision=1" % task["id"])["initial_code"] == template["initial_code"]
        # Add a labelled synthetic private record to test public-mode disclosure boundaries.
        private = json.loads((state_dir / "runs" / runs[0]["id"] / "run.json").read_text())
        private["id"], private["mode"] = "20261018T000000Z-aaaaaaaaaa", "live"
        path = state_dir / "runs" / private["id"]
        path.mkdir()
        (path / "run.json").write_text(json.dumps(private))
        with server(public=True):
            assert request("/api/projects") == []
            request("/api/projects", {"name": "blocked"}, status=403)
            request("/api/projects/" + project["id"], status=404)
            request("/api/project-tasks/" + task["id"], status=404)
            request("/api/runs/" + private["id"], status=404)
            assert request("/api/runs/search?mode=live")["items"] == []
            assert request("/api/overview")["runs"] == 4
            assert request("/api/overview")["projects"] == 0
            assert len(request("/api/runs")) == 4
            assert not request("/api/config")["projects_enabled"]

    report = {"passed": True, "scope": "isolated HTTP + real fixture execution + CI prepare-only; no model request or Docker execution",
              "checks": checks, "runs": runs, "restart_persistence": True, "public_private_isolation": True, "ci_prepare_only": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"passed": True, "http_checks": len(checks), "runs": runs, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
