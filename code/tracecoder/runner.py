import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

WORKER = Path(__file__).with_name("worker.py")


class Runner:
    def __init__(self, backend="local", timeout=10, image="tracecoder-sandbox:1"):
        if backend not in ("local", "docker"):
            raise ValueError("Unknown execution backend")
        self.backend, self.timeout, self.image = backend, timeout, image

    def evaluate(self, code, task):
        payload = {"code": code, "test": task["test"], "entry_point": task.get("entry_point", ""), "timeout": self.timeout}
        name = "tracecoder-" + uuid.uuid4().hex
        if self.backend == "docker":
            if not shutil.which("docker"):
                raise RuntimeError("未安装 Docker。真实模型代码需要独立执行容器；请参阅部署说明。")
            command = ["docker", "run", "--rm", "--pull=never", "-i", "--name", name,
                       "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--pids-limit=64", "--memory=512m", "--memory-swap=512m", "--cpus=1",
                       "--user=65534:65534", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m", self.image]
        else:
            command = [sys.executable, "-I", str(WORKER)]
        clean_env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "LANG", "LC_ALL", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")}
        clean_env.update({"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        with tempfile.TemporaryDirectory(prefix="tracecoder-eval-") as directory:
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=errors,
                                           cwd=directory, env=clean_env, start_new_session=True)
                try:
                    process.communicate(json.dumps(payload).encode(), timeout=self.timeout + (5 if self.backend == "docker" else 1))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    if self.backend == "docker":
                        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
                    return self.failure("执行超时（%s 秒）" % self.timeout, "timeout")
                output.seek(0)
                errors.seek(0)
                raw = output.read(5_000_001)
                if len(raw) > 5_000_000:
                    return self.failure("执行输出超过上限", "output_limit")
                try:
                    result = json.loads(raw)
                    if not isinstance(result, dict) or "cases" not in result:
                        raise ValueError("Invalid evaluator response")
                except (ValueError, UnicodeError):
                    return self.failure(errors.read(6000).decode(errors="replace") or "执行器未返回有效 JSON", "worker_error")
                result.update({"exit_code": process.returncode, "backend": self.backend})
                if process.returncode not in (0, 1):
                    result.update({"passed": False, "valid": False})
                return result

    def failure(self, message, kind):
        return {"passed": False, "passed_count": 0, "total_count": 0, "cases": [],
                "valid": False, "error": message, "kind": kind, "exit_code": None, "backend": self.backend}


def feedback(result):
    """Bounded diagnostic view; full per-test evidence remains in run artifacts."""
    failures = [case for case in result["cases"] if case["status"] != "passed"]
    successes = [case for case in result["cases"] if case["status"] == "passed"]
    selected = failures[:8] + successes[:2]
    return {"passed_count": result["passed_count"], "total_count": result["total_count"],
            "error": result.get("error"), "valid": result["valid"],
            "cases": [{**case, "stdout": "\n".join(case["stdout"].splitlines()[-50:])} for case in selected],
            "omitted_cases": len(result["cases"]) - len(selected)}
