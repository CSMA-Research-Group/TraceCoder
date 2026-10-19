"""Non-billable deployment inspection. This never calls a model endpoint."""
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time

from . import __version__
from .providers import CompatibleProvider


class Readiness:
    def __init__(self, workspace, public_demo=False):
        self.workspace, self.public_demo = Path(workspace), public_demo
        self.lock = threading.Lock()
        self.cached, self.checked_at = None, 0

    def inspect(self):
        with self.lock:
            if self.cached and time.monotonic() - self.checked_at < 30:
                return self.cached
            configured = bool(os.environ.get("TRACECODER_API_BASE") and os.environ.get("TRACECODER_MODEL"))
            model_error = "尚未配置模型地址与模型名称"
            if configured:
                try:
                    CompatibleProvider()
                    model_error = "配置已读取，尚需真实接口联调"
                except ValueError as exc:
                    configured, model_error = False, str(exc)
            docker = bool(shutil.which("docker"))
            image = os.environ.get("TRACECODER_SANDBOX_IMAGE", "tracecoder-sandbox:1")
            sandbox = False
            sandbox_detail = "未检测到 Docker；演示案例仍可运行"
            if docker and not self.public_demo:
                try:
                    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=6)
                    sandbox = result.returncode == 0
                    sandbox_detail = "镜像与 Docker 服务可用；隔离行为仍需部署验收" if sandbox else "Docker 服务或执行镜像不可用"
                except (OSError, subprocess.TimeoutExpired):
                    sandbox_detail = "Docker 检查失败或超时"
            elif self.public_demo:
                sandbox_detail = "公开演示配置关闭自定义代码执行"
            token = bool(os.environ.get("TRACECODER_ACCESS_TOKEN"))
            self.cached = {"version": __version__, "python": platform.python_version(),
                "profile": "public-demo" if self.public_demo else "workspace",
                "model_configured": configured, "docker_installed": docker, "sandbox_ready": sandbox,
                "live_enabled": configured and sandbox and not self.public_demo,
                "public_demo": self.public_demo, "projects_enabled": not self.public_demo,
                "model": os.environ.get("TRACECODER_MODEL", "未配置"), "access_protected": token,
                "max_rounds": 5, "checks": [
                    {"key": "service", "title": "应用运行环境", "status": "ready", "detail": "Python %s · v%s · 检测程序可正常运行" % (platform.python_version(), __version__)},
                    {"key": "storage", "title": "持久化存储", "status": "ready" if os.access(self.workspace, os.W_OK) else "missing", "detail": "SQLite 项目索引 + 逐次运行证据"},
                    {"key": "model", "title": "模型接口", "status": "configured" if configured else "missing", "detail": model_error},
                    {"key": "sandbox", "title": "隔离执行环境", "status": "configured" if sandbox else "missing", "detail": sandbox_detail},
                    {"key": "access", "title": "实例访问保护", "status": "configured" if token else "local", "detail": "已启用服务器访问口令" if token else "本机工作台 / 固定公开演示；团队部署需设置访问口令"}],
                "evidence_note": "就绪检查不调用模型，不代表远程部署或隔离验收已经完成。"}
            self.checked_at = time.monotonic()
            return self.cached
