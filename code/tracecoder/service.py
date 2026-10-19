#!/usr/bin/env python3
"""Shared job service for the official Flask application, API and durable workspace."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

from tracecoder import __version__
from tracecoder.catalog import DATA, benchmark_index, demos, get_task, public_demo
from tracecoder.storage import Store, RUN_ID
from tracecoder.readiness import Readiness
from tracecoder.engine import Config, Run
from tracecoder.providers import CompatibleProvider, FixtureProvider
from tracecoder.runner import Runner

ID = RUN_ID


class Workbench:
    def __init__(self, public_demo=False, state_dir=None, workers=None, queue_limit=None):
        self.public_demo = public_demo
        self.state_dir = Path(state_dir or os.environ.get("TRACECODER_STATE_DIR", str(DATA)))
        self.run_dir = self.state_dir / "runs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.state_dir / "workspace")
        self.readiness = Readiness(self.store.directory, public_demo)
        self.workers = workers or int(os.environ.get("TRACECODER_WORKERS", "2"))
        self.queue_limit = int(os.environ.get("TRACECODER_QUEUE_LIMIT", "6")) if queue_limit is None else queue_limit
        if not 1 <= self.workers <= 8 or not 0 <= self.queue_limit <= 32:
            raise ValueError("并发数量须为 1–8，排队上限须为 0–32")
        self.runs, self.futures, self.rate = {}, {}, {}
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="tracecoder")
        self.lock = threading.RLock()
        self.recover()

    def recover(self):
        for path in self.run_dir.glob("*/run.json"):
            if not ID.fullmatch(path.parent.name):
                continue
            try:
                value = json.loads(path.read_text())
                if value["status"] in ("queued", "running"):
                    value.update({"status": "error", "stop_reason": "server_restarted", "stage": "finished",
                                  "error": "服务已重启；保留中断证据。请人工重新提交，不会自动重复付费调用。"})
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
                    temporary.replace(path)
                self.store.index_run(value)
            except (ValueError, KeyError, OSError):
                continue

    def config(self):
        return {**self.readiness.inspect(), "workers": self.workers, "queue_limit": self.queue_limit}

    def create(self, request, address):
        allowed = {"mode", "dataset", "task_id", "revision", "max_rounds", "patience", "timeout", "initial_code", "start_mode", "enable_instrumentation"}
        if set(request) - allowed:
            raise ValueError("不支持的请求字段")
        mode = request.get("mode", "fixture")
        if mode not in ("fixture", "live"):
            raise ValueError("Unknown mode")
        for key, default in (("max_rounds", 5), ("patience", 2), ("timeout", 10)):
            if type(request.get(key, default)) is not int:
                raise ValueError(key + " 必须是整数")
        config = Config(request.get("max_rounds", 5), request.get("patience", 2), request.get("timeout", 10), request.get("enable_instrumentation", True))
        dataset = request.get("dataset", "demo")
        if dataset == "project":
            if self.public_demo or mode != "live":
                raise ValueError("项目代码只能在已配置的私有工作台通过隔离容器运行")
            revision = request.get("revision")
            if type(revision) is not int or revision < 1:
                raise ValueError("项目运行必须指定固定 revision")
            task = self.store.task(request.get("task_id"), revision)
            if "initial_code" in request or request.get("start_mode", task.get("start_mode", "frozen")) != task.get("start_mode", "frozen"):
                raise ValueError("项目运行使用保存版本的代码；修改请先保存为新版本")
            task = self.store.task(request.get("task_id"), revision)
        else:
            task = get_task(dataset, request.get("task_id", "paper-sunday"))
        if mode == "fixture":
            if dataset != "demo" or request.get("initial_code") is not None or request.get("start_mode", "frozen") != "frozen":
                raise ValueError("预设响应演示仅接受原样案例；修改代码请使用真实模型模式。")
            provider, initial = FixtureProvider(task), task["initial_code"]
            runner = Runner("local", config.timeout)
        else:
            if not self.config()["live_enabled"]:
                raise ValueError("真实模型模式尚未就绪，请查看部署中心的模型与执行环境检查。")
            provider = CompatibleProvider()
            runner = Runner("docker", config.timeout, os.environ.get("TRACECODER_SANDBOX_IMAGE", "tracecoder-sandbox:1"))
            start = request.get("start_mode", task.get("start_mode", "frozen") if dataset == "project" else "generate")
            if start not in ("generate", "frozen"):
                raise ValueError("Unknown start mode")
            initial = (task["initial_code"] if dataset == "project" else request.get("initial_code")) if start == "frozen" else None
            if start == "frozen" and (not isinstance(initial, str) or not initial.strip() or len(initial) > 60000):
                raise ValueError("请填写有效的初始代码，最多 60000 字符")
        with self.lock:
            now = time.monotonic()
            self.rate = {ip: [stamp for stamp in stamps if now - stamp < 60] for ip, stamps in self.rate.items() if stamps and now - stamps[-1] < 60}
            recent = self.rate.get(address, [])
            if len(recent) >= 12:
                raise ValueError("请求过于频繁，请稍后重试。")
            if len(self.runs) >= self.workers + self.queue_limit:
                raise ValueError("运行队列已满，请稍后提交。")
            run = Run(task, provider, runner, config, initial, self.run_dir, self.store.index_run)
            run_id = run.state["id"]
            self.runs[run_id] = run
            self.rate[address] = recent + [now]
            self.futures[run_id] = self.executor.submit(self._work, run)
            return run.snapshot()

    def _work(self, run):
        try:
            run.execute()
        finally:
            with self.lock:
                self.runs.pop(run.state["id"], None)
                self.futures.pop(run.state["id"], None)

    def cancel(self, run_id):
        with self.lock:
            run = self.runs.get(run_id)
            if not run:
                self.get(run_id)
                return {"requested": False, "message": "运行已结束"}
            run.cancelled.set()
            future = self.futures.get(run_id)
            if future and future.cancel():
                run.finish("cancelled", time.monotonic(), "用户在排队阶段取消")
                self.runs.pop(run_id, None)
                self.futures.pop(run_id, None)
            return {"requested": True, "message": "已停止排队，或将在当前阶段结束后停止"}

    def get(self, run_id):
        if not isinstance(run_id, str) or not ID.fullmatch(run_id):
            raise ValueError("Invalid run ID")
        with self.lock:
            run = self.runs.get(run_id)
        if run:
            value = run.snapshot()
        else:
            path = self.run_dir / run_id / "run.json"
            if not path.exists():
                raise FileNotFoundError("运行记录不存在")
            value = json.loads(path.read_text())
        if self.public_demo and value.get("mode") != "fixture":
            raise FileNotFoundError("公开演示不开放私有运行记录")
        return value

    def history(self):
        return self.search_runs(limit=50)["items"]

    def search_runs(self, **query):
        if self.public_demo:
            if query.get("mode") == "live":
                return {"items": [], "total": 0, "limit": 20, "offset": 0}
            query["mode"] = "fixture"
        return self.store.search_runs(**query)

    def overview(self):
        value = self.store.overview()
        if self.public_demo:
            total = self.store.search_runs(mode="fixture", limit=6)
            value.update({"runs": total["total"], "projects": 0, "tasks": 0,
                          "by_mode": {"fixture": total["total"]}, "by_status": {}, "recent": total["items"]})
        with self.lock:
            value["capacity"] = {"workers": self.workers, "queue_limit": self.queue_limit,
                                 "active": sum(run.state["status"] == "running" for run in self.runs.values()),
                                 "queued": sum(run.state["status"] == "queued" for run in self.runs.values())}
        return value

    def close(self):
        with self.lock:
            for run in self.runs.values():
                run.cancelled.set()
        self.executor.shutdown(wait=True)

