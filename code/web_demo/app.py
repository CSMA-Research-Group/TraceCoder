"""Flask routes derived from official WebDemo @ 49beb48; see provenance/UPSTREAM.json.
Every run uses tracecoder.service.Workbench, never a global mutable model runner.
"""
import hmac
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from tracecoder.catalog import DATA, benchmark_index, demos, get_task, public_demo
from tracecoder.reports import html_report, review_bundle
from tracecoder.service import Workbench
from tracecoder.storage import validate_task

CODE = Path(__file__).resolve().parents[1]
PAGES = {"/": ("home", "overview"), "/generate": ("generate", "generate"),
         "/history": ("history", "runs"), "/examples": ("examples", "examples"),
         "/settings": ("settings", "settings"), "/help": ("help", "help"),
         "/workbench": ("workbench", "workbench"), "/projects": ("projects", "projects"),
         "/deployment": ("deployment", "deployment")}


def release_info():
    path = CODE / "releases/latest.json"
    if path.exists():
        value = json.loads(path.read_text())
        archive = path.parent / value["filename"]
        if archive.is_file() and archive.parent == path.parent and archive.suffix == ".zip":
            return value, archive
    return {"available": False, "message": "运行 python3 code/scripts/package_release.py 生成完整代码包"}, None


def create_app(workbench=None, token=None, local_only=True):
    app = Flask(__name__, static_folder=str(CODE / "static"), static_url_path="/static")
    app.config.update(MAX_CONTENT_LENGTH=250000)
    app.json.ensure_ascii = False
    # Run exports contain ordered test identities. Do not reorder mappings.
    app.json.sort_keys = False
    service = workbench if workbench is not None else Workbench()
    app.extensions["workbench"] = service
    access = os.environ.get("TRACECODER_ACCESS_TOKEN", "") if token is None else token
    streams = threading.BoundedSemaphore(4)

    @app.before_request
    def authorize():
        if request.path == "/healthz":
            return None
        if local_only and urlparse("http://" + request.host).hostname not in ("localhost", "127.0.0.1", "::1"):
            return jsonify(error="Host rejected"), 403
        if not request.path.startswith("/api/"):
            return None
        if access and not hmac.compare_digest(request.headers.get("X-TraceCoder-Token", "").encode(), access.encode()):
            return jsonify(error="请输入服务器访问口令"), 401
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and urlparse(origin).netloc != request.host:
                return jsonify(error="Cross-origin request rejected"), 403
            if request.mimetype != "application/json":
                return jsonify(error="Expected application/json"), 415
            if not request.content_length or request.content_length > 250000:
                return jsonify(error="Request size invalid"), 413

    @app.after_request
    def headers(response):
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "Content-Security-Policy":
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, FileNotFoundError):
            return jsonify(error=str(exc)), 404
        if isinstance(exc, (ValueError, KeyError, TypeError)):
            return jsonify(error=str(exc)), 400
        app.logger.exception("TraceCoder request failed")
        return jsonify(error="请求处理失败，请检查服务器运行目录与配置。"), 500

    @app.get("/healthz")
    def health():
        return jsonify(ok=True)

    def page():
        template, view = PAGES[request.path]
        return render_template(template + ".html", active_page=view, run_id="")
    for index, path in enumerate(PAGES):
        app.add_url_rule(path, "page_%d" % index, page)

    @app.get("/history/<run_id>")
    def history_detail(run_id):
        if not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{10}", run_id):
            raise FileNotFoundError("无效运行 ID")
        return render_template("history.html", active_page="workbench", run_id=run_id)

    @app.get("/index.html")
    def index():
        return render_template("home.html", active_page="overview", run_id="")

    @app.get("/<asset>")
    def legacy_asset(asset):
        if asset not in {"app.js", "system.js", "portal.js", "style.css", "portal.css", "favicon.svg"}:
            raise FileNotFoundError("Not found")
        return send_file(CODE / "static" / asset)

    def examples():
        return json.loads((DATA / "official/examples.json").read_text())

    def private():
        if service.public_demo:
            raise FileNotFoundError("公开演示不开放项目数据")

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError("Request must be an object")
        return value

    def download(content, mime, filename):
        response = Response(content, mimetype=mime)
        response.headers["Content-Disposition"] = 'attachment; filename="%s"' % filename
        return response

    @app.get("/api/<path:name>")
    def get_api(name):
        if name in {"config", "deployment", "status"}:
            return jsonify(service.config())
        if name == "overview":
            return jsonify(service.overview())
        if name == "models":
            config = service.config()
            return jsonify(models=[config["model"]] if config["model_configured"] else [], configured=config["model_configured"])
        if name == "release":
            return jsonify(release_info()[0])
        if name == "release/download":
            info, archive = release_info()
            if archive is None:
                raise FileNotFoundError("代码包尚未构建")
            return send_file(archive, as_attachment=True, download_name=info["filename"])
        if name == "examples":
            return jsonify(examples())
        if name.startswith("examples/"):
            value = next((v for v in examples() if v["id"] == name.split("/")[1]), None)
            if value is None:
                raise FileNotFoundError("示例不存在")
            return jsonify(value)
        if name == "provenance":
            return jsonify(json.loads((CODE / "provenance/UPSTREAM.json").read_text()))
        if name == "templates":
            return jsonify([json.loads(p.read_text()) for p in sorted((DATA / "templates").glob("*.json"))])
        if name == "projects":
            return jsonify([] if service.public_demo else service.store.projects())
        project = re.fullmatch(r"projects/(p-[a-f0-9]{12})", name)
        task = re.fullmatch(r"project-tasks/(t-[a-f0-9]{12})(/revisions)?", name)
        if project or task:
            private()
            if project:
                return jsonify({**service.store.project(project[1]), "tasks": service.store.tasks(project[1])})
            if task[2]:
                return jsonify(service.store.revisions(task[1]))
            revision = request.args.get("revision")
            return jsonify(service.store.task(task[1], int(revision) if revision else None))
        if name == "runs/search":
            if set(request.args) - {"query", "mode", "status", "project_id", "limit", "offset"}:
                raise ValueError("无效筛选字段")
            return jsonify(service.search_runs(**request.args.to_dict()))
        if name == "tasks":
            return jsonify(demos=[public_demo(t) for t in demos()], benchmarks=benchmark_index())
        if name == "task":
            value = get_task(request.args.get("dataset", "demo"), request.args.get("id", ""))
            return jsonify({k: v for k, v in value.items() if k != "steps"})
        if name == "runs":
            return jsonify(service.history())
        run = re.fullmatch(r"runs/([^/]+)(/export|/bundle|/report)?", name)
        if run:
            value = service.get(run[1])
            if run[2] == "/bundle":
                return download(review_bundle(value), "application/zip", run[1] + "-review.zip")
            if run[2] == "/report":
                return download(html_report(value), "text/html", run[1] + "-report.html")
            if run[2] == "/export":
                return download(json.dumps(value, ensure_ascii=False), "application/json", run[1] + ".json")
            return jsonify(value)
        raise FileNotFoundError("Not found")

    def save_generation(value):
        private()
        if set(value) - {"task", "project_id", "task_id"}:
            raise ValueError("不支持的保存字段")
        task = validate_task(value.get("task"))  # Validate before creating an inbox.
        project = value.get("project_id") or service.store.generation_inbox()["id"]
        return service.store.save_task(project, task, value.get("task_id"))

    def official_run(value):
        allowed = {"problem_description", "test_cases", "title", "entry_point", "source_ref", "initial_code", "start_mode",
                   "max_debug_attempts", "max_no_improvement", "timeout", "enable_instrumentation", "task_id", "revision", "mode"}
        if set(value) - allowed:
            raise ValueError("不支持的字段；模型接口与密钥只在服务器配置")
        options = {"mode": value.get("mode", "live"), "max_rounds": value.get("max_debug_attempts", 5),
                   "patience": value.get("max_no_improvement", 2), "timeout": value.get("timeout", 10),
                   "enable_instrumentation": value.get("enable_instrumentation", True)}
        if options["mode"] == "fixture":
            if set(value) - {"mode", "task_id", "max_debug_attempts", "max_no_improvement", "timeout", "enable_instrumentation"}:
                raise ValueError("预设响应演示只接受原样案例")
            options.update(dataset="demo", task_id=value.get("task_id", "paper-sunday"))
        elif "task_id" in value:
            if set(value) & {"problem_description", "test_cases", "initial_code", "start_mode", "title", "entry_point", "source_ref"}:
                raise ValueError("运行固定版本时不能覆盖输入，请先保存新版本")
            options.update(dataset="project", task_id=value["task_id"], revision=value.get("revision"))
        else:
            # Reject before writing if model is not ready; use /api/generation/tasks to prepare inputs.
            if not service.config()["live_enabled"]:
                raise ValueError("真实模型模式尚未就绪；可先保存任务，稍后配置模型与 Docker")
            from tracecoder.engine import Config
            Config(options["max_rounds"], options["patience"], options["timeout"], options["enable_instrumentation"])
            task = save_generation({"task": {"title": value.get("title", "需求生成任务"),
                "description": value.get("problem_description", ""), "test": value.get("test_cases", ""),
                "initial_code": value.get("initial_code", ""), "start_mode": value.get("start_mode", "generate"),
                "entry_point": value.get("entry_point", ""), "source_ref": value.get("source_ref", "")}})
            options.update(dataset="project", task_id=task["id"], revision=task["revision"])
        return service.create(options, request.remote_addr)

    def stream_run(run):
        released = threading.Event()
        def release():
            if not released.is_set():
                released.set()
                streams.release()
        def event(kind, value):
            return "event: %s\ndata: %s\n\n" % (kind, json.dumps(value, ensure_ascii=False))
        def generate():
            try:
                yield event("run", {"id": run["id"], "status": run["status"], "evidence": run["evidence"]})
                cursor, deadline = 0, time.monotonic() + 25
                while True:
                    current = service.get(run["id"])
                    for step in current["events"][cursor:]:
                        yield event("step", step)
                    cursor = len(current["events"])
                    if current["status"] not in ("running", "queued"):
                        yield event("result", current)
                        break
                    if time.monotonic() > deadline:
                        # Release HTTP thread; job continues. Poll /api/runs/{id}.
                        yield event("resume", {"id": run["id"], "url": "/api/runs/" + run["id"]})
                        break
                    yield ": heartbeat\n\n"
                    time.sleep(.25)
            finally:
                release()
        response = Response(generate(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no"})
        response.call_on_close(release)
        return response

    @app.post("/api/<path:name>")
    def post_api(name):
        value = body()
        if service.public_demo and (name.startswith("projects") or name == "generation/tasks"):
            return jsonify(error="公开演示不开放项目写入"), 403
        if name == "runs":
            return jsonify(service.create(value, request.remote_addr)), 202
        if name == "generation/tasks":
            return jsonify(save_generation(value)), 201
        if name == "projects":
            return jsonify(service.store.create_project(value)), 201
        project = re.fullmatch(r"projects/(p-[a-f0-9]{12})/tasks(?:/(t-[a-f0-9]{12}))?", name)
        if project:
            return jsonify(service.store.save_task(project[1], value, project[2])), 201
        cancel = re.fullmatch(r"runs/([^/]+)/cancel", name)
        if cancel:
            service.get(cancel[1])  # Enforce public/private visibility before cancellation.
            return jsonify(service.cancel(cancel[1]))
        if name == "stop_tracecoder":
            if set(value) != {"run_id"}:
                raise ValueError("必须指定且仅指定 run_id；停止只影响该运行")
            service.get(value["run_id"])
            return jsonify(service.cancel(value["run_id"]))
        if name == "validate_config":
            if value:
                raise ValueError("这里只检查服务器配置，不接受浏览器密钥，也不发起模型请求")
            return jsonify({**service.config(), "model_called": False})
        if name == "run_tracecoder":
            return jsonify(official_run(value)), 202
        if name == "run_tracecoder_stream":
            if not streams.acquire(blocking=False):
                return jsonify(error="流式连接已满，请使用普通提交接口与运行查询"), 429
            try:
                run = official_run(value)
            except BaseException:
                streams.release()
                raise
            return stream_run(run)
        raise FileNotFoundError("Not found")

    return app
