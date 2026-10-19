"""SQLite workspace metadata. Task revisions are immutable; run JSON is evidence."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import uuid

from .catalog import digest

RUN_ID = re.compile(r"^\d{8}T\d{6}Z-[a-f0-9]{10}$")
OBJECT_ID = re.compile(r"^[pt]-[a-f0-9]{12}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def string(value, label, maximum, required=True):
    if not isinstance(value, str) or len(value) > maximum or (required and not value.strip()):
        raise ValueError("%s 必须是%s不超过 %d 字符的文本" % (label, "非空、" if required else "", maximum))
    return value


def validate_task(value):
    if not isinstance(value, dict):
        raise ValueError("任务必须是 JSON 对象")
    fields = {"title", "description", "initial_code", "test", "entry_point", "source_ref", "start_mode"}
    if set(value) - fields:
        raise ValueError("任务包含未定义字段")
    start = value.get("start_mode", "frozen")
    if start not in ("generate", "frozen"):
        raise ValueError("start_mode 必须为 generate 或 frozen")
    optional = {"entry_point", "source_ref"} | ({"initial_code"} if start == "generate" else set())
    result = {key: string(value.get(key, ""), key, limit, key not in optional)
              for key, limit in {"title": 120, "description": 16000, "initial_code": 60000,
                                 "test": 60000, "entry_point": 120, "source_ref": 300}.items()}
    if start == "generate" and result["initial_code"].strip():
        raise ValueError("生成任务不接受初始代码；已有失败代码请选择 frozen")
    result["start_mode"] = start
    try:
        ast.parse(result["initial_code"])
        tests = ast.parse(result["test"])
    except SyntaxError as exc:
        raise ValueError("代码或测试存在语法错误：第 %s 行 %s" % (exc.lineno, exc.msg)) from None
    checks = [node for node in tests.body if isinstance(node, ast.FunctionDef) and node.name == "check"]
    methods = [node for cls in tests.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test")]
    # Module-level assertions (including loops) are the official WebDemo format.
    class Assertions(ast.NodeVisitor):
        found = False
        def visit_Assert(self, node):
            self.found = True
        def visit_FunctionDef(self, node):
            pass
        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
    assertions = Assertions()
    assertions.visit(tests)
    if sum((bool(checks), bool(methods), assertions.found)) > 1:
        raise ValueError("一个任务请选择一种测试入口；不要混用顶层 assert、unittest 和 check(candidate)")
    if not checks and not methods and not assertions.found:
        raise ValueError("请提供顶层 assert、unittest 测试类或 check(candidate)；当前不支持 pytest 及多文件测试工程")
    if checks:
        if not result["entry_point"].isidentifier():
            raise ValueError("check(candidate) 任务需要有效的 entry_point 函数名")
        if not any(isinstance(node, ast.Assert) or
                   (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "assertion")
                   for node in ast.walk(checks[0])):
            raise ValueError("check(candidate) 至少需要一个断言")
    elif result["entry_point"] and not result["entry_point"].isidentifier():
        raise ValueError("entry_point 必须为有效的 Python 标识符")
    return result


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "system.db"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
                    source_ref TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    task_id TEXT NOT NULL, project_id TEXT NOT NULL REFERENCES projects(id),
                    revision INTEGER NOT NULL, title TEXT NOT NULL, payload TEXT NOT NULL,
                    sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, revision)
                );
                CREATE INDEX IF NOT EXISTS project_revisions ON revisions(project_id, task_id, revision);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, project_id TEXT NOT NULL,
                    task_id TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, summary TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS run_created ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS run_filter ON runs(mode, status, project_id);
                PRAGMA user_version=1;
            """)

    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create_project(self, value):
        if set(value) - {"name", "description", "source_ref"}:
            raise ValueError("项目包含未定义字段")
        project = {"id": "p-" + uuid.uuid4().hex[:12],
                   "name": string(value.get("name"), "项目名称", 80).strip(),
                   "description": string(value.get("description", ""), "项目说明", 2000, False),
                   "source_ref": string(value.get("source_ref", ""), "来源标识", 300, False), "created_at": now()}
        with self.connect() as db:
            db.execute("INSERT INTO projects VALUES (:id,:name,:description,:source_ref,:created_at)", project)
        return project

    def projects(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("""SELECT p.*, COUNT(DISTINCT r.task_id) task_count
                FROM projects p LEFT JOIN revisions r ON p.id=r.project_id
                GROUP BY p.id ORDER BY p.created_at DESC""")]

    def generation_inbox(self):
        """Stable workspace for the official requirement/test generation entry."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM projects WHERE source_ref=? LIMIT 1", ("tracecoder://generation-inbox",)).fetchone()
            if row:
                return dict(row)
            value = {"id": "p-" + uuid.uuid4().hex[:12], "name": "生成与修复任务",
                     "description": "从需求与固定测试创建的任务；所有输入按版本保存。",
                     "source_ref": "tracecoder://generation-inbox", "created_at": now()}
            db.execute("INSERT INTO projects VALUES (:id,:name,:description,:source_ref,:created_at)", value)
            return value

    def project(self, project_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            raise FileNotFoundError("项目不存在")
        return dict(row)

    def save_task(self, project_id, value, task_id=None):
        task = validate_task(value)
        project = self.project(project_id)
        with self.connect() as db:
            # Serialize revision assignment, including concurrent API requests.
            db.execute("BEGIN IMMEDIATE")
            if task_id:
                row = db.execute("SELECT MAX(revision) n FROM revisions WHERE task_id=? AND project_id=?", (task_id, project_id)).fetchone()
                if row["n"] is None:
                    raise FileNotFoundError("任务不存在于此项目")
                revision = row["n"] + 1
            else:
                task_id, revision = "t-" + uuid.uuid4().hex[:12], 1
            sha = digest(json.dumps(task, sort_keys=True, ensure_ascii=False))
            task.update({"id": task_id, "dataset": "project", "project_id": project_id,
                         "project_name": project["name"], "revision": revision, "revision_sha256": sha,
                         "created_at": now(), "provenance": "已冻结需求、测试与起始方式；保存只做静态校验，未执行代码。",
                         "tags": ["项目任务", "固定测试", "v%d" % revision]})
            db.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?)",
                       (task_id, project_id, revision, task["title"], json.dumps(task, ensure_ascii=False), sha, task["created_at"]))
        return task

    def tasks(self, project_id):
        self.project(project_id)
        with self.connect() as db:
            rows = db.execute("""SELECT r.* FROM revisions r WHERE r.project_id=? AND r.revision=
                (SELECT MAX(b.revision) FROM revisions b WHERE b.task_id=r.task_id) ORDER BY r.created_at DESC""", (project_id,))
            return [{key: row[key] for key in ("task_id", "revision", "title", "sha256", "created_at")} for row in rows]

    def task(self, task_id, revision=None):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM revisions WHERE task_id=? " +
                             ("AND revision=? " if revision is not None else "") + "ORDER BY revision DESC LIMIT 1",
                             (task_id, revision) if revision is not None else (task_id,)).fetchone()
        if not row:
            raise FileNotFoundError("任务版本不存在")
        return json.loads(row["payload"])

    def revisions(self, task_id):
        self.task(task_id)
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT revision,sha256,created_at,title FROM revisions WHERE task_id=? ORDER BY revision DESC", (task_id,))]

    def index_run(self, run):
        evaluation = run.get("final_eval") or {}
        initial = run.get("initial_eval") or {}
        task = run["task"]
        summary = {"id": run["id"], "title": task.get("title", task["id"]), "project_id": task.get("project_id", ""),
                   "project_name": task.get("project_name", ""), "task_id": task["id"], "dataset": task.get("dataset", "demo"),
                   "revision": task.get("revision"), "mode": run["mode"], "status": run["status"],
                   "created_at": run["created_at"], "stage": run.get("stage"), "stop_reason": run.get("stop_reason"),
                   "passed": bool(evaluation.get("passed")) and run["status"] == "completed",
                   "score": "%s/%s" % (evaluation.get("passed_count", "–"), evaluation.get("total_count", "–")),
                   "initial_passed": initial.get("passed_count"), "final_passed": evaluation.get("passed_count"),
                   "total_tests": evaluation.get("total_count"), "rounds": len(run["rounds"]),
                   "elapsed_seconds": run.get("elapsed_seconds"), "error": run.get("error"),
                   "lessons": len(run["lessons"]), "model": run["model"]}
        with self.connect() as db:
            db.execute("""INSERT INTO runs VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, summary=excluded.summary""",
                       (summary["id"], summary["title"], summary["project_id"], summary["task_id"], summary["mode"],
                        summary["status"], summary["created_at"], json.dumps(summary, ensure_ascii=False)))

    def search_runs(self, query="", mode="", status="", project_id="", limit=20, offset=0):
        if mode not in ("", "live", "fixture") or status not in ("", "queued", "running", "completed", "error", "cancelled"):
            raise ValueError("无效的运行筛选条件")
        limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
        filters, params = [], []
        for column, value in (("mode", mode), ("status", status), ("project_id", project_id)):
            if value:
                filters.append(column + "=?")
                params.append(value)
        if query:
            filters.append("(title LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\')")
            pattern = "%" + query[:120].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            params.extend((pattern, pattern))
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self.connect() as db:
            count = db.execute("SELECT COUNT(*) FROM runs" + where, params).fetchone()[0]
            rows = db.execute("SELECT summary FROM runs" + where + " ORDER BY created_at DESC LIMIT ? OFFSET ?", params + [limit, offset])
            return {"items": [json.loads(row[0]) for row in rows], "total": count, "limit": limit, "offset": offset}

    def overview(self):
        with self.connect() as db:
            counts = {"runs": db.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                      "projects": db.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
                      "tasks": db.execute("SELECT COUNT(DISTINCT task_id) FROM revisions").fetchone()[0]}
            by_mode = {row["mode"]: row["n"] for row in db.execute("SELECT mode,COUNT(*) n FROM runs GROUP BY mode")}
            by_status = {row["status"]: row["n"] for row in db.execute("SELECT status,COUNT(*) n FROM runs GROUP BY status")}
        return {**counts, "by_mode": by_mode, "by_status": by_status, "recent": self.search_runs(limit=6)["items"]}

    def backup(self, destination):
        """Consistent SQLite backup, including a live WAL, without copying secrets."""
        with self.connect() as source, sqlite3.connect(str(destination)) as target:
            source.backup(target)
