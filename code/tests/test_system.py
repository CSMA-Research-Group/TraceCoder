"""System contracts: immutable inputs, durable jobs, bounded queues and exports."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from app import Workbench
from tracecoder.catalog import DATA, get_demo
from tracecoder.engine import Config, Run
from tracecoder.providers import FixtureProvider
from tracecoder.reports import html_report, review_bundle
from tracecoder.runner import Runner
from tracecoder.storage import Store, validate_task

TEMPLATE = json.loads((DATA / "templates/api-window.task.json").read_text())


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.project = self.store.create_project({"name": "API project", "source_ref": "repo@abc"})

    def test_versions_preserve_previous_source_and_hash(self):
        first = self.store.save_task(self.project["id"], TEMPLATE)
        changed = {**TEMPLATE, "initial_code": TEMPLATE["initial_code"].replace("start <=", "start <")}
        second = self.store.save_task(self.project["id"], changed, first["id"])
        self.assertEqual((first["revision"], second["revision"]), (1, 2))
        self.assertNotEqual(first["revision_sha256"], second["revision_sha256"])
        self.assertEqual(self.store.task(first["id"], 1), first)
        self.assertEqual(self.store.task(first["id"]), second)
        self.assertEqual(len(self.store.tasks(self.project["id"])), 1)
        self.assertEqual(len(self.store.revisions(first["id"])), 2)

    def test_concurrent_version_numbers_are_unique(self):
        first = self.store.save_task(self.project["id"], TEMPLATE)
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(lambda _: self.store.save_task(self.project["id"], TEMPLATE, first["id"]), range(8)))
        self.assertEqual(sorted(v["revision"] for v in values), list(range(2, 10)))

    def test_unknown_project_cannot_replace_task(self):
        first = self.store.save_task(self.project["id"], TEMPLATE)
        other = self.store.create_project({"name": "another"})
        with self.assertRaises(FileNotFoundError):
            self.store.save_task(other["id"], TEMPLATE, first["id"])

    def test_save_never_executes_untrusted_code(self):
        path = Path(self.temp.name) / "MUST_NOT_EXIST"
        dangerous = {**TEMPLATE, "initial_code": "open(%r, 'w').write('executed')\n" % str(path)}
        self.store.save_task(self.project["id"], dangerous)
        self.assertFalse(path.exists())

    def test_import_rejects_unsupported_or_empty_test_contracts(self):
        for value in [{**TEMPLATE, "test": "def test_x(): assert True"},
                      {**TEMPLATE, "test": "def check(candidate): pass"},
                      {**TEMPLATE, "test": "def check(candidate): assert candidate()", "entry_point": "../x"},
                      {**TEMPLATE, "initial_code": "def invalid(:"},
                      {**TEMPLATE, "shell_command": "anything"}]:
            with self.assertRaises(ValueError):
                validate_task(value)

    def test_consistent_backup_restores_version_history(self):
        task = self.store.save_task(self.project["id"], TEMPLATE)
        dest = Path(self.temp.name) / "backup.db"
        self.store.backup(dest)
        with sqlite3.connect(str(dest)) as db:
            value = json.loads(db.execute("SELECT payload FROM revisions").fetchone()[0])
        self.assertEqual(value, task)


class QueueAndEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workbench = Workbench(state_dir=self.temp.name, workers=1, queue_limit=1)
        self.addCleanup(self.workbench.close)

    def test_bounded_queue_and_cancel_before_execution(self):
        entered, release = threading.Event(), threading.Event()
        original = Run.execute

        def hold(run):
            entered.set()
            release.wait(5)
            return original(run)

        with patch.object(Run, "execute", hold):
            first = self.workbench.create({"task_id": "paper-sunday"}, "test")
            self.assertTrue(entered.wait(2))
            second = self.workbench.create({"task_id": "api-window"}, "test")
            with self.assertRaisesRegex(ValueError, "队列已满"):
                self.workbench.create({}, "test")
            self.workbench.cancel(second["id"])
            cancelled = self.workbench.get(second["id"])
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(cancelled["calls"], [])
            self.assertIsNone(cancelled["initial_eval"])
            self.workbench.cancel(first["id"])
            release.set()
            self.workbench.close()
        self.assertEqual(self.workbench.get(first["id"])["stop_reason"], "cancelled")
        self.assertEqual(self.workbench.runs, {})
        self.assertEqual(self.workbench.futures, {})

    def test_complete_run_index_report_and_search(self):
        run = self.workbench.create({"task_id": "api-window"}, "test")
        deadline = time.monotonic() + 10
        while self.workbench.get(run["id"])["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(.02)
        result = self.workbench.get(run["id"])
        self.assertEqual(result["stop_reason"], "all_tests_passed")
        self.assertEqual(result["initial_eval"]["passed_count"], 4)
        self.assertEqual(result["final_eval"]["passed_count"], 6)
        self.assertEqual(self.workbench.store.search_runs(query="API", mode="fixture")["total"], 1)
        self.assertEqual(self.workbench.store.search_runs(query="%", mode="fixture")["total"], 0)
        self.assertEqual(self.workbench.store.search_runs(query="' OR 1=1 --")["total"], 0)
        with zipfile.ZipFile(io.BytesIO(review_bundle(result))) as archive:
            self.assertEqual(archive.read("best.py").decode(), result["final_code"])
            self.assertEqual(archive.read("fixed_tests.py").decode(), TEMPLATE["test"])
            self.assertIn("-        if start <=", archive.read("candidate.patch").decode())
            for line in archive.read("SHA256SUMS").decode().splitlines():
                sha, name = line.split("  ", 1)
                self.assertEqual(sha, hashlib.sha256(archive.read(name)).hexdigest())

    def test_recovery_marks_interrupted_jobs_without_replaying(self):
        task = get_demo("paper-sunday")
        run = Run(task, FixtureProvider(task), Runner(), Config(), task["initial_code"], self.workbench.run_dir)
        run.state["status"] = "running"
        run.persist()
        restarted = Workbench(state_dir=self.temp.name)
        try:
            result = restarted.get(run.state["id"])
            self.assertEqual(result["stop_reason"], "server_restarted")
            self.assertEqual(result["calls"], [])
            self.assertEqual(restarted.store.search_runs(status="error")["total"], 1)
        finally:
            restarted.close()

    def test_public_demo_and_unconfigured_model_fail_closed(self):
        task = self.workbench.store.save_task(self.workbench.store.create_project({"name": "test"})["id"], TEMPLATE)
        with self.assertRaisesRegex(ValueError, "固定 revision"):
            self.workbench.create({"mode": "live", "dataset": "project", "task_id": task["id"]}, "test")
        with self.assertRaisesRegex(ValueError, "新版本"):
            self.workbench.create({"mode": "live", "dataset": "project", "task_id": task["id"], "revision": 1, "initial_code": "x=1"}, "test")
        self.workbench.public_demo = True
        with self.assertRaisesRegex(ValueError, "私有工作台"):
            self.workbench.create({"mode": "live", "dataset": "project", "task_id": task["id"], "revision": 1}, "test")
        with self.assertRaises(ValueError):
            self.workbench.create({"max_rounds": 1.5}, "test")

    def test_standalone_report_escapes_untrusted_html(self):
        task = get_demo("api-window")
        task["title"] = "<script>alert('x')</script>"
        run = Run(task, FixtureProvider(task), Runner(), initial_code=task["initial_code"], output_dir=self.workbench.run_dir)
        report = html_report(run.snapshot())
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)


if __name__ == "__main__":
    unittest.main()
