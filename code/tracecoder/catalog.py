import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def demos():
    return [json.loads(p.read_text()) for p in sorted((DATA / "examples").glob("*.json"))]


def get_demo(task_id):
    for task in demos():
        if task["id"] == task_id:
            return task
    raise ValueError("Unknown demo task")


def get_task(dataset, task_id):
    if dataset == "demo":
        return get_demo(task_id)
    if dataset not in ("humaneval", "humanevalplus", "classeval", "bigcodebench"):
        raise ValueError("Unknown dataset")
    with (DATA / "benchmarks" / (dataset + ".jsonl")).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["task_id"] == task_id:
                return {"id": row["task_id"], "dataset": dataset,
                        "title": row.get("class_name") or row.get("entry_point") or row["task_id"],
                        "description": row.get("prompt") or row.get("complete_prompt") or row.get("skeleton"),
                        "test": row["test"], "entry_point": row.get("entry_point", ""),
                        "provenance": "由本目录提供的数据转换；测试代码保持原文。"}
    raise ValueError("Unknown task ID")


def public_demo(task):
    return {key: task[key] for key in ("id", "title", "subtitle", "description", "initial_code", "test", "provenance", "tags")}


def benchmark_index():
    return json.loads((DATA / "benchmark_index.json").read_text())
