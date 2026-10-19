#!/usr/bin/env python3
"""Convert the supplied local data, with source hashes; never fetch substitutes.

Requires pyarrow only when rebuilding the two parquet-derived datasets.
The generated JSONL files are already included, so normal startup is stdlib-only.
"""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare():
    import pyarrow.parquet as pq
    DATA.mkdir(exist_ok=True)
    (DATA / "benchmarks").mkdir(exist_ok=True)
    sources = {
        "humaneval": "datasets/human_eval/data/HumanEval.jsonl.gz",
        "humanevalplus": "datasets/human_eval_plus/data/test-00000-of-00001-5973903632b82d40.parquet",
        "classeval": "datasets/ClassEval/data/ClassEval_data.json",
        "bigcodebench": "datasets/BigCodeBench/data/v0.1.4-00000-of-00001.parquet",
    }
    manifest, index, all_rows = [], {}, {}
    for name, relative in sources.items():
        source = ROOT / relative
        if source.suffix == ".gz":
            with gzip.open(source, "rt") as stream:
                rows = [json.loads(line) for line in stream]
        elif source.suffix == ".json":
            rows = json.loads(source.read_text())
        else:
            rows = pq.read_table(source).to_pylist()
        target = DATA / "benchmarks" / (name + ".jsonl")
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        all_rows[name] = rows
        manifest.append({"dataset": name, "source": relative, "source_sha256": sha(source),
                         "output": str(target.relative_to(ROOT)), "output_sha256": sha(target), "rows": len(rows),
                         "transformation": "JSON/JSONL serialization only; all fields including tests and reference solutions retained"})
        index[name] = [{"id": row["task_id"], "title": row.get("class_name") or row.get("entry_point") or row["task_id"]} for row in rows]
    write_json(DATA / "source_manifest.json", manifest)
    write_json(DATA / "benchmark_index.json", index)

    initial = """def next_sunday(day):
    days = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
    index = days.index(day)
    remaining = (7 - index) % 7
    return remaining
"""
    regression = initial.replace("(7 - index)", "(6 - index)")
    repaired = initial.replace("(7 - index) % 7", "7 - index")
    tests = "import unittest\n\nclass SundayTests(unittest.TestCase):\n" + "".join(
        "    def test_%s(self):\n        self.assertEqual(next_sunday(%r), %d)\n\n" % (day.lower(), day, value)
        for day, value in zip(["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"], [7, 6, 5, 4, 3, 2, 1]))
    sunday = {"id": "paper-sunday", "title": "下一个星期日", "subtitle": "论文 Figure 1 · 回滚与历史学习",
        "dataset": "demo", "tags": ["论文示例", "2 轮修复", "HLLM + RM"],
        "description": "输入今天的星期缩写 SUN、MON、TUE、WED、THU、FRI 或 SAT，返回距离下一个星期日的天数（从明天起算）。例如 MON → 6，WED → 4，SUN → 7。",
        "provenance": "题意和初始缺陷改编自论文 Figure 1；提取 index/remaining 便于观察。7 个测试由本复现补全。第一轮失败候选和第二轮正确候选均为人工编写，用于稳定展示回滚和历史学习。",
        "initial_code": initial, "entry_point": "next_sunday", "test": tests, "steps": [
            {"analysis": {"root_cause": "预设的失败假设：怀疑日期偏移常量有误，尝试整体减 1。此轮故意保留错误方案以展示回滚。",
                "evidence": ["SUN 的 index=0，remaining=0，预期为 7；其他日期当前通过。"], "lesson_feedback": "尚无历史修复记录。",
                "repair_plan": ["将 remaining 中的常量 7 改为 6，并由固定测试验证该假设。"],
                "instrumentation_suggestions": "观察所有日期的 index 和 remaining，确认是否破坏原本通过的日期。"},
             "code": regression, "change_summary": "预设失败候选：7 → 6；真实测试将检测到退化。"},
            {"analysis": {"root_cause": "模 7 将 SUN 对应的 7 变成 0；题目要求至少等待一天。索引本身没有偏移。",
                "evidence": ["回滚后 SUN: index=0, remaining=0。", "前次将 7 改成 6 使原本正确的 MON/WED 等日期也失败。"],
                "lesson_feedback": "历史记录显示整体偏移破坏了全部日期。保留 7-index，移除将 7 压回 0 的取模。",
                "repair_plan": ["从回滚后的历史最优代码开始。", "将 remaining 改为 7 - index，保留全部日期的索引规则。"],
                "instrumentation_suggestions": "若仍失败，继续记录 index 与 remaining，重点检查 SUN 和 SAT。"},
             "code": repaired, "change_summary": "删除取模，SUN 返回 7，其他日期保持正确。"}]}
    write_json(DATA / "examples" / "01_paper_sunday.json", sunday)

    human = all_rows["humaneval"][3]
    failed = """def below_zero(operations):
    balance = sum(operations)
    return balance < 0
"""
    fixed = """def below_zero(operations):
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
"""
    write_json(DATA / "examples" / "02_humaneval_balance.json", {
        "id": "humaneval-balance", "title": "账户余额的中间状态", "subtitle": "HumanEval/3 · 控制流与状态轨迹", "dataset": "demo",
        "tags": ["HumanEval/3", "原始测试", "中间状态"], "description": human["prompt"],
        "provenance": "题面和全部 6 个断言直接来自随附 HumanEval.jsonl.gz 的 HumanEval/3。初始代码及智能体响应为人工编写的教学缺陷，不宣称来自模型。",
        "initial_code": failed, "entry_point": human["entry_point"], "test": human["test"], "steps": [{
            "analysis": {"root_cause": "程序只观察最终余额，遗漏了中途透支后再次入账的情况。",
                "evidence": ["操作 [1, 2, -4, 5, 6] 的最终 balance=10，但第三次操作后余额为 -1。"],
                "lesson_feedback": "尚无历史失败方案；应检查每次状态变化，而非只替换最终比较符号。",
                "repair_plan": ["从余额 0 开始逐条执行操作。", "每次更新后立即判断余额是否小于 0；出现负数即返回 True。", "遍历完成且未透支时返回 False。"],
                "instrumentation_suggestions": "在 balance += op 后记录余额和当前操作。"},
            "code": fixed, "change_summary": "将最终求和改为逐操作检查。"}]})
    bracket = all_rows["classeval"][7]
    correct = bracket["solution_code"]
    assert correct.count("if self.stack:\n            return False") == 1
    wrong = correct.replace("if self.stack:\n            return False", "if self.stack:\n            return True")
    write_json(DATA / "examples" / "03_classeval_brackets.json", {
        "id": "classeval-brackets", "title": "括号匹配的残留栈", "subtitle": "ClassEval_7 · 类方法与对象状态", "dataset": "demo",
        "tags": ["ClassEval_7", "原始测试", "对象状态"], "description": bracket["skeleton"],
        "provenance": "题面、15 个 unittest 测试及参考实现来自随附 ClassEval_data.json 的 ClassEval_7。初始缺陷为人为将栈非空分支的 False 改成 True。预设修复恢复原参考实现；通过这组测试不代表覆盖所有未见输入。",
        "initial_code": wrong, "entry_point": "", "test": bracket["test"], "steps": [{
            "analysis": {"root_cause": "遍历结束后若栈仍有左括号，说明存在未闭合括号，当前却返回 True。",
                "evidence": ["输入 [a(b){c} 过滤成 [(){}，仍有左括号 [ 留在栈中。", "失败集中于未闭合表达式，clear_expr 测试仍通过。"],
                "lesson_feedback": "尚无历史修复；保留字符过滤与成对匹配逻辑。",
                "repair_plan": ["仅将循环后 self.stack 非空的返回值改为 False。"],
                "instrumentation_suggestions": "在最后的栈非空判断前观察 self.stack。"},
            "code": correct, "change_summary": "栈非空时返回 False，恢复随附参考实现。"}]})
    # Preserve a pre-work inventory. Existing root materials are never rewritten.
    inventory_path = DATA / "original_materials.sha256.json"
    if not inventory_path.exists():
        files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".pdf", ".md", ".txt")]
        files += [p for sub in ("src", "datasets", "results") for p in (ROOT / sub).rglob("*")
                  if p.is_file() and p.suffix not in (".pyc",) and p.name != ".DS_Store"]
        write_json(inventory_path, {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})
    print(json.dumps({name: len(rows) for name, rows in all_rows.items()}))


if __name__ == "__main__":
    prepare()
