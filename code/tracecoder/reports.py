"""Portable review artifacts, generated only from recorded execution evidence."""
import difflib
import hashlib
import html
import io
import json
import zipfile


def text_report(run):
    initial, final = run.get("initial_eval") or {}, run.get("final_eval") or {}
    score = lambda value: "%s / %s" % (value.get("passed_count", "—"), value.get("total_count", "—"))
    lines = ["TraceCoder · 修复评审报告", "=" * 48, "",
             "任务：" + run["task"].get("title", run["task"]["id"]), "运行 ID：" + run["id"],
             "创建时间：" + run["created_at"], "证据类型：" + run["evidence"],
             "模型：" + run["model"], "状态：" + run["status"], "停止原因：" + str(run.get("stop_reason")),
             "初始测试：" + score(initial), "最优测试：" + score(final),
             "修复轮数：%d；失败历史：%d" % (len(run["rounds"]), len(run["lessons"])),
             "", "项目版本：" + str(run["task"].get("revision", "不适用")),
             "测试 SHA-256：" + run["hashes"]["tests_sha256"],
             "代码 SHA-256：" + run["hashes"].get("final_code_sha256", "尚无最优代码"), ""]
    if run.get("error"):
        lines.extend(["执行错误：" + run["error"], ""])
    for r in run["rounds"]:
        a = r.get("analysis", {})
        lines.extend(["第 %d 轮 · %s" % (r["number"], r.get("decision", {}).get("action", "尚未决策")),
                      "候选测试：" + score(r.get("evaluation") or {}),
                      "根因：" + a.get("root_cause", "尚无诊断"),
                      "轨迹依据：" + "；".join(a.get("evidence", [])),
                      "修复计划：" + "；".join(a.get("repair_plan", [])),
                      "历史记录使用数：" + str(r.get("lesson_count_used", 0)), ""])
    lines.extend(["评审边界", "- 预设响应来自人工编写案例，执行结果来自本次测试；它不是模型能力实验。" if run["mode"] == "fixture" else
                  "- 本记录包含真实模型调用；测试通过仅覆盖本任务提供的固定测试。",
                  "- direct_pass 表示初始实现已通过，不计作修复成功。",
                  "- 候选补丁供人工评审；本系统不自动修改仓库、提交代码或合并变更。",
                  "- 全量提示、轨迹、失败候选和回滚决策见 run.json。"])
    return "\n".join(lines) + "\n"


def html_report(run):
    # Standalone, script-free, printable, with escaped user/model content.
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>TraceCoder · 修复评审报告</title><style>"
            "body{margin:48px auto;padding:0 24px;max-width:960px;color:#1e293b;font:14px/1.8 system-ui}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;border-top:3px solid #0891b2;padding-top:24px}"
            "@media print{body{margin:20px}pre{font-size:11px}}</style>"
            "<pre>" + html.escape(text_report(run)) + "</pre></html>")


def review_bundle(run):
    initial, final = run.get("initial_code", ""), run.get("final_code") or ""
    files = {"run.json": json.dumps(run, ensure_ascii=False, indent=2),
             "report.html": html_report(run), "report.txt": text_report(run),
             "initial.py": initial, "best.py": final, "fixed_tests.py": run["task"]["test"],
             "candidate.patch": "".join(difflib.unified_diff(initial.splitlines(True), final.splitlines(True),
                                                           fromfile="a/candidate.py", tofile="b/candidate.py")),
             "README.txt": "TraceCoder review bundle\n先阅读 report.html，再核查 run.json 与 candidate.patch。\n"
                           "candidate.py 为评审占位路径；请在代码仓库中人工核对目标路径。\n"
                           "best.py 是历史最优候选，可能仍有失败测试；不会自动应用补丁。\n"
                           "本包可能包含项目源代码与模型交互记录，请按项目权限分发。\n"}
    files["SHA256SUMS"] = "\n".join(hashlib.sha256(v.encode()).hexdigest() + "  " + k for k, v in files.items()) + "\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return buffer.getvalue()
