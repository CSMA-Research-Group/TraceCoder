from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import platform
import threading
import time
import uuid

from . import __version__
from .catalog import DATA, digest
from .instrumentation import validate_instrumentation
from .policy import rollback
from .providers import build_messages, parse_response
from .runner import feedback


@dataclass(frozen=True)
class Config:
    max_rounds: int = 5
    patience: int = 2
    timeout: int = 10
    enable_instrumentation: bool = True

    def __post_init__(self):
        if any(type(value) is not int for value in (self.max_rounds, self.patience, self.timeout)):
            raise ValueError("轮数、停滞阈值和超时必须是整数")
        if type(self.enable_instrumentation) is not bool:
            raise ValueError("enable_instrumentation 必须是布尔值")
        if not 1 <= self.max_rounds <= 5 or not 1 <= self.patience <= 5 or not 1 <= self.timeout <= 30:
            raise ValueError("轮数和停滞阈值必须在 1–5，超时必须在 1–30 秒。")


class Run:
    def __init__(self, task, provider, runner, config=None, initial_code=None, output_dir=None, on_persist=None):
        self.task, self.provider, self.runner = task, provider, runner
        self.config = config or Config()
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10]
        self.directory = Path(output_dir or DATA / "runs") / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.initial_code = initial_code
        self.on_persist = on_persist
        self.state = {"id": run_id, "status": "queued", "stage": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": provider.mode, "model": provider.model, "version": __version__, "host_python": platform.python_version(),
            "start_mode": "generate" if initial_code is None else "frozen",
            "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
            "task": {k: v for k, v in task.items() if k != "steps"}, "config": asdict(self.config),
            "evidence": "预设智能体响应 + 本次真实代码执行；不属于模型实验" if provider.mode == "fixture" else "本次实时模型调用 + 本次代码执行",
            "runner": runner.backend, "events": [], "rounds": [], "lessons": [], "calls": [],
            "hashes": {"tests_sha256": digest(task["test"]), "description_sha256": digest(task["description"])},
            "initial_eval": None, "final_eval": None, "final_code": None, "stop_reason": None}
        self.persist()

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def persist(self):
        with self.lock:
            target = self.directory / "run.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
            temporary.replace(target)
            if self.on_persist:
                self.on_persist(self.state)

    def event(self, stage, message, **details):
        with self.lock:
            self.state["stage"] = stage
            self.state["events"].append({"seq": len(self.state["events"]), "stage": stage,
                "message": message, "time": datetime.now(timezone.utc).isoformat(), **details})
            self.persist()

    def call(self, role, context, number):
        if self.cancelled.is_set():
            raise InterruptedError("用户停止运行")
        self.event(role, {"generation": "生成初始实现", "instrumentation": "插入诊断打印探针", "analysis": "分析运行轨迹与失败历史", "repair": "依据修复计划生成候选代码"}[role], round=number)
        call = {"role": role, "round": number, "model": self.provider.model,
                "messages": build_messages(role, context)}
        try:
            response, metadata = self.provider.complete(role, context, number)
            call.update(metadata)
            # Persist the raw response even if parsing/validation fails.
            with self.lock:
                self.state["calls"].append(call)
                self.persist()
            parsed = parse_response(role, response)
            if role in ("generation", "instrumentation", "repair"):
                if not isinstance(parsed.get("code"), str) or not parsed["code"].strip():
                    raise ValueError(role + " response has no code")
            else:
                for field in ("root_cause", "lesson_feedback", "instrumentation_suggestions"):
                    if not isinstance(parsed.get(field), str):
                        raise ValueError("Missing analysis field: " + field)
                if not isinstance(parsed.get("repair_plan"), list) or not parsed["repair_plan"] or not all(isinstance(x, str) for x in parsed["repair_plan"]):
                    raise ValueError("Analysis must return a nonempty repair_plan")
            return parsed
        except Exception as exc:
            call["error"] = str(exc)
            with self.lock:
                if not any(c is call for c in self.state["calls"]):
                    self.state["calls"].append(call)
                self.persist()
            raise

    def evaluate(self, code, stage, number):
        if self.cancelled.is_set():
            raise InterruptedError("用户停止运行")
        self.event(stage, "执行固定测试并捕获本次输出", round=number)
        return self.runner.evaluate(code, self.task)

    def execute(self):
        started = time.monotonic()
        with self.lock:
            self.state["status"] = "running"
            self.state["started_at"] = datetime.now(timezone.utc).isoformat()
            self.persist()
        try:
            current = self.initial_code
            if current is None:
                current = self.call("generation", {"description": self.task["description"]}, 0)["code"]
            with self.lock:
                self.state["initial_code"] = current
                self.state["hashes"]["initial_code_sha256"] = digest(current)
            initial = self.evaluate(current, "initial_test", 0)
            with self.lock:
                self.state["initial_eval"] = initial
                self.state["final_code"] = current
                self.state["final_eval"] = initial
            if not initial["valid"]:
                raise RuntimeError("初始执行无有效测试结果，不能建立修复基线：" + str(initial.get("error")))
            if initial["passed"]:
                return self.finish("direct_pass", started)
            best_code, best_eval, last_eval = current, initial, initial
            previous_score, stagnation, suggestions = initial["passed_count"], 0, ""
            baseline_ids = [case["id"] for case in initial["cases"]]
            for number in range(1, self.config.max_rounds + 1):
                record = {"number": number, "base_code": current, "base_eval": last_eval,
                          "best_before": best_eval["passed_count"], "previous_score": previous_score,
                          "lesson_count_used": len(self.state["lessons"])}
                with self.lock:
                    self.state["rounds"].append(record)
                instrumented = self.call("instrumentation", {"code": current, "failure_feedback": feedback(last_eval), "instrumentation_suggestions": suggestions}, number)["code"] if self.config.enable_instrumentation else current
                record["instrumented_code"] = instrumented
                record["instrumentation_check"] = validate_instrumentation(current, instrumented) if self.config.enable_instrumentation else {"ast_preserved": True, "probe_count": 0}
                record["instrumentation_check"]["enabled"] = self.config.enable_instrumentation
                if not self.config.enable_instrumentation:
                    self.event("instrumentation_disabled", "已按参数关闭插桩：这是消融配置，非完整 TraceCoder 方法", round=number)
                trace = self.evaluate(instrumented, "trace", number)
                record["trace"] = trace
                # Exact per-case status agreement is stronger than score equality.
                signature = lambda result: [(case["id"], case["status"]) for case in result["cases"]]
                if not trace["valid"] or signature(trace) != signature(last_eval):
                    record["instrumentation_check"]["test_outcomes_preserved"] = False
                    raise RuntimeError("插桩后的逐项测试结果变化；拒绝将该轨迹用于诊断。")
                record["instrumentation_check"]["test_outcomes_preserved"] = True
                analysis = self.call("analysis", {"description": self.task["description"], "instrumented_code": instrumented,
                    "runtime_trace": feedback(trace), "lesson_record": self.state["lessons"]}, number)
                record["analysis"] = analysis
                suggestions = analysis["instrumentation_suggestions"]
                candidate = self.call("repair", {"description": self.task["description"], "code": current,
                    "failure_feedback": feedback(trace), "repair_plan": analysis}, number)
                code = candidate["code"]
                record.update({"candidate_code": code, "change_summary": candidate.get("change_summary", ""),
                    "diff": "".join(difflib.unified_diff(current.splitlines(True), code.splitlines(True), fromfile="before.py", tofile="candidate.py"))})
                result = self.evaluate(code, "candidate_test", number)
                if result["valid"] and [case["id"] for case in result["cases"]] != baseline_ids:
                    result.update({"valid": False, "passed": False, "error": "测试集合与冻结基线不一致"})
                record["evaluation"] = result
                score = result["passed_count"] if result["valid"] else 0
                decision = rollback(code, score, best_code, best_eval["passed_count"], previous_score, stagnation)
                record["decision"] = asdict(decision)
                # Algorithm 1 logs every non-passing candidate, including partial improvements.
                if not result["passed"]:
                    self.state["lessons"].append({"round": number, "repair_plan": analysis,
                        "repaired_code": code, "error_feedback": feedback(result), "passed_count": score,
                        "changed_code": record["diff"], "decision": decision.action})
                if decision.action == "accept":
                    best_code, best_eval = code, result
                current, stagnation = decision.next_code, decision.stagnation
                # S_previous is the last ATTEMPT score, even after rollback.
                previous_score = score
                last_eval = best_eval if decision.action == "rollback" else result
                with self.lock:
                    self.state["final_code"], self.state["final_eval"] = best_code, best_eval
                self.event("decision", {"accept": "提升为当前最优实现", "rollback": "候选退化，回滚至历史最优实现", "continue": "保持当前候选继续探索"}[decision.action],
                           round=number, action=decision.action, attempted_score=score, best_score=best_eval["passed_count"], stagnation=stagnation)
                if result["passed"]:
                    return self.finish("all_tests_passed", started)
                if stagnation >= self.config.patience:
                    return self.finish("patience_exhausted", started)
            return self.finish("round_budget_exhausted", started)
        except InterruptedError as exc:
            return self.finish("cancelled", started, str(exc))
        except Exception as exc:
            return self.finish("error", started, str(exc))

    def finish(self, reason, started, error=None):
        with self.lock:
            self.state.update({"status": "error" if error and reason != "cancelled" else ("cancelled" if reason == "cancelled" else "completed"),
                "stop_reason": reason, "error": error, "elapsed_seconds": round(time.monotonic() - started, 3),
                "finished_at": datetime.now(timezone.utc).isoformat()})
            if self.state["final_code"] is not None:
                self.state["hashes"]["final_code_sha256"] = digest(self.state["final_code"])
                (self.directory / "final.py").write_text(self.state["final_code"])
            self.event("finished", error or reason)
            return self.snapshot()
