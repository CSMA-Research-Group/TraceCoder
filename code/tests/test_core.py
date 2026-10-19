import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tracecoder.catalog import demos, digest, get_demo, get_task
from tracecoder.engine import Config, Run
from tracecoder.instrumentation import fixture_probes, validate_instrumentation
from tracecoder.policy import rollback
from tracecoder.providers import CompatibleProvider, FixtureProvider, parse_object
from tracecoder.runner import Runner


class RollbackTests(unittest.TestCase):
    def test_strict_improvement_promotes_and_resets(self):
        result = rollback("new", 4, "best", 3, 2, 2)
        self.assertEqual((result.action, result.best_code, result.next_code, result.stagnation), ("accept", "new", "new", 0))

    def test_equal_keeps_candidate_and_retains_best(self):
        result = rollback("new", 3, "best", 3, 2, 0)
        self.assertEqual((result.action, result.best_code, result.next_code, result.stagnation), ("continue", "best", "new", 1))

    def test_regression_relative_to_previous_rolls_back(self):
        result = rollback("new", 1, "best", 3, 2, 1)
        self.assertEqual((result.action, result.next_code, result.stagnation), ("rollback", "best", 2))

    def test_below_best_but_not_previous_continues(self):
        result = rollback("new", 2, "best", 4, 1, 1)
        self.assertEqual((result.action, result.next_code, result.best_score), ("continue", "new", 4))


class InstrumentationTests(unittest.TestCase):
    def test_nested_probes_preserve_structure(self):
        source = "def f(x):\n    total = 0\n    for a in x:\n        total += a\n    return total\n"
        result = validate_instrumentation(source, fixture_probes(source))
        self.assertTrue(result["ast_preserved"])
        self.assertGreaterEqual(result["probe_count"], 4)

    def test_preserves_original_prints(self):
        original = "def f(x):\n    print(x)\n    return x\n"
        validate_instrumentation(original, "def f(x):\n    print('entry', x)\n    print(x)\n    return x\n")
        with self.assertRaises(ValueError):
            validate_instrumentation(original, "def f(x):\n    print('entry', x)\n    return x\n")

    def test_repair_disguised_as_probe_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_instrumentation("def f(x):\n    return x\n", "def f(x):\n    print(x)\n    return x+1\n")

    def test_side_effectful_probe_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_instrumentation("def f(x):\n    return x\n", "def f(x):\n    print(x.pop())\n    return x\n")

    def test_shadowed_print_rejected(self):
        with self.assertRaises(ValueError):
            fixture_probes("def f(print):\n    return print\n")

    def test_no_probes_rejected(self):
        with self.assertRaises(ValueError):
            validate_instrumentation("x = 1", "x = 1")


class RunnerTests(unittest.TestCase):
    def run_code(self, code, test, entry="f", timeout=3):
        return Runner(timeout=timeout).evaluate(code, {"test": test, "entry_point": entry})

    def test_check_collects_all_assertions_after_failures(self):
        result = self.run_code("def f(x):\n    return x\n", "def check(candidate):\n    assert candidate(0) == 1\n    assert candidate(2) == 2\n    assert candidate(3) == 4\n")
        self.assertEqual((result["passed_count"], result["total_count"], result["exit_code"]), (1, 3, 1))
        self.assertTrue(result["valid"])
        self.assertFalse(result["passed"])

    def test_candidate_exception_does_not_skip_later_assertions(self):
        result = self.run_code("def f(x):\n    return 1/x\n", "def check(candidate):\n    assert candidate(0) == 0\n    assert candidate(1) == 1\n")
        self.assertEqual((result["passed_count"], result["total_count"]), (1, 2))
        self.assertIn("ZeroDivisionError", result["cases"][0]["error"])

    def test_expanded_check_assertion_helper_adapter(self):
        result = self.run_code("def f(x):\n    return x\n", "def assertion(out, exp, atol):\n    assert out == exp\ndef check(candidate):\n    for x,y in [(1,1),(2,3),(4,4)]:\n        assertion(candidate(x),y,0)\n")
        self.assertEqual((result["passed_count"], result["total_count"]), (2, 3))

    def test_unittest_and_stdout_per_case(self):
        code = "def f(x):\n    print('state', x)\n    return x\n"
        test = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(f(1),1)\n    def test_b(self):\n        self.assertEqual(f(2),3)\n"
        result = self.run_code(code, test)
        self.assertEqual((result["passed_count"], result["total_count"]), (1, 2))
        self.assertIn("state 1", result["cases"][0]["stdout"])
        self.assertNotIn("state 2", result["cases"][0]["stdout"])

    def test_zero_tests_not_success(self):
        for test in ["", "def check(candidate):\n    pass\n"]:
            result = self.run_code("def f(): return 1", test)
            self.assertFalse(result["passed"])
            self.assertFalse(result["valid"])
            self.assertEqual(result["total_count"], 0)

    def test_syntax_failure_not_success(self):
        result = self.run_code("def f(:", "def check(candidate):\n    assert candidate() == 1\n")
        self.assertFalse(result["valid"])
        self.assertIn("SyntaxError", result["error"])

    def test_skip_not_success(self):
        result = self.run_code("x=1", "import unittest\nclass T(unittest.TestCase):\n    @unittest.skip('missing library')\n    def test_x(self): pass\n")
        self.assertFalse(result["passed"])
        self.assertEqual(result["passed_count"], 0)

    def test_subtests_keep_same_parent_test_ids(self):
        test = "import unittest\nclass T(unittest.TestCase):\n    def test_values(self):\n        for x in [1,2,3]:\n            with self.subTest(x=x):\n                self.assertEqual(f(x),x)\n"
        failing = self.run_code("def f(x): return 1", test)
        passing = self.run_code("def f(x): return x", test)
        self.assertEqual(failing["total_count"], 1)
        self.assertEqual(passing["total_count"], 1)
        self.assertEqual(failing["cases"][0]["id"], passing["cases"][0]["id"])
        self.assertFalse(failing["passed"])
        self.assertTrue(passing["passed"])

    def test_class_fixture_error_is_invalid_execution(self):
        test = "import unittest\nclass T(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls):\n        raise RuntimeError('fixture unavailable')\n    def test_value(self):\n        self.assertEqual(f(1),1)\n"
        result = self.run_code("def f(x): return x", test)
        self.assertFalse(result["valid"])
        self.assertIn("fixture", result["error"])

    def test_timeout_is_failure(self):
        result = self.run_code("def f():\n    while True: pass\n", "def check(candidate):\n    assert candidate()\n", timeout=1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid"])

    def test_environment_keys_not_inherited(self):
        with patch.dict(os.environ, {"TRACECODER_API_KEY": "private-test-marker"}):
            result = self.run_code("import os\ndef f():\n    return os.environ.get('TRACECODER_API_KEY')\n", "def check(candidate):\n    assert candidate() is None\n")
        self.assertTrue(result["passed"])


class EngineTests(unittest.TestCase):
    def execute(self, task, config=None, provider=None, initial=None):
        with tempfile.TemporaryDirectory(prefix="tracecoder-test-") as directory:
            run = Run(task, provider or FixtureProvider(task), Runner(), config or Config(),
                      initial if initial is not None else task["initial_code"], directory)
            result = run.execute()
            stored = json.loads((run.directory / "run.json").read_text())
            self.assertEqual(stored, result)
            return result

    def test_sunday_real_execution_rollback_and_hllm(self):
        task = get_demo("paper-sunday")
        result = self.execute(task)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([result["initial_eval"]["passed_count"]] + [r["evaluation"]["passed_count"] for r in result["rounds"]], [6, 0, 7])
        self.assertEqual([r["decision"]["action"] for r in result["rounds"]], ["rollback", "accept"])
        self.assertEqual(len(result["lessons"]), 1)
        self.assertEqual(result["rounds"][1]["lesson_count_used"], 1)
        self.assertEqual(result["rounds"][1]["base_code"], result["initial_code"])
        second_analysis = [c for c in result["calls"] if c["role"] == "analysis" and c["round"] == 2][0]
        self.assertIn(json.dumps(task["steps"][0]["code"], ensure_ascii=False)[1:-1], second_analysis["messages"][1]["content"])
        self.assertEqual(result["hashes"]["tests_sha256"], digest(task["test"]))
        self.assertTrue(all(c["response_source"] == "authored_fixture_not_llm" for c in result["calls"]))
        self.assertTrue(all(r["instrumentation_check"]["test_outcomes_preserved"] for r in result["rounds"]))

    def test_all_shipped_examples_pass_real_fixed_tests(self):
        for task in demos():
            with self.subTest(task=task["id"]):
                result = self.execute(task)
                self.assertTrue(result["final_eval"]["passed"], result.get("error"))
                self.assertGreater(result["final_eval"]["total_count"], 0)

    def test_patience_retains_best_and_failed_history(self):
        result = self.execute(get_demo("paper-sunday"), Config(patience=1))
        self.assertEqual(result["stop_reason"], "patience_exhausted")
        self.assertEqual(result["final_eval"]["passed_count"], 6)
        self.assertEqual(result["final_code"], result["initial_code"])
        self.assertEqual(len(result["lessons"]), 1)

    def test_round_limit_is_distinct(self):
        result = self.execute(get_demo("paper-sunday"), Config(max_rounds=1, patience=2))
        self.assertEqual(result["stop_reason"], "round_budget_exhausted")

    def test_direct_success_is_not_a_repair(self):
        task = get_demo("paper-sunday")
        result = self.execute(task, initial=task["steps"][-1]["code"])
        self.assertEqual(result["stop_reason"], "direct_pass")
        self.assertEqual(result["rounds"], [])
        self.assertEqual(result["calls"], [])

    def test_unfaithful_instrumentation_halts_before_analysis(self):
        task = get_demo("paper-sunday")
        class BadProbe(FixtureProvider):
            def complete(self, role, context, round_number):
                response, meta = super().complete(role, context, round_number)
                if role == "instrumentation":
                    response["code"] = response["code"].replace("% 7", "+ 7")
                return response, meta
        result = self.execute(task, provider=BadProbe(task))
        self.assertEqual(result["status"], "error")
        self.assertEqual([c["role"] for c in result["calls"]], ["instrumentation"])

    def test_invalid_response_keeps_raw_evidence(self):
        task = get_demo("paper-sunday")
        class Invalid(FixtureProvider):
            def complete(self, *args):
                return "not json", {"raw_response": "not json", "response_source": "test_double"}
        result = self.execute(task, provider=Invalid(task))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["calls"][0]["raw_response"], "not json")
        self.assertIn("error", result["calls"][0])

    def test_config_rejects_out_of_paper_budget(self):
        for values in [{"max_rounds": 6}, {"patience": 0}, {"timeout": 100}]:
            with self.assertRaises(ValueError):
                Config(**values)


class ProviderTests(unittest.TestCase):
    def test_json_fence_parsing(self):
        self.assertEqual(parse_object('```json\n{"code":"pass"}\n```'), {"code": "pass"})
        with self.assertRaises(ValueError):
            parse_object("[]")

    def test_transport_contract_without_real_model_call(self):
        requests = []
        class Response(io.BytesIO):
            pass
        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                return Response(json.dumps({"choices": [{"message": {"content": '{"code":"pass"}'}, "finish_reason": "stop"}], "usage": {"total_tokens": 12}}).encode())
        with patch.dict(os.environ, {"TRACECODER_API_BASE": "https://test.invalid/v1", "TRACECODER_MODEL": "test-model", "TRACECODER_API_KEY": "test-only-secret"}):
            provider = CompatibleProvider()
            with patch("urllib.request.build_opener", return_value=Opener()):
                raw, meta = provider.complete("generation", {"description": "Implement a function"}, 0)
        body = json.loads(requests[0].data)
        self.assertEqual(requests[0].full_url, "https://test.invalid/v1/chat/completions")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(parse_object(raw), {"code": "pass"})
        self.assertNotIn("test-only-secret", json.dumps(meta))
        self.assertNotIn("canonical_solution", json.dumps(body))

    def test_benchmark_task_does_not_expose_reference_solution(self):
        for dataset, task_id in [("humaneval", "HumanEval/0"), ("humanevalplus", "HumanEval/0"), ("classeval", "ClassEval_7"), ("bigcodebench", "BigCodeBench/0")]:
            task = get_task(dataset, task_id)
            self.assertNotIn("canonical_solution", task)
            self.assertNotIn("solution_code", task)
            self.assertTrue(task["test"])


if __name__ == "__main__":
    unittest.main()
