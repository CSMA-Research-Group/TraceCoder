"""Standalone evaluation worker. Invoked in a fresh process/container per version.

Local execution is ONLY for the shipped, trusted fixtures. Resource limits are
not a security sandbox. Live generated code must use the Docker runner.
"""
import ast
import contextlib
import io
import json
import linecache
import os
import signal
import sys
import time
import traceback
import types
import unittest


class LimitedOutput(io.StringIO):
    def __init__(self, limit=12000):
        super().__init__()
        self.limit = limit
        self.truncated = False

    def write(self, text):
        remaining = self.limit - self.tell()
        if len(text) > remaining:
            self.truncated = True
        super().write(text[:max(0, remaining)])
        return len(text)

    def content(self):
        return self.getvalue() + ("\n[output truncated]" if self.truncated else "")


def register_source(name, source):
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    return compile(source, name, "exec")


def evaluate(payload):
    started = time.monotonic()
    cases = []
    module_output = LimitedOutput()
    namespace = types.ModuleType("candidate")
    sys.modules["candidate"] = namespace
    error = None
    collection = ""
    test_output = LimitedOutput()
    output_budget = [100000]

    def record(name, status, failure="", output=None):
        nonlocal test_output
        captured = output if output is not None else test_output.content()
        truncated = test_output.truncated or len(captured) > max(0, output_budget[0])
        captured = captured[:max(0, output_budget[0])]
        output_budget[0] -= len(captured)
        cases.append({"id": name, "status": status, "stdout": captured, "stdout_truncated": truncated,
                      "error": failure[-5000:], "error_truncated": len(failure) > 5000})
        test_output.seek(0)
        test_output.truncate(0)
        test_output.truncated = False
        if len(cases) > 30000:
            raise RuntimeError("Test execution limit exceeded")

    try:
        with contextlib.redirect_stdout(module_output), contextlib.redirect_stderr(module_output):
            exec(register_source("candidate.py", payload["code"]), namespace.__dict__)
        tests = types.ModuleType("tracecoder_tests")
        tests.__dict__.update(namespace.__dict__)
        tests.__name__ = "tracecoder_tests"
        sys.modules[tests.__name__] = tests
        test_source = payload["test"]
        test_tree = ast.parse(test_source)
        # Prevent only the conventional unittest.main entrypoint; preserve tests.
        test_tree.body = [node for node in test_tree.body if not (
            isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body))]
        check_defs = [n for n in test_tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
        class ModuleAssertions(ast.NodeVisitor):
            found = False
            def visit_Assert(self, node):
                self.found = True
            def visit_FunctionDef(self, node):
                pass
            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef
        module_assertions = ModuleAssertions()
        module_assertions.visit(test_tree)
        test_methods = [node for cls in test_tree.body if isinstance(cls, ast.ClassDef)
                        for node in cls.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test")]
        if sum((bool(check_defs), bool(test_methods), module_assertions.found)) > 1:
            raise ValueError("Mixed test entrypoints are unsupported; no tests are silently omitted")
        if check_defs or module_assertions.found:
            prefix = "check" if check_defs else "assert"
            collection = prefix + ": independently recorded assertion sites"
            index = [0]

            def run_assertion(callback, label):
                index[0] += 1
                try:
                    callback()
                except BaseException:
                    record(prefix + "/%04d: %s" % (index[0], label), "failed", traceback.format_exc())
                else:
                    record(prefix + "/%04d: %s" % (index[0], label), "passed")

            # Wrapping assertion sites keeps the remaining fixed cases running
            # after a failure. Helper implementations and inputs stay unchanged.
            class AssertionSites(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # Module helpers are code under test, not separate tests.
                    return self.generic_visit(node) if check_defs else node

                visit_AsyncFunctionDef = visit_FunctionDef
                visit_ClassDef = visit_FunctionDef

                def visit_Assert(self, node):
                    return self.wrap(node)

                def visit_Expr(self, node):
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "assertion":
                        return self.wrap(node)
                    return node

                def wrap(self, node):
                    label = ast.get_source_segment(test_source, node) or "assertion"
                    fn = ast.FunctionDef(name="_tc_assertion_site", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[node], decorator_list=[])
                    call = ast.Expr(value=ast.Call(func=ast.Name(id="_tc_record_assertion", ctx=ast.Load()), args=[ast.Name(id="_tc_assertion_site", ctx=ast.Load()), ast.Constant(value=label[:300])], keywords=[]))
                    return [ast.copy_location(fn, node), ast.copy_location(call, node)]

            target = check_defs[0] if check_defs else test_tree
            target.body = [child for node in target.body for child in (
                lambda result: result if isinstance(result, list) else [result])(AssertionSites().visit(node))]
            ast.fix_missing_locations(test_tree)
            tests.__dict__["_tc_record_assertion"] = run_assertion
            with contextlib.redirect_stdout(test_output), contextlib.redirect_stderr(test_output):
                exec(compile(test_tree, "tests.py", "exec"), tests.__dict__)
                if check_defs:
                    tests.__dict__["check"](namespace.__dict__[payload["entry_point"]])
        else:
            collection = "unittest.TestLoader"
            with contextlib.redirect_stdout(module_output), contextlib.redirect_stderr(module_output):
                exec(compile(test_tree, "tests.py", "exec"), tests.__dict__)
            suite = unittest.defaultTestLoader.loadTestsFromModule(tests)

            class Result(unittest.TestResult):
                def __init__(self):
                    super().__init__()
                    self.active = None
                    self.outcome = "passed"
                    self.details = []
                    self.fixture_errors = []

                def startTest(self, test):
                    super().startTest(test)
                    self.active, self.outcome, self.details = test, "passed", []
                    test_output.seek(0)
                    test_output.truncate(0)

                def outcome_for(self, test, status, detail=""):
                    if self.active is test:
                        self.outcome = status
                        if detail:
                            self.details.append(detail)
                    else:
                        # setUpClass/setUpModule failures have no startTest.
                        self.fixture_errors.append(detail or status)
                        record(test.id(), status, detail)

                def stopTest(self, test):
                    record(test.id(), self.outcome, "\n".join(self.details))
                    self.active = None
                    super().stopTest(test)

                def addSuccess(self, test):
                    super().addSuccess(test)
                    self.outcome_for(test, "passed")

                def addFailure(self, test, err):
                    super().addFailure(test, err)
                    self.outcome_for(test, "failed", "".join(traceback.format_exception(*err)))

                def addError(self, test, err):
                    super().addError(test, err)
                    self.outcome_for(test, "error", "".join(traceback.format_exception(*err)))

                def addSkip(self, test, reason):
                    super().addSkip(test, reason)
                    self.outcome_for(test, "skipped", reason)

                def addExpectedFailure(self, test, err):
                    super().addExpectedFailure(test, err)
                    self.outcome_for(test, "failed", "Expected failure is not counted as a pass")

                def addUnexpectedSuccess(self, test):
                    super().addUnexpectedSuccess(test)
                    self.outcome_for(test, "error", "Unexpected success")

                def addSubTest(self, test, subtest, err):
                    super().addSubTest(test, subtest, err)
                    if err is not None:
                        # Count the parent test method once, whether its subtests
                        # pass or fail, keeping the frozen test IDs comparable.
                        self.outcome_for(test, "failed", subtest.id() + "\n" + "".join(traceback.format_exception(*err)))

            with contextlib.redirect_stdout(test_output), contextlib.redirect_stderr(test_output):
                result = Result()
                suite.run(result)
            if result.fixture_errors:
                error = "Test fixture setup/teardown failed: " + "\n".join(result.fixture_errors)
        if not cases:
            error = "No tests collected. Zero tests never count as success."
        if any(case["id"].startswith("unittest.loader._FailedTest") for case in cases):
            error = "Test collection failed"
    except BaseException:
        error = traceback.format_exc()[-6000:]
    passed = sum(case["status"] == "passed" for case in cases)
    return {"passed": bool(cases) and passed == len(cases) and error is None,
            "passed_count": passed, "total_count": len(cases), "cases": cases,
            "valid": bool(cases) and error is None, "error": error,
            "collection": collection, "module_stdout": module_output.content(),
            "duration_seconds": round(time.monotonic() - started, 4),
            "python_version": sys.version.split()[0]}


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read(2_000_000))
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (payload["timeout"] + 1, payload["timeout"] + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4_000_000, 4_000_000))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024**2, 1536 * 1024**2))
    except (ImportError, ValueError, OSError):
        pass
    result = evaluate(payload)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["passed"] else 1)
