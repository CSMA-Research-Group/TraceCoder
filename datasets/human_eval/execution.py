from typing import Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile


# Python 3.9+ for ast.unparse. For older versions, a library like 'astor' would be needed
# or compile the AST object directly. We will use compile directly.

# --- Helper function to count assertions (remains mostly the same) ---
def _get_total_test_cases(test_code: str, task_id_for_warning: str = "Unknown") -> int:
    """
    Parses the test script and counts the number of assert statements
    within the first function named 'check' found in the AST.
    This is a static count of defined assertions.
    """
    try:
        tree = ast.parse(test_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "check":
                assert_count = 0
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Assert):
                        assert_count += 1
                return assert_count
        return 0
    except SyntaxError:
        print(
            f"Warning: SyntaxError parsing test code for task_id {task_id_for_warning} (static count). Assuming 0 test cases.")
        return 0


# --- AST Transformer for individual assertion counting ---
class AssertTransformer(ast.NodeTransformer):
    def __init__(self, passed_var: str, failed_var: str, total_executed_var: str):
        self.passed_var = passed_var
        self.failed_var = failed_var
        self.total_executed_var = total_executed_var
        super().__init__()

    def visit_Assert(self, node: ast.Assert) -> list:  # Return a list of nodes
        # Increment total assertions encountered counter
        # globals()['VAR_NAME'] is used to ensure we modify the exec_globals scope
        inc_total_stmt_str = f"globals()['{self.total_executed_var}'] += 1"
        inc_total_stmt = ast.parse(inc_total_stmt_str).body[0]

        # Original assertion test and message
        test_expr = node.test
        msg_expr = node.msg

        # try:
        #   assert <original_test>, <original_msg>
        #   globals()['_eval_passed_count'] += 1
        # except AssertionError:
        #   globals()['_eval_failed_count'] += 1
        pass_stmt_str = f"globals()['{self.passed_var}'] += 1"
        pass_stmt = ast.parse(pass_stmt_str).body[0]

        fail_stmt_str = f"globals()['{self.failed_var}'] += 1"
        fail_stmt = ast.parse(fail_stmt_str).body[0]

        try_body = [
            ast.Assert(test=test_expr, msg=msg_expr),  # The original assert
            pass_stmt
        ]
        # Catch AssertionError, record failure, and continue execution
        except_handler = ast.ExceptHandler(
            type=ast.Name(id='AssertionError', ctx=ast.Load()),
            name=None,  # Don't need the exception instance 'e' for simple counting
            body=[fail_stmt]
        )
        try_node = ast.Try(body=try_body, handlers=[except_handler], orelse=[], finalbody=[])

        # Return a list: first the total increment, then the try-except block
        # This ensures that even if the assert itself isn't wrapped correctly later, total is counted.
        # However, for 'check' function, it's better to put inc_total_stmt inside the try or before,
        # but outside is fine if we assume all asserts in 'check' are meant to be tested.
        # For simplicity, we consider an "encountered" assertion as one that the AST transformer processes.
        return [inc_total_stmt, try_node]


def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    task_id = problem.get("task_id", "Unknown")

    # Static count of assertions defined in the 'check' function
    defined_total_count = _get_total_test_cases(problem["test"], task_id_for_warning=task_id)

    # Names for counters in exec_globals
    PASSED_COUNTER = "_eval_passed_count"
    FAILED_COUNTER = "_eval_failed_count"
    TOTAL_EXECUTED_COUNTER = "_eval_total_executed_assertions"

    # This function will be executed in a separate process.
    def unsafe_execute(result_list_proxy):
        exec_globals = {
            PASSED_COUNTER: 0,
            FAILED_COUNTER: 0,
            TOTAL_EXECUTED_COUNTER: 0,
            "__builtins__": __builtins__  # Ensure builtins are available
        }

        with create_tempdir():
            import os as _os_module
            import shutil as _shutil_module
            _original_os_chdir = _os_module.chdir
            _original_os_getcwd = _os_module.getcwd
            _original_os_rmdir = _os_module.rmdir
            _original_shutil_rmtree = _shutil_module.rmtree

            reliability_guard()

            # Transform the test code to count individual assertions
            try:
                test_ast = ast.parse(problem["test"])
                transformer = AssertTransformer(PASSED_COUNTER, FAILED_COUNTER, TOTAL_EXECUTED_COUNTER)
                transformed_test_ast = transformer.visit(test_ast)
                ast.fix_missing_locations(transformed_test_ast)  # Important after transformations
                # Compile the transformed AST to a code object
                # The 'filename' argument to compile helps in tracebacks.
                compiled_transformed_test = compile(transformed_test_ast, filename=f"<transformed_test_{task_id}>",
                                                    mode="exec")
            except Exception as e:
                # If transformation fails, we can't run granular tests
                result_list_proxy.append({
                    "status": f"failed: AST transformation error: {type(e).__name__}: {e}",
                    PASSED_COUNTER: 0,
                    FAILED_COUNTER: defined_total_count,  # All defined tests are failed
                    TOTAL_EXECUTED_COUNTER: 0
                })
                # Restore os/shutil functions before exiting context managers
                _os_module.chdir = _original_os_chdir
                _os_module.getcwd = _original_os_getcwd
                _os_module.rmdir = _original_os_rmdir
                _shutil_module.rmtree = _original_shutil_rmtree
                return

            # Construct the check program using the compiled transformed test
            # We exec the prompt and completion first, then the compiled test, then the check call.
            # This order ensures the `problem['entry_point']` function is defined by `prompt + completion`
            # before the transformed `check` function (from `compiled_transformed_test`) tries to use it.

            # The check_program will now be executed in stages or carefully combined.
            # Let's exec them separately into the same exec_globals for clarity
            program_to_run_entry_point = problem["prompt"] + completion
            check_call_str = f"\ncheck({problem['entry_point']})"  # The call to the (now transformed) check function

            # print(f"Task {task_id} - Executing entry point def:\n{program_to_run_entry_point[:200]}...")
            # print(f"Task {task_id} - Executing transformed test and check call...")

            try:
                with swallow_io():
                    with time_limit(timeout):
                        # Execute the candidate's code (prompt + completion)
                        exec(program_to_run_entry_point, exec_globals)
                        # Execute the transformed test definitions
                        exec(compiled_transformed_test, exec_globals)
                        # Execute the call to the check function
                        exec(check_call_str, exec_globals)

                # If exec completes without other exceptions, counts are in exec_globals
                result_list_proxy.append({
                    "status": "completed_execution",  # Indicates exec ran, individual counts matter
                    PASSED_COUNTER: exec_globals.get(PASSED_COUNTER, 0),
                    FAILED_COUNTER: exec_globals.get(FAILED_COUNTER, 0),
                    TOTAL_EXECUTED_COUNTER: exec_globals.get(TOTAL_EXECUTED_COUNTER, 0)
                })

            except TimeoutException:
                result_list_proxy.append({
                    "status": "timed out",
                    PASSED_COUNTER: 0,  # Or could try to get partial counts if they were updated before timeout
                    FAILED_COUNTER: defined_total_count,  # If timed out, all are considered failed
                    TOTAL_EXECUTED_COUNTER: exec_globals.get(TOTAL_EXECUTED_COUNTER, 0)
                    # How many were hit before timeout
                })
            except BaseException as e:  # Catch other errors during exec (e.g. SyntaxError in completion, runtime error)
                err_type = type(e).__name__
                err_msg = str(e).replace('\n', ' ')
                result_list_proxy.append({
                    "status": f"failed: {err_type}: {err_msg}",
                    PASSED_COUNTER: 0,  # Or partial counts if available and meaningful
                    FAILED_COUNTER: defined_total_count,
                    TOTAL_EXECUTED_COUNTER: exec_globals.get(TOTAL_EXECUTED_COUNTER, 0)
                })
            finally:
                _os_module.chdir = _original_os_chdir
                _os_module.getcwd = _original_os_getcwd
                _os_module.rmdir = _original_os_rmdir
                _shutil_module.rmtree = _original_shutil_rmtree

    manager = multiprocessing.Manager()
    execution_result_details_list = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(execution_result_details_list,))
    p.start()
    p.join(timeout=timeout + 1)

    passed_count = 0
    failed_count = 0
    # executed_count = 0 # Dynamically counted assertions that were run
    final_result_str = "failed: unknown execution error"  # Default

    if p.is_alive():
        p.kill()
        if not execution_result_details_list:  # If killed before appending
            execution_result_details_list.append({
                "status": "timed out (killed)",
                PASSED_COUNTER: 0,
                FAILED_COUNTER: defined_total_count,
                TOTAL_EXECUTED_COUNTER: 0
            })

    if not execution_result_details_list:
        # Process died without appending, or some other issue
        final_result_str = "failed: process terminated without result"
        failed_count = defined_total_count
    else:
        exec_details = execution_result_details_list[0]
        status = exec_details["status"]
        passed_count = exec_details[PASSED_COUNTER]
        failed_count_from_exec = exec_details[FAILED_COUNTER]
        # executed_count = exec_details[TOTAL_EXECUTED_COUNTER] # This is the number of asserts actually run

        if status == "completed_execution":
            if failed_count_from_exec == 0:
                # Check if all *defined* assertions were actually executed and passed
                # This can differ if 'check' has conditional asserts.
                # For strictness, one might check: passed_count == defined_total_count
                if passed_count >= defined_total_count:  # All defined asserts were executed and passed
                    final_result_str = "passed"
                elif passed_count > 0 and defined_total_count > 0:  # Some passed, ensure no fails
                    final_result_str = f"passed_partial: {passed_count}/{defined_total_count} assertions passed (check conditional logic?)"
                else:  # No fails, but also no passes, or executed less than defined
                    final_result_str = f"failed: 0/{defined_total_count} assertions passed (no asserts executed or all skipped?)"

            else:  # Some assertions failed
                final_result_str = f"failed: {passed_count}/{passed_count + failed_count_from_exec} assertions passed"

            failed_count = failed_count_from_exec  # This is the count of asserts that failed
            # `passed_count` is already set from exec_details

        else:  # Timeout, AST error, runtime error in completion etc.
            final_result_str = status  # e.g., "timed out", "failed: SyntaxError: ..."
            passed_count = 0  # No individual assertions considered passed
            failed_count = defined_total_count  # All defined assertions considered failed

    return dict(
        task_id=task_id,
        passed=final_result_str == "passed",  # Overall pass is only if all defined asserts passed and no other errors
        result=final_result_str,
        completion_id=completion_id,
        passed_count=passed_count,
        failed_count=failed_count,  # This should reflect actual failed assertions if execution completed, else total
        total_count=defined_total_count,  # Total assertions defined
    )


# --- Other helper functions (time_limit, swallow_io, create_tempdir, etc.) remain the same ---
# ... (ensure TimeoutException, WriteOnlyStringIO, redirect_stdin, chdir, reliability_guard are defined as before)
# (Make sure to copy them here if running this snippet standalone)

# Placeholder for other functions for completeness if you run this
@contextlib.contextmanager
def time_limit(seconds: float):
    if platform.system() == "Windows":  # Fallback for Windows
        yield
        return

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs): raise IOError

    def readline(self, *args, **kwargs): raise IOError

    def readlines(self, *args, **kwargs): raise IOError

    def readable(self, *args, **kwargs): return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    if maximum_memory_bytes is not None:
        if platform.system() != "Windows":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if not platform.uname().system == 'Darwin':
                resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None
    import os as guarded_os, shutil as guarded_shutil, subprocess as guarded_subprocess, sys as guarded_sys
    guarded_os.environ['OMP_NUM_THREADS'] = '1'
    for module in [guarded_os, guarded_shutil, guarded_subprocess]:  # type: ignore
        for attr in ['kill', 'system', 'putenv', 'remove', 'removedirs', 'rmdir', 'setuid',
                     'fork', 'forkpty', 'killpg', 'rename', 'renames', 'truncate', 'replace',
                     'unlink', 'fchmod', 'fchown', 'chmod', 'chown', 'chroot', 'lchflags',
                     'lchmod', 'lchown', 'getcwd', 'chdir', 'rmtree', 'move', 'Popen']:
            if hasattr(module, attr): setattr(module, attr, None)
    if isinstance(__builtins__, dict):
        __builtins__['help'] = None
    else:
        builtins.help = None
    for mod_name in ['ipdb', 'joblib', 'resource', 'psutil', 'tkinter']:
        guarded_sys.modules[mod_name] = None