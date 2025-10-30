import multiprocessing
import platform
import signal
import sys
import os
import contextlib
import io
import tempfile
import time
import ast
from typing import Dict, Optional


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs): raise IOError

    def readline(self, *args, **kwargs): raise IOError

    def readlines(self, *args, **kwargs): raise IOError

    def readable(self, *args, **kwargs): return False


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
def redirect_stdin(stream):
    old_stdin = sys.stdin
    sys.stdin = stream
    try:
        yield
    finally:
        sys.stdin = old_stdin


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    # 保存原始函数引用
    import os
    original_chdir = os.chdir
    original_getcwd = os.getcwd
    cwd = original_getcwd()
    original_chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        # 使用原始函数引用，避免因函数被覆盖而导致的问题
        try:
            original_chdir(cwd)
        except:
            # 如果无法切换回原目录，记录日志但不抛出异常
            pass


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def _get_total_test_cases(test_code: str) -> int:
    """
    解析测试代码并计算其中的assert语句数量
    """
    try:
        tree = ast.parse(test_code)
        assert_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_count += 1
        return assert_count
    except SyntaxError:
        return 0


def run_test_in_subprocess(prompt: str, completion: str, test_code: str, timeout: float, result_queue):
    """在子进程中运行测试"""
    try:
        # 计算总的测试用例数量
        total_count = _get_total_test_cases(test_code)
        if total_count == 0:
            total_count = 1  # 如果没有找到assert语句，默认为1个测试用例
        
        # 创建一个安全的执行环境
        def unsafe_execute():
            with create_tempdir():
                with swallow_io():
                    with time_limit(timeout):
                        # 合并提示代码、生成的代码和测试代码
                        full_code = prompt + completion + "\n" + test_code
                        
                        # 创建执行环境
                        exec_globals = {
                            "__builtins__": __builtins__,
                        }
                        
                        # 执行代码
                        exec(full_code, exec_globals)
                        
                        # 如果执行到这里没有异常，说明测试通过
                        return {
                            "status": "passed",
                            "passed_count": total_count,
                            "failed_count": 0,
                            "total_count": total_count,
                        }
        
        result = unsafe_execute()
        result_queue.put(result)
    except TimeoutException:
        result_queue.put({
            "status": "timed out",
            "passed_count": 0,
            "failed_count": total_count,
            "total_count": total_count,
        })
    except SyntaxError as e:
        # 特别处理语法错误
        result_queue.put({
            "status": f"failed: SyntaxError: {str(e)}",
            "passed_count": 0,
            "failed_count": total_count,
            "total_count": total_count,
        })
    except IndentationError as e:
        # 特别处理缩进错误
        result_queue.put({
            "status": f"failed: IndentationError: {str(e)}",
            "passed_count": 0,
            "failed_count": total_count,
            "total_count": total_count,
        })
    except Exception as e:
        result_queue.put({
            "status": f"failed: {type(e).__name__}: {str(e)}",
            "passed_count": 0,
            "failed_count": total_count,
            "total_count": total_count,
        })


def simple_check_correctness(problem: Dict, completion: str, timeout: float = 10.0) -> Dict:
    """
    简化版的测试函数，只使用生成的代码和测试用例来进行测试
    
    Args:
        problem: 包含问题信息的字典，应包含'prompt'和'test'键
        completion: 生成的代码
        timeout: 超时时间（秒）
        
    Returns:
        包含测试结果的字典
    """
    # 获取问题信息
    prompt = problem.get("prompt", "")
    test_code = problem.get("test", "")
    task_id = problem.get("task_id", "Unknown")
    
    # 使用多进程来隔离执行环境
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    
    # 启动子进程执行测试
    process = multiprocessing.Process(
        target=run_test_in_subprocess,
        args=(prompt, completion, test_code, timeout, result_queue)
    )
    process.start()
    process.join(timeout=timeout + 1)
    
    # 处理结果
    if process.is_alive():
        # 进程仍在运行，说明超时了
        process.terminate()
        process.join()
        total_count = _get_total_test_cases(test_code)
        if total_count == 0:
            total_count = 1
        return {
            "task_id": task_id,
            "passed": False,
            "result": "timed out",
            "passed_count": 0,
            "failed_count": total_count,
            "total_count": total_count,
        }
    
    # 获取执行结果
    if not result_queue.empty():
        result = result_queue.get()
        return {
            "task_id": task_id,
            "passed": result["status"] == "passed",
            "result": result["status"],
            "passed_count": result["passed_count"],
            "failed_count": result["failed_count"],
            "total_count": result["total_count"],
        }
    else:
        # 没有获取到结果，说明执行过程中出现了问题
        total_count = _get_total_test_cases(test_code)
        if total_count == 0:
            total_count = 1
        return {
            "task_id": task_id,
            "passed": False,
            "result": "failed: process terminated without result",
            "passed_count": 0,
            "failed_count": total_count,
            "total_count": total_count,
        }


# 示例用法
if __name__ == "__main__":
    # 示例问题
    problem = {
        "task_id": "test_problem",
        "prompt": "def add(a, b):\n    \"\"\"Return the sum of two numbers.\"\"\"\n",
        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n"
    }
    
    # 示例生成的代码
    generated_code = "    return a + b"
    
    # 运行测试
    result = simple_check_correctness(problem, generated_code, timeout=5.0)
    print(f"测试结果: {result}")