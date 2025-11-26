import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile
import subprocess
import multiprocessing
import time
import types
import unittest
import sys
import resource # Ensure resource is imported for reliability_guard
import traceback # Import traceback for detailed error reporting

# --- Assumed Helper Preamble (from problem description context) ---
# Ensure these definitions are available in the execution environment.

TIMEOUT_LIMIT = 10.0  # Default execution timeout for a task

class TimeoutException(Exception):
    """Custom exception for timeouts triggered by the signal handler."""
    pass

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""
    def read(self, *args, **kwargs):
        raise IOError("Cannot read from WriteOnlyStringIO")
    def readline(self, *args, **kwargs):
        raise IOError("Cannot read from WriteOnlyStringIO")
    def readlines(self, *args, **kwargs):
        raise IOError("Cannot read from WriteOnlyStringIO")
    def readable(self, *args, **kwargs):
        return False

@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager to limit execution time using SIGALRM."""
    # Check if SIGALRM is available (not on Windows)
    if not hasattr(signal, 'SIGALRM'):
        # print("Warning: SIGALRM not available on this platform. Timeout cannot be enforced.")
        yield # Proceed without timeout on platforms without SIGALRM
        return

    def signal_handler(signum, frame):
        """Signal handler that raises a TimeoutException."""
        raise TimeoutException(f"Timed out after {seconds} seconds!")

    # Store the original signal handler and timer setting
    original_handler = signal.getsignal(signal.SIGALRM)
    # Set the timer. ITIMER_REAL decrements in real time.
    signal.setitimer(signal.ITIMER_REAL, seconds)
    # Register the custom signal handler for SIGALRM
    signal.signal(signal.SIGALRM, signal_handler)

    try:
        yield # Execute the code block within the context manager
    finally:
        # Clear the timer by setting the interval to 0
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Restore the original signal handler
        signal.signal(signal.SIGALRM, original_handler)


@contextlib.contextmanager
def swallow_subprocess_output():
    """
    Context manager to redirect subprocess stdout/stderr to PIPE
    if not explicitly set, preventing output leakage to the main process's streams.
    This version simplifies the original patch slightly for clarity.
    """
    original_popen = subprocess.Popen
    original_run = subprocess.run

    # Note: The original patch logic had a slight inversion.
    # If capture_output=True, subprocess handles piping internally.
    # We want to force piping *unless* capture_output=True OR std streams are explicitly set.
    # A simpler approach is to *always* default to PIPE if not set,
    # unless the user explicitly provides None or DEVNULL.
    # However, for the purpose of *swallowing*, we force PIPE if stdout/stderr are default (None).

    def _popen_patch(*args, **kwargs):
        if kwargs.get('stdout') is None:
            kwargs['stdout'] = subprocess.PIPE
        if kwargs.get('stderr') is None:
            kwargs['stderr'] = subprocess.PIPE
        # Remove capture_output if present, as we handle piping explicitly
        kwargs.pop('capture_output', None)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if kwargs.get('stdout') is None:
            kwargs['stdout'] = subprocess.PIPE
        if kwargs.get('stderr') is None:
            kwargs['stderr'] = subprocess.PIPE
        # Remove capture_output if present, as we handle piping explicitly
        kwargs.pop('capture_output', None)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run

# Determine the correct base class for redirect_stdin based on Python version
# This handles changes in the internal structure of contextlib across versions.
if sys.version_info < (3, 10):
    # For Python < 3.10, _RedirectStream was the typical base class.
    _RedirectStreamBase = contextlib._RedirectStream # type: ignore
elif hasattr(contextlib, '_RedirectStreamCM'):
     # For Python 3.10+, _RedirectStreamCM might be used.
    _RedirectStreamBase = contextlib._RedirectStreamCM # type: ignore
elif hasattr(contextlib, '_RedirectStream'):
    # Fallback to _RedirectStream if _RedirectStreamCM isn't present.
    _RedirectStreamBase = contextlib._RedirectStream # type: ignore
else:
    # As a last resort, define a minimal placeholder if internals changed drastically.
    # This might lack functionality but prevents import errors.
    class _MinimalRedirectStream:
        def __init__(self, new_target):
            self._new_target = new_target
        def __enter__(self):
            # Basic redirection logic would go here if needed
            pass
        def __exit__(self, exctype, excinst, exctb):
            # Restore logic would go here
            pass
    _RedirectStreamBase = _MinimalRedirectStream


class redirect_stdin(_RedirectStreamBase): # type: ignore
    """Context manager to redirect sys.stdin."""
    _stream = "stdin"


@contextlib.contextmanager
def create_tempdir():
    """
    Creates a temporary directory, changes the current working directory to it,
    and automatically cleans up the directory on exit.
    """
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname
            # TemporaryDirectory automatically cleans up 'dirname'

@contextlib.contextmanager
def chdir(root):
    """
    Context manager to temporarily change the current working directory.
    """
    if root == ".": # No change needed if root is the current directory
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        # Ensure we change back to the original directory even if errors occur
        os.chdir(cwd)

@contextlib.contextmanager
def safe_environment():
    """
    Placeholder context manager for potentially setting up a safer execution environment.
    A full implementation might involve patching os functions (e.g., network, file access).
    """
    # In a real scenario, this could involve:
    # - Patching os.system, os.popen, subprocess functions further
    # - Patching network libraries (socket, requests)
    # - Using security contexts or containers if available
    # For now, it's a no-op placeholder.
    yield


def reliability_guard(max_as_limit_mb, max_data_limit_mb, max_stack_limit_mb):
    """
    Applies various settings to enhance execution reliability and security.
    Sets timezone, thread limits, TensorFlow log levels, disables faulthandler,
    sets resource limits, and patches exit/quit builtins.
    """
    # Set timezone to UTC
    os.environ['TZ'] = 'UTC'
    if hasattr(time, 'tzset'):
        time.tzset() # Apply the timezone change

    # Limit CPU cores used by libraries like OpenMP, MKL, etc.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1" # Often used with numpy/scipy
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # For Numexpr
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # For OpenBLAS

    # Reduce verbosity of TensorFlow if used
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" # Errors only
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0" # Disable oneDNN custom ops (can cause issues)

    # Create a new session ID (on Unix-like systems)
    # This helps isolate the process group.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
             # Can fail if already a process group leader
            pass

    # Set resource limits (if possible)
    if hasattr(resource, "setrlimit"):
        # Convert MB to bytes
        # Note: Original code had a factor of 10, keeping it for consistency, but usually it's just 1024*1024
        factor = 1024 * 1024 * 10
        max_as_limit = max_as_limit_mb * factor
        max_data_limit = max_data_limit_mb * factor
        max_stack_limit = max_stack_limit_mb * factor
        try:
            # RLIMIT_AS: Max virtual memory size
            resource.setrlimit(resource.RLIMIT_AS, (max_as_limit, max_as_limit))
            # RLIMIT_DATA: Max heap size
            resource.setrlimit(resource.RLIMIT_DATA, (max_data_limit, max_data_limit))
            # RLIMIT_STACK: Max stack size (not available/effective on macOS)
            if platform.system() != "Darwin":
                 resource.setrlimit(resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit))
        except Exception as e:
             # It's common for setrlimit to fail due to permissions or existing limits
             # print(f"Warning: Failed to set resource limits: {e}")
            pass

    # Disable faulthandler in the child process to avoid duplicate crash reports
    # The parent process might handle fault reporting if needed.
    faulthandler.disable()

    # Patch builtins.exit and builtins.quit to prevent clean exits
    # that might bypass error reporting. Use os._exit for immediate termination.
    import builtins
    builtins.exit = lambda *args, **kwargs: os._exit(1)
    builtins.quit = lambda *args, **kwargs: os._exit(1)

    # Attempt to configure Matplotlib for non-interactive use
    try:
        import matplotlib
        matplotlib.use('Agg') # Use a non-GUI backend
        import matplotlib.pyplot as plt
        plt.close('all') # Close any pre-existing plots
    except ImportError:
        pass # Matplotlib not installed, ignore

# --- End of Assumed Helper Preamble ---

# Constants for status codes used for inter-process communication
_SUCCESS = 0        # Tests ran, all passed
_FAILED = 1         # Tests ran, some failed/errored
_TIMEOUT = 2        # Execution exceeded time limit (internal or external)
_UNKNOWN = 3        # Initial state, or state if child process dies unexpectedly
_FRAMEWORK_ERROR = 4 # Error within the execution framework itself (setup, exec, etc.)

# Maximum size for captured stdout/stderr to prevent memory issues
MAX_PRINT_OUTPUT_SIZE = 1 * 1024 * 1024 # 1 MB

@contextlib.contextmanager
def capture_std_streams_for_reporting():
    """
    Context manager to capture stdout and stderr into a single StringIO object,
    while redirecting stdin to a write-only stream and suppressing subprocess output.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    # Use a single StringIO to capture both streams sequentially
    string_io = io.StringIO()
    try:
        sys.stdout = string_io
        sys.stderr = string_io
        # Redirect stdin to prevent reading, swallow subprocess output
        with redirect_stdin(WriteOnlyStringIO()), swallow_subprocess_output():
            yield string_io # Provide the StringIO object to the inner context
    finally:
        # Ensure original streams are restored
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def unsafe_execute_with_details(
    generated_code_str: str,
    test_code_str: str,
    internal_timeout_duration: float,
    max_as_limit_mb: int,
    max_data_limit_mb: int,
    max_stack_limit_mb: int,
    # Shared memory objects for results (Proxies/Array from multiprocessing.Manager):
    status_val,            # Value('i', _UNKNOWN) -> Final status code
    passed_count_val,      # Value('i', 0) -> Number of passed tests
    failed_count_val,      # Value('i', 0) -> Number of failed/errored tests
    total_count_val,       # Value('i', 0) -> Total number of tests run
    error_details_dict,    # Dict() -> Dictionary for storing error tracebacks/messages
    prints_buffer_shared,  # Array('c', MAX_PRINT_OUTPUT_SIZE) -> Shared buffer for stdout/stderr
    prints_length_shared   # Value('i', 0) -> Actual length of data written to prints_buffer_shared
):
    """
    Executes generated code and test code within a sandboxed environment in a child process.

    Applies reliability guards, captures output, runs tests, handles timeouts and errors,
    and reports results back to the parent process via shared memory objects.

    WARNING: This function is intended to be run in a separate process. It applies
    process-wide settings (resource limits, patches builtins).
    """
    # --- 1. Apply Reliability Guard ---
    try:
        # Apply resource limits, env vars, patches etc. *first*
        reliability_guard(max_as_limit_mb, max_data_limit_mb, max_stack_limit_mb)
    except Exception as e_guard:
        # If guarding fails, report as framework error and exit
        status_val.value = _FRAMEWORK_ERROR
        error_details_dict["RELIABILITY_GUARD_ERROR"] = f"Failed to apply reliability guard: {str(e_guard)}\n{traceback.format_exc()}"
        # Try to report the error message via the prints buffer as well
        encoded_prints = str(e_guard).encode('utf-8', 'replace')
        length_to_copy = min(len(encoded_prints), MAX_PRINT_OUTPUT_SIZE - 1)
        # Use bytearray assignment which is compatible with Array('c')
        prints_buffer_shared[:length_to_copy] = encoded_prints[:length_to_copy]
        if length_to_copy < MAX_PRINT_OUTPUT_SIZE:
             prints_buffer_shared[length_to_copy] = 0 # Null-terminate if space allows
        prints_length_shared.value = length_to_copy
        # No further execution possible
        return # Exit the function (and thus the child process)

    # Variable to hold captured prints, initialized empty
    current_prints_output_str = ""

    # --- 2. Setup Execution Environment ---
    # Use safe_environment (placeholder) and create a temporary directory
    # All file operations should ideally be contained within this temp dir.
    with safe_environment(), create_tempdir():
        # Define a unique module name for the executed code
        module_name = "__exec_module__"
        # Create a dictionary to serve as the global namespace for exec
        exec_globals = types.ModuleType(module_name).__dict__
        exec_globals.update({
            '__builtins__': __builtins__, # Provide standard builtins
            '__file__': f"{module_name}.py", # Simulate a file path
            'unittest': unittest, # Explicitly provide unittest in the execution scope
        })

        # Basic type checking for inputs (belt-and-suspenders)
        if not isinstance(generated_code_str, str):
            generated_code_str = "" # Default to empty string if not a string
            error_details_dict["SETUP_ERROR"] = "Generated code received was not a string."
            # Potentially set status to _FRAMEWORK_ERROR here if this is critical
        if not isinstance(test_code_str, str):
            test_code_str = ""
            error_details_dict["SETUP_ERROR"] = "Test code received was not a string."

        # Combine the generated code and test code into a single script
        full_code_to_execute = f"{generated_code_str}\n\n{test_code_str}"

        # --- 3. Execute Code and Run Tests ---
        try:
            # Capture stdout/stderr, redirect stdin, suppress subprocess output
            with capture_std_streams_for_reporting() as captured_streams_io:
                # Apply the internal time limit for the core execution part
                with time_limit(internal_timeout_duration):
                    # Compile the code first to catch syntax errors early
                    try:
                        compiled_code = compile(full_code_to_execute, f"{module_name}.py", 'exec')
                    except SyntaxError as e_compile:
                         # Treat compile errors as framework errors related to the input code
                        status_val.value = _FRAMEWORK_ERROR
                        error_details_dict["COMPILE_ERROR"] = f"Syntax error during compilation:\n{traceback.format_exc()}"
                        # Re-raise to be caught by the outer exception handler which handles prints
                        raise e_compile

                    # Execute the compiled code in the prepared namespace
                    exec(compiled_code, exec_globals)

                    # --- Find and Run Unit Tests ---
                    # Check if 'TestCases' (the expected test class name) exists
                    if 'TestCases' not in exec_globals:
                        raise NameError("The 'TestCases' class was not found after executing the provided code. "
                                        "Ensure the test code defines a class named 'TestCases' inheriting "
                                        "from 'unittest.TestCase'.")

                    TestCasesCls = exec_globals['TestCases']

                    # Validate that 'TestCases' is a class and subclasses unittest.TestCase
                    if not isinstance(TestCasesCls, type) or not issubclass(TestCasesCls, unittest.TestCase):
                        raise TypeError(f"'TestCases' must be a class and a subclass of unittest.TestCase, but got {type(TestCasesCls)}.")

                    # Discover tests within the TestCases class
                    loader = unittest.TestLoader()
                    suite = loader.loadTestsFromTestCase(TestCasesCls)

                    # Use a dummy stream for the runner to prevent double printing
                    # (we are already capturing stdout/stderr)
                    runner_stream = io.StringIO()
                    test_runner = unittest.TextTestRunner(stream=runner_stream, verbosity=0, failfast=False) # Set verbosity to 0
                    test_result = test_runner.run(suite)

                    # --- Report Test Results ---
                    # Update shared memory with test counts
                    total_count_val.value = test_result.testsRun
                    num_failures = len(test_result.failures)
                    num_errors = len(test_result.errors)
                    failed_count_val.value = num_failures + num_errors
                    passed_count_val.value = test_result.testsRun - failed_count_val.value

                    # Determine final status based on test results
                    if test_result.wasSuccessful():
                        # Check if any tests actually ran
                        if test_result.testsRun > 0:
                            status_val.value = _SUCCESS
                        else:
                            # Successful run but no tests found/executed
                            status_val.value = _FRAMEWORK_ERROR # Treat as error if TestCases was empty or misconfigured
                            error_details_dict["EXECUTION_INFO"] = "Test execution finished successfully, but zero tests were run. Check TestCases definition."
                    else:
                        status_val.value = _FAILED

                    # Store detailed tracebacks for failures and errors
                    for test_case, traceback_str in test_result.failures:
                        test_name = getattr(test_case, '_testMethodName', str(test_case).split(' ')[0]) # Get method name
                        error_details_dict[f"FAILURE_{test_name}"] = traceback_str
                    for test_case, traceback_str in test_result.errors:
                        test_name = getattr(test_case, '_testMethodName', str(test_case).split(' ')[0]) # Get method name
                        error_details_dict[f"ERROR_{test_name}"] = traceback_str

                # If execution reached here without timeout, store captured output
                current_prints_output_str = captured_streams_io.getvalue()

        # --- 4. Handle Exceptions ---
        except TimeoutException as e_timeout:
            # Internal time limit exceeded
            status_val.value = _TIMEOUT
            error_details_dict["INTERNAL_TIMEOUT"] = f"Test execution timed out after {internal_timeout_duration}s: {str(e_timeout)}"
            # Try to get whatever output was captured before the timeout
            if 'captured_streams_io' in locals() and captured_streams_io:
                current_prints_output_str = captured_streams_io.getvalue()
            else:
                # Should not happen if capture_std_streams_for_reporting worked
                current_prints_output_str = "[No stream capture available at timeout]"

        except Exception as e_exec:
            # Catch any other exception during exec, test loading, or running
            status_val.value = _FRAMEWORK_ERROR
            tb_str = traceback.format_exc() # Get full traceback
            error_type = type(e_exec).__name__
            error_details_dict["EXECUTION_FRAMEWORK_ERROR"] = f"Error during test execution ({error_type}): {str(e_exec)}\n{tb_str}"
            # Append the traceback to the captured output if possible
            if 'captured_streams_io' in locals() and captured_streams_io:
                # Ensure we capture output generated *before* the exception
                current_prints_output_str = captured_streams_io.getvalue() + f"\n--- Framework Error Traceback ---\n{tb_str}"
            else:
                current_prints_output_str = f"--- Framework Error Traceback (no stream capture) ---\n{tb_str}"

        # --- 5. Finalize and Report Output ---
        finally:
            # Encode the captured output string to bytes (UTF-8, replacing errors)
            encoded_prints = current_prints_output_str.encode('utf-8', 'replace')
            # Determine how many bytes to copy, respecting the buffer size limit
            length_to_copy = min(len(encoded_prints), MAX_PRINT_OUTPUT_SIZE - 1) # Reserve space for potential null terminator

            # Copy the encoded bytes into the shared memory buffer
            # Directly assign the slice of bytes to the Array 'c' slice
            prints_buffer_shared[:length_to_copy] = encoded_prints[:length_to_copy]

            # Null-terminate the string in the shared buffer if space permits.
            # This helps C-style consumers, though Python consumers will use the length.
            if length_to_copy < MAX_PRINT_OUTPUT_SIZE:
                 prints_buffer_shared[length_to_copy] = 0 # Assign integer 0 for null byte

            # Store the actual length of the copied data (excluding potential null term)
            prints_length_shared.value = length_to_copy

            # Clean up the executed module from sys.modules to avoid potential state leakage
            # if the process were somehow reused (unlikely here, but good practice).
            if module_name in sys.modules:
                try:
                    del sys.modules[module_name]
                except KeyError:
                    pass # Ignore if already deleted or never fully loaded

# --- Main Orchestration Function (Parent Process) ---

def check_correctness(problem_data: dict, generated_code_str: str, code_generation_time: float) -> dict:
    """
    Orchestrates the sandboxed execution of generated code against test code.

    Sets up a separate process, manages communication via shared memory,
    handles overall timeouts and process termination, and formats the final result.

    Args:
        problem_data: Dictionary potentially containing 'test' key with test code string.
        generated_code_str: The Python code string generated by the LLM.
        code_generation_time: Time taken for code generation (passed through).

    Returns:
        A dictionary containing:
        - 'passed': Boolean indicating if all tests passed.
        - 'passed_count', 'failed_count', 'total_count': Integer counts of tests.
        - 'result': String containing error messages or tracebacks if execution failed.
        - 'prints_output': String containing captured stdout/stderr (truncated if > MAX_PRINT_OUTPUT_SIZE).
        - 'status': String ("success", "failure", "timeout", "error").
        - 'llm_code_generation_time': The input code_generation_time.
        - Debug fields (_debug_...) providing internal status details.
    """
    # --- 1. Input Validation and Setup ---
    test_code_str = problem_data.get('test')

    # Handle missing test code early
    if not test_code_str:
        return {
            'passed': False,
            'passed_count': 0, 'failed_count': 0, 'total_count': 0,
            'result': 'SETUP_ERROR: Test code ("test" key) not found in problem_data.',
            'prints_output': '', 'status': 'error',
            'llm_code_generation_time': code_generation_time,
            '_debug_process_status_summary': 'setup_failed_no_test_code',
            '_debug_final_status_code': _FRAMEWORK_ERROR,
            '_debug_process_exit_code': None
        }

    # Ensure generated_code_str is also a string
    if not isinstance(generated_code_str, str):
        # Decide whether to treat as error or empty string
        # Returning an error is safer:
         return {
            'passed': False,
            'passed_count': 0, 'failed_count': 0, 'total_count': 0,
            'result': 'SETUP_ERROR: Generated code is not a string.',
            'prints_output': '', 'status': 'error',
            'llm_code_generation_time': code_generation_time,
            '_debug_process_status_summary': 'setup_failed_bad_gen_code_type',
            '_debug_final_status_code': _FRAMEWORK_ERROR,
            '_debug_process_exit_code': None
        }
        # Alternatively: generated_code_str = "" # Allow execution with empty generated code

    # Define timeouts and resource limits
    process_overall_timeout = TIMEOUT_LIMIT # Max time the parent waits for the child process
    # Internal timeout for the test execution logic within the child
    # Give a small buffer (e.g., 5s) for process startup/shutdown vs overall timeout
    internal_test_exec_timeout = max(1.0, process_overall_timeout - 5.0)

    # Resource limits in MB (adjust as needed)
    rlimit_as_mb = 2048    # Virtual memory
    rlimit_data_mb = 1024   # Heap size
    rlimit_stack_mb = 64    # Stack size

    # --- 2. Setup Inter-Process Communication (IPC) ---
    manager = None # Initialize manager to None
    try:
        # Use a multiprocessing Manager to create shared objects easily
        manager = multiprocessing.Manager()

        # Create shared objects using the manager's factory methods
        # These return Proxy objects that synchronize access
        status_val_proxy = manager.Value("i", _UNKNOWN) # 'i' for integer
        passed_count_proxy = manager.Value("i", 0)
        failed_count_proxy = manager.Value("i", 0)
        total_count_proxy = manager.Value("i", 0)
        error_details_manager_proxy = manager.dict() # Shared dictionary for errors

        # Create a raw shared memory buffer for prints output (avoids Manager overhead for large data)
        # 'c' for character/byte array
        prints_buffer_shared = multiprocessing.Array('c', MAX_PRINT_OUTPUT_SIZE, lock=False) # Lock=False might improve performance slightly if writes are atomic enough
        prints_length_proxy = manager.Value('i', 0) # Shared int for the length used in the buffer

        # --- 3. Prepare and Start Child Process ---
        process_args = (
            generated_code_str, test_code_str, internal_test_exec_timeout,
            rlimit_as_mb, rlimit_data_mb, rlimit_stack_mb,
            # Pass the *proxy* objects to the child process
            status_val_proxy, passed_count_proxy, failed_count_proxy, total_count_proxy,
            error_details_manager_proxy, prints_buffer_shared, prints_length_proxy
        )

        # Create the child process targeting the unsafe_execute function
        process = multiprocessing.Process(target=unsafe_execute_with_details, args=process_args)
        process.daemon = True # Allow parent to exit even if daemon child hangs (though we try to terminate)

        # Initialize variables to store results retrieved from child
        final_prints_str = ""
        process_status_summary = "unknown_termination" # Debug summary of how process ended
        final_status_code = _UNKNOWN
        process_exit_code_val = None
        p_count, f_count, t_count = 0, 0, 0 # Local copies of counts
        local_error_details = {} # Local copy of error details
        ipc_failed = False # Flag to indicate if communication with child failed

        # --- 4. Manage Child Process Execution and Timeout ---
        start_time = time.monotonic()
        process.start()
        # Wait for the process to complete, with the overall timeout
        process.join(timeout=process_overall_timeout)
        join_duration = time.monotonic() - start_time

        # --- 5. Handle Process Outcome ---
        if process.is_alive():
            # Overall timeout reached, process still running
            process_status_summary = f"main_timeout_terminated_after_{join_duration:.2f}s"
            # print(f"Warning: Process {process.pid} timed out after {process_overall_timeout}s. Terminating.") # Optional debug log
            process.terminate() # Send SIGTERM
            time.sleep(0.2) # Give it a moment to exit gracefully
            if process.is_alive():
                # print(f"Warning: Process {process.pid} did not terminate gracefully. Killing.") # Optional debug log
                process.kill() # Send SIGKILL
                time.sleep(0.1) # Short wait after kill

            final_status_code = _TIMEOUT
            local_error_details["PROCESS_CONTROL"] = (
                f"Main process timed out waiting for child after {process_overall_timeout:.1f}s. "
                "Child process was terminated forcefully."
            )
            # Assume IPC is unreliable after forceful termination
            ipc_failed = True
            final_prints_str = "[Output potentially lost due to main process timeout and termination]"

        else:
            # Process finished on its own (normally or crashed)
            process_exit_code_val = process.exitcode
            process_status_summary = f"finished_exitcode_{process_exit_code_val}_in_{join_duration:.2f}s"

            # --- 6. Retrieve Results via IPC (if process didn't timeout externally) ---
            try:
                # Access shared memory objects via their proxies to get final values
                final_status_code = status_val_proxy.value
                local_error_details.update(dict(error_details_manager_proxy.items())) # Copy shared dict items

                # Check for inconsistencies: Non-zero exit code but status wasn't updated?
                if process_exit_code_val != 0 and final_status_code == _UNKNOWN:
                    # Child likely crashed before reporting status properly
                    final_status_code = _FRAMEWORK_ERROR
                    local_error_details["PROCESS_CONTROL"] = (
                        f"Subprocess exited abnormally (exit code {process_exit_code_val}) "
                        "before reporting detailed status via shared memory. Likely crashed."
                    )
                    ipc_failed = True # Status might be unreliable
                elif final_status_code == _UNKNOWN:
                     # Exit code 0, but status still unknown - implies logic error in child's reporting
                     final_status_code = _FRAMEWORK_ERROR
                     local_error_details["PROCESS_CONTROL"] = (
                        f"Subprocess exited normally (exit code 0) but status remained UNKNOWN. "
                        "Possible logic error in child's status reporting."
                    )

                # Retrieve the captured prints output from the shared buffer
                actual_len = prints_length_proxy.value
                if 0 < actual_len < MAX_PRINT_OUTPUT_SIZE:
                    # Read the exact length reported by the child
                    final_prints_bytes = bytes(prints_buffer_shared[:actual_len])
                    final_prints_str = final_prints_bytes.decode('utf-8', 'replace')
                elif actual_len == 0:
                    # No output reported
                    if final_status_code == _TIMEOUT:
                        # If timeout occurred, use the error message if no prints captured
                         final_prints_str = local_error_details.get("INTERNAL_TIMEOUT", "[No print output captured during timeout]")
                    else:
                        final_prints_str = "" # Otherwise, genuinely no output
                elif actual_len >= MAX_PRINT_OUTPUT_SIZE - 1:
                    # Output was likely truncated
                    final_prints_bytes = bytes(prints_buffer_shared[:MAX_PRINT_OUTPUT_SIZE-1]) # Read up to max size - 1
                    final_prints_str = final_prints_bytes.decode('utf-8', 'replace') + "\n[... TRUNCATED ...]"
                # Note: case actual_len < 0 shouldn't happen but indicates an error if it did.

                # Retrieve test counts
                p_count = passed_count_proxy.value
                f_count = failed_count_proxy.value
                t_count = total_count_proxy.value

            except (BrokenPipeError, EOFError, ConnectionResetError, OSError) as e_ipc:
                # Catch errors indicating the communication channel is broken
                # This often happens if the child process crashed unexpectedly
                ipc_failed = True
                final_status_code = _FRAMEWORK_ERROR # Mark as framework error
                process_status_summary += f"_ipc_error_{type(e_ipc).__name__}"
                local_error_details["IPC_ERROR"] = (
                    f"IPC error accessing shared data after child process ended (exit code {process_exit_code_val}): {type(e_ipc).__name__}: {e_ipc}. "
                    "Child process (or its manager) likely crashed or manager shutdown prematurely. Shared data is unreliable."
                )
                final_prints_str = f"[IPC error prevented reliable retrieval of prints_output. Error: {str(e_ipc)}]"
                # Reset counts as they might be corrupted
                p_count, f_count, t_count = 0, 0, 0

    except Exception as e_proc_manage:
        # Catch errors in the parent process related to managing the child (e.g., starting process)
        final_status_code = _FRAMEWORK_ERROR
        local_error_details["PROCESS_MANAGER_ERROR"] = f"Error managing subprocess: {type(e_proc_manage).__name__}: {str(e_proc_manage)}\n{traceback.format_exc()}"
        process_status_summary = f"manager_exception_{type(e_proc_manage).__name__}"
        ipc_failed = True # Cannot trust any potential IPC setup
        final_prints_str = "[Error during process management prevented execution and output retrieval]"

    finally:
        # --- 7. Cleanup ---
        # Ensure the manager is shut down to release resources, regardless of errors
        if manager:
            try:
                manager.shutdown()
            except Exception as e_shutdown:
                # Log if manager shutdown fails, but don't overwrite primary error
                # print(f"Warning: Exception during multiprocessing manager shutdown: {e_shutdown}")
                 if "IPC_ERROR" not in local_error_details and "PROCESS_MANAGER_ERROR" not in local_error_details:
                     local_error_details["MANAGER_SHUTDOWN_ERROR"] = f"Error shutting down manager: {e_shutdown}"


    # --- 8. Format Final Result ---
    status_str_map = {
        _SUCCESS: "success",
        _FAILED: "failure",
        _TIMEOUT: "timeout",
        _FRAMEWORK_ERROR: "error",
        _UNKNOWN: "error", # Treat unknown final state as error
    }
    final_status_str = status_str_map.get(final_status_code, "error") # Default to error

    # If IPC failed, the result is inherently unreliable unless it was a clean timeout reported by parent
    if ipc_failed and final_status_str != "timeout":
        final_status_str = "error"
        if "IPC_ERROR" not in local_error_details and "PROCESS_MANAGER_ERROR" not in local_error_details:
            # Add a generic IPC error if none was specifically recorded
            local_error_details["IPC_ERROR"] = "An unspecified IPC failure occurred, results are unreliable."

    # Determine the overall 'passed' status
    # Passed requires: status="success", at least one test ran, and no failures/errors.
    passed_bool = (final_status_str == "success" and t_count > 0 and f_count == 0)

    # Handle edge case: status is SUCCESS but no tests ran (e.g., empty TestCases)
    if t_count == 0 and final_status_str == "success":
        passed_bool = False # Cannot pass if no tests ran
        if not local_error_details: # Add explanation if no other error exists
             local_error_details["EXECUTION_INFO"] = "Execution reported success, but no tests were run (total_count is 0). Check TestCases class."
        # If no tests ran, it's arguably an error in the setup/test code, not a true success.
        # Only change status if IPC didn't fail (otherwise keep the IPC error status)
        if not ipc_failed:
            final_status_str = "error"


    # Combine error details into a single result string
    result_string = ""
    if local_error_details:
        error_messages_list = []
        # Sort keys for consistent output order (optional)
        # sorted_keys = sorted(local_error_details.keys())
        # for key in sorted_keys:
        for key, message in local_error_details.items():
            message_str = str(message) # Ensure message is a string
            # Format with key and indented message
            error_messages_list.append(f"[{key}]:\n{message_str.strip()}")
        result_string = "\n\n".join(error_messages_list)

    # If no specific error details were captured, but status indicates failure/error/timeout, provide a generic message.
    if not result_string and final_status_str != "success":
        if final_status_str == "timeout":
            result_string = (f"Execution timed out. "
                               f"(Overall process timeout: {process_overall_timeout}s or "
                               f"internal test execution timeout: {internal_test_exec_timeout}s). "
                               "No specific error details were reported or retrieved.")
        elif final_status_code == _UNKNOWN:
             result_string = (f"Execution resulted in an UNKNOWN final state. "
                              f"Process exit code: {process_exit_code_val}. "
                              f"Child process may have crashed or failed to report status correctly.")
        else: # General failure/error without details
            result_string = f"Execution resulted in status '{final_status_str}' with no specific error details available. Process exit code: {process_exit_code_val}."


    # Return the comprehensive results dictionary
    return {
        'passed': passed_bool,
        'passed_count': p_count,
        'failed_count': f_count,
        'total_count': t_count,
        'result': result_string,          # Formatted error details or generic message
        'prints_output': final_prints_str, # Captured stdout/stderr (potentially truncated)
        'status': final_status_str,        # "success", "failure", "timeout", "error"
        'llm_code_generation_time': code_generation_time, # Pass through generation time
        # --- Debug Information ---
        '_debug_process_status_summary': process_status_summary, # How the process terminated
        '_debug_final_status_code': final_status_code,          # Raw status code (_SUCCESS, etc.)
        '_debug_process_exit_code': process_exit_code_val,      # Exit code of the child process
        '_debug_ipc_failed': ipc_failed                        # Flag indicating IPC issues
    }