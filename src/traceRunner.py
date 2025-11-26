import subprocess
import tempfile
import os
import sys
import time  # Explicitly import the time module
import json
import io  # Required for io.StringIO
import traceback  # Required for formatting exception information
import re  # Used for preprocessing the code string

# --- Helper code to be injected into the user's code ---
HELPER_CODE_TEMPLATE = """
import unittest
import io
import sys
import json
import traceback
import os # Not used directly, but may be useful for more complex test loading scenarios

# --- JSON result markers for parsing by the main process ---
JSON_RESULTS_START_MARKER = "---JSON_RESULTS_START---"
JSON_RESULTS_END_MARKER = "---JSON_RESULTS_END---"

class EnhancedTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_data = [] # Stores detailed information for each test
        self._original_stdout = None
        self._original_stderr = None
        self._current_test_stdout = None
        self._current_test_stderr = None

    def startTest(self, test):
        super().startTest(test)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._current_test_stdout = io.StringIO()
        self._current_test_stderr = io.StringIO()
        sys.stdout = self._current_test_stdout
        sys.stderr = self._current_test_stderr

    def _get_captured_outputs_and_restore_streams(self):
        stdout_val = ""
        stderr_val = ""

        if self._current_test_stdout:
            stdout_val = self._current_test_stdout.getvalue()
            self._current_test_stdout.close()
            self._current_test_stdout = None

        if self._current_test_stderr:
            stderr_val = self._current_test_stderr.getvalue()
            self._current_test_stderr.close()
            self._current_test_stderr = None

        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr:
            sys.stderr = self._original_stderr
            self._original_stderr = None

        return stdout_val, stderr_val

    def addSuccess(self, test):
        super().addSuccess(test)
        stdout_val, stderr_val = self._get_captured_outputs_and_restore_streams()
        self.results_data.append({
            "name": test.id(),
            "status": "PASSED",
            "stdout": stdout_val,
            "stderr": stderr_val,
        })

    def addError(self, test, err):
        super().addError(test, err)
        stdout_val, stderr_val = self._get_captured_outputs_and_restore_streams()
        exc_type, exc_value, exc_tb = err
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        self.results_data.append({
            "name": test.id(),
            "status": "ERROR",
            "stdout": stdout_val,
            "stderr": stderr_val,
            "traceback": "".join(tb_lines)
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        stdout_val, stderr_val = self._get_captured_outputs_and_restore_streams()
        exc_type, exc_value, exc_tb = err
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        self.results_data.append({
            "name": test.id(),
            "status": "FAILED",
            "stdout": stdout_val,
            "stderr": stderr_val,
            "traceback": "".join(tb_lines)
        })

    def stopTest(self, test):
        # Ensure streams are restored if the test exits early without error/failure
        if self._original_stdout is not None or self._original_stderr is not None :
             self._get_captured_outputs_and_restore_streams()
        super().stopTest(test)


def run_tests_with_custom_result():
    runner_stream = io.StringIO()
    loader = unittest.TestLoader()

    try:
        # Load tests from the main module where the user's code is executed
        suite = loader.loadTestsFromModule(sys.modules['__main__'])
    except Exception as e:
        # This error occurs if test loading itself fails
        print(f"Error loading tests: {{str(e)}}", file=sys.stderr) # To actual stderr
        sys.stderr.flush()

        # Ensure we're writing JSON to the original process stdout
        print("\\n" + JSON_RESULTS_START_MARKER)
        json.dump({
            "runner_summary": f"Error: Failed to load test cases: {{str(e)}}. No tests were run.",
            "detailed_results": []
        }, sys.stdout)
        print("\\n" + JSON_RESULTS_END_MARKER)
        sys.stdout.flush()
        return

    runner = unittest.TextTestRunner(stream=runner_stream, verbosity=2, resultclass=EnhancedTestResult)
    result_object = runner.run(suite)

    # Prepare the data to be output as JSON
    output_data = {
        "runner_summary": runner_stream.getvalue(),
        "detailed_results": result_object.results_data
    }

    # Print the JSON data to stdout, wrapped in markers for easy parsing
    sys.stdout.flush()
    print("\\n" + JSON_RESULTS_START_MARKER)
    json.dump(output_data, sys.stdout)
    print("\\n" + JSON_RESULTS_END_MARKER)
    sys.stdout.flush()

if __name__ == '__main__':
    run_tests_with_custom_result()
"""


def _preprocess_code_string_to_deactivate_main(code_string: str) -> str:
    """
    Attempts to disable `unittest.main()` calls in the user's code string by commenting them out,
    preventing them from running before our custom runner.
    This is a "best-effort" approach based on regex.
    """
    lines = code_string.splitlines()
    new_lines = []
    main_call_pattern = re.compile(r"^(\s*)unittest\.main\s*\(.*")
    for line in lines:
        stripped_line = line.lstrip()
        # Preserve comments
        if stripped_line.startswith("#"):
            new_lines.append(line)
            continue

        match = main_call_pattern.match(line)
        if match:
            indent = match.group(1)
            original_call_line_content = line.strip()
            # Comment out the original line and add a note
            new_lines.append(f"{indent}# original call: {original_call_line_content} # Disabled by execution wrapper")
            # Add a pass statement in case the call was the only line in a block (e.g., inside an if)
            new_lines.append(f"{indent}pass # Placeholder in case the original call is the only statement in the block")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def execute_code_and_capture_prints_last(
        code_string: str,
        timeout_seconds: int = 10,
        n_last_lines: int = 50
) -> dict:
    """
    Executes the given Python code string (expected to contain unittest test cases),
    captures all its standard output (prints) and test results,
    and presents the output for passed and failed/errored test cases separately.
    User's `unittest.main()` calls are attempted to be disabled to allow execution
    by the injected custom runner.

    Parameters:
    code_string (str): The Python code to execute (should contain unittest tests and
                       typically an `if __name__=='__main__':` block).
    timeout_seconds (int): Timeout for the execution in seconds.
    n_last_lines (int, optional): If provided, only the last n lines of the combined
                                  display output will be returned. None or <=0 means all lines.

    Returns:
    dict: A dictionary containing the following keys:
        "passed_section" (str): Formatted string containing details of passed test cases.
        "failed_errored_section" (str): Formatted string containing details of failed or errored test cases.
        "other_info_section" (str): Formatted string containing module-level output, Runner summary,
                                     subprocess stderr, and execution errors.
        "display_output" (str): A single string combining all sections, with n_last_lines
                                 truncation applied (if specified).
        "error_occurred" (bool): Indicates if any type of error occurred during execution.
    """
    temp_file_path = None
    error_occurred_flag = False

    # Initialize lists to hold different parts of the output
    module_stdout_pre_json_lines = []
    passed_tests_detail_lines = []
    failed_or_error_tests_detail_lines = []
    runner_summary_lines = []
    subprocess_stderr_lines = []
    module_stdout_post_json_lines = []
    execution_error_log_lines = []

    # Preprocess the user code and append our helper code
    preprocessed_code_string = _preprocess_code_string_to_deactivate_main(code_string)
    full_code_to_execute = preprocessed_code_string + "\n\n" + HELPER_CODE_TEMPLATE

    # Define markers for parsing JSON from the subprocess output
    raw_json_start_marker = "---JSON_RESULTS_START---"
    raw_json_end_marker = "---JSON_RESULTS_END---"
    actual_start_marker_in_stream = "\n" + raw_json_start_marker
    actual_end_marker_in_stream = "\n" + raw_json_end_marker

    try:
        # Create a temporary file to run the code
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding='utf-8') as fp:
            fp.write(full_code_to_execute)
            temp_file_path = fp.name

        python_executable = sys.executable
        command = [python_executable, temp_file_path]

        # Start the subprocess
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        stdout_data_full = ""
        stderr_data_full = ""

        try:
            # Communicate with the process and get output
            stdout_data_full, stderr_data_full = process.communicate(timeout=timeout_seconds)

            json_data = None
            pre_json_stdout_str = ""
            post_json_stdout_str = ""

            # Find the JSON block in the standard output
            start_idx = stdout_data_full.find(actual_start_marker_in_stream)

            if start_idx != -1:
                pre_json_stdout_str = stdout_data_full[:start_idx].strip()
                json_content_start_idx = start_idx + len(actual_start_marker_in_stream)
                end_idx = stdout_data_full.find(actual_end_marker_in_stream, json_content_start_idx)

                if end_idx != -1:
                    json_string_content = stdout_data_full[json_content_start_idx:end_idx].strip()
                    post_json_stdout_str = stdout_data_full[end_idx + len(actual_end_marker_in_stream):].strip()

                    try:
                        if json_string_content:
                            json_data = json.loads(json_string_content)
                        else:
                            json_data = {"runner_summary": "Note: JSON content is empty.", "detailed_results": []}
                            execution_error_log_lines.append(f"--- Warning: Received empty JSON content from subprocess. ---")
                    except json.JSONDecodeError as je:
                        execution_error_log_lines.append(f"--- JSON Parse Error ---\nCould not parse test results from subprocess: {je}")
                        execution_error_log_lines.append(
                            f"Raw content attempted to parse (up to 500 chars): '{json_string_content[:500]}...'")
                        error_occurred_flag = True
                else:
                    execution_error_log_lines.append(f"--- Warning: Found JSON start marker but no end marker ---")
                    unparsed_after_start_marker = stdout_data_full[start_idx:]
                    if unparsed_after_start_marker.strip():
                        execution_error_log_lines.append(
                            "--- Raw standard output (found JSON start marker but no end marker, content follows) ---")
                        execution_error_log_lines.extend(unparsed_after_start_marker.strip().splitlines())
                    error_occurred_flag = True
            else:
                # No JSON block found, treat all stdout as pre-JSON output
                pre_json_stdout_str = stdout_data_full.strip()

            if pre_json_stdout_str:
                module_stdout_pre_json_lines.append("--- Module-level Standard Output (before test results JSON) ---")
                module_stdout_pre_json_lines.extend(pre_json_stdout_str.splitlines())

            if json_data:
                current_passed_details = []
                current_failed_errored_details = []

                for result in json_data.get("detailed_results", []):
                    test_name = result.get("name", "Unknown Test")
                    status = result.get("status", "Unknown Status").upper()
                    test_stdout = result.get("stdout", "").strip()
                    test_stderr = result.get("stderr", "").strip()
                    traceback_info = result.get("traceback", "").strip()

                    output_for_test = [f"Test: {test_name} - Status: {status}"]
                    if test_stdout:
                        output_for_test.append("  --- Test Standard Output ---")
                        output_for_test.extend([f"    {line}" for line in test_stdout.splitlines()])
                    if test_stderr:
                        output_for_test.append("  --- Test Standard Error (from within test) ---")
                        output_for_test.extend([f"    {line}" for line in test_stderr.splitlines()])
                    if traceback_info:
                        output_for_test.append("  --- Traceback Information ---")
                        output_for_test.extend([f"    {line}" for line in traceback_info.splitlines()])

                    if status == "PASSED":
                        current_passed_details.extend(output_for_test)
                        if test_stdout or test_stderr or traceback_info:
                             current_passed_details.append("-" * 20)
                    else:
                        current_failed_errored_details.extend(output_for_test)
                        current_failed_errored_details.append("-" * 20)

                if current_passed_details:
                    if current_passed_details[-1] == "-" * 20: current_passed_details.pop()
                    passed_tests_detail_lines.extend(current_passed_details)

                if current_failed_errored_details:
                    if current_failed_errored_details[-1] == "-" * 20: current_failed_errored_details.pop()
                    failed_or_error_tests_detail_lines.extend(current_failed_errored_details)

                runner_summary_str = json_data.get("runner_summary", "").strip()
                if runner_summary_str:
                    runner_summary_lines.append("\n--- Unittest Runner Summary ---")
                    runner_summary_lines.extend(runner_summary_str.splitlines())

            if stderr_data_full and stderr_data_full.strip():
                # Avoid printing stderr if it's just a copy of the runner summary (which can happen)
                is_stderr_exact_summary = json_data and stderr_data_full.strip() == json_data.get("runner_summary", "").strip()
                if not is_stderr_exact_summary:
                    subprocess_stderr_lines.append(f"\n--- Subprocess Standard Error (exit code: {process.returncode}) ---")
                    subprocess_stderr_lines.extend(stderr_data_full.strip().splitlines())

            if post_json_stdout_str:
                module_stdout_post_json_lines.append("\n--- Module-level Standard Output (after test results JSON) ---")
                module_stdout_post_json_lines.extend(post_json_stdout_str.splitlines())

            if process.returncode != 0:
                error_occurred_flag = True
                has_failures_in_json = json_data and any(
                    r["status"] != "PASSED" for r in json_data.get("detailed_results", [])
                )
                # Check if the runner summary already indicates a test loading error
                is_loader_error_in_summary = json_data and "Error: Failed to load test cases" in json_data.get("runner_summary", "")

                # Log a generic warning if the exit code is non-zero without other obvious errors
                if not (stderr_data_full.strip() or has_failures_in_json or is_loader_error_in_summary):
                    execution_error_log_lines.append(
                        f"--- Execution Warning: Process ended with non-zero exit code ({process.returncode}) but no apparent error output or JSON failures recorded ---")

        except subprocess.TimeoutExpired:
            error_occurred_flag = True
            process.kill()
            stdout_partial, stderr_partial = ("", "")
            try:
                # Try to get any remaining output
                stdout_partial, stderr_partial = process.communicate(timeout=1)
            except Exception:
                pass # Ignore errors during cleanup

            # Log all partial output that was captured
            if stdout_data_full and stdout_data_full.strip():
                execution_error_log_lines.append("--- Partial Standard Output before timeout (may be incomplete) ---")
                execution_error_log_lines.extend(stdout_data_full.strip().splitlines())
            if stdout_partial and stdout_partial.strip():
                execution_error_log_lines.append("--- Additional Standard Output attempted after timeout ---")
                execution_error_log_lines.extend(stdout_partial.strip().splitlines())
            if stderr_data_full and stderr_data_full.strip():
                execution_error_log_lines.append("--- Partial Standard Error before timeout (may be incomplete) ---")
                execution_error_log_lines.extend(stderr_data_full.strip().splitlines())
            if stderr_partial and stderr_partial.strip():
                execution_error_log_lines.append("--- Additional Standard Error attempted after timeout ---")
                execution_error_log_lines.extend(stderr_partial.strip().splitlines())
            execution_error_log_lines.append(f"--- Execution Timed Out ({timeout_seconds}s) ---")

        except Exception as e:
            error_occurred_flag = True
            execution_error_log_lines.append(
                f"--- Error within main execution function ---\n{type(e).__name__}: {str(e)}\nDetailed Traceback:\n{traceback.format_exc()}")
            # Log any captured output during the error
            if stdout_data_full and stdout_data_full.strip():
                execution_error_log_lines.append("--- Captured Raw Standard Output (during error handling) ---")
                execution_error_log_lines.extend(stdout_data_full.strip().splitlines())
            if stderr_data_full and stderr_data_full.strip():
                execution_error_log_lines.append("--- Captured Raw Standard Error (during error handling) ---")
                execution_error_log_lines.extend(stderr_data_full.strip().splitlines())

    except IOError as e:
        error_occurred_flag = True
        execution_error_log_lines.append(f"--- File Operation Error ---\n{type(e).__name__}: {str(e)}")
    except Exception as e:
        error_occurred_flag = True
        execution_error_log_lines.append(
            f"--- Unexpected error in execution wrapper ---\n{type(e).__name__}: {str(e)}\nDetailed Traceback:\n{traceback.format_exc()}")
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                # Write to actual stderr since this is a cleanup issue
                sys.stderr.write(f"Warning: Failed to delete temporary file {temp_file_path}: {e}\n")

    # --- Assemble the output sections ---
    passed_section_str = ""
    if passed_tests_detail_lines:
        passed_section_str = "\n--- Passed Test Cases ---\n" + "\n".join(passed_tests_detail_lines)

    failed_errored_section_str = ""
    if failed_or_error_tests_detail_lines:
        failed_errored_section_str = "\n--- Failed or Errored Test Cases ---\n" + "\n".join(failed_or_error_tests_detail_lines)

    other_info_parts = []
    if module_stdout_pre_json_lines: other_info_parts.extend(module_stdout_pre_json_lines)
    if runner_summary_lines: other_info_parts.extend(runner_summary_lines)
    if subprocess_stderr_lines: other_info_parts.extend(subprocess_stderr_lines)
    if module_stdout_post_json_lines: other_info_parts.extend(module_stdout_post_json_lines)
    if execution_error_log_lines: other_info_parts.extend(execution_error_log_lines)
    other_info_section_str = "\n".join(other_info_parts).strip()

    # --- Assemble the combined display_output (for n_last_lines truncation) ---
    display_output_lines = []
    if module_stdout_pre_json_lines: display_output_lines.extend(module_stdout_pre_json_lines)
    if passed_tests_detail_lines:
        display_output_lines.append("\n--- Passed Test Cases ---")
        display_output_lines.extend(passed_tests_detail_lines)
    if failed_or_error_tests_detail_lines:
        display_output_lines.append("\n--- Failed or Errored Test Cases ---")
        display_output_lines.extend(failed_or_error_tests_detail_lines)
    if runner_summary_lines: display_output_lines.extend(runner_summary_lines)
    if subprocess_stderr_lines: display_output_lines.extend(subprocess_stderr_lines)
    if module_stdout_post_json_lines: display_output_lines.extend(module_stdout_post_json_lines)
    if execution_error_log_lines: display_output_lines.extend(execution_error_log_lines)

    # Clean up trailing separators and blank lines
    temp_combined_lines = [line for line in display_output_lines if line is not None]
    while temp_combined_lines and temp_combined_lines[-1].strip() == "-" * 20:
        temp_combined_lines.pop()
    while temp_combined_lines and not temp_combined_lines[-1].strip():
        temp_combined_lines.pop()

    # Apply n_last_lines truncation if needed
    final_display_lines_for_truncation = temp_combined_lines
    if n_last_lines is not None and n_last_lines > 0 and len(final_display_lines_for_truncation) > n_last_lines:
        num_omitted = len(final_display_lines_for_truncation) - n_last_lines
        final_display_lines_for_truncation = [
             f"... ({num_omitted} lines of output omitted for brevity) ..."] + final_display_lines_for_truncation[-n_last_lines:]

    display_output_str = "\n".join(final_display_lines_for_truncation).strip()

    # Provide a default message if no output was generated
    if not display_output_str and not error_occurred_flag:
        display_output_str = "Code executed successfully, but there was no test output or standard output."
    elif not display_output_str and error_occurred_flag:
        display_output_str = "An error occurred during execution, but no specific output was captured."
        if not other_info_section_str:
            other_info_section_str = display_output_str

    return {
        "passed_section": passed_section_str.strip(),
        "failed_errored_section": failed_errored_section_str.strip(),
        "other_info_section": other_info_section_str.strip(),
        "display_output": display_output_str,
        "error_occurred": error_occurred_flag
    }
