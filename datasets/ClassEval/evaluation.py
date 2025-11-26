import unittest
import io
import sys
import traceback
import datetime
import logging
# import concurrent.futures # No longer primary, but TimeoutError might be caught
import pebble # Import pebble
from pebble import ProcessExpired # For pebble's specific timeout/expired exception

# Helper function to recursively get all individual test case names from a suite
# This function must be at the top level to be picklable by multiprocessing
def get_all_test_ids(suite_or_test):
    """Recursively collects test IDs from a test suite or single test case."""
    test_ids = set()
    try:
        for item in suite_or_test:
            test_ids.update(get_all_test_ids(item))
    except TypeError:
        try:
            test_ids.add(suite_or_test.id())
        except AttributeError:
            pass
    return test_ids


# Worker function to run tests in a separate process
# This function must be at the top level to be picklable by multiprocessing
def _execute_tests_in_worker(full_code_to_run):
    """
    Executes the test code in a sandboxed manner.
    This function is intended to be run in a separate process.
    """
    worker_results = {
        'total_tests_in_suite': 0,
        'all_test_ids_in_suite': set(),
        'test_result_summary': {
            'was_successful': False,
            'failures': [],
            'errors': [],
        },
        'stderr_output_worker': '',
        'worker_execution_error': None
    }

    old_stdout_worker = sys.stdout
    old_stderr_worker = sys.stderr
    redirected_stdout_worker = io.StringIO()
    redirected_stderr_worker = io.StringIO()
    sys.stdout = redirected_stdout_worker
    sys.stderr = redirected_stderr_worker

    try:
        exec_namespace = {
            'unittest': unittest,
            'datetime': datetime,
            'logging': logging,
        }
        exec(full_code_to_run, exec_namespace, exec_namespace)

        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        test_case_classes = []

        for name, obj in exec_namespace.items():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                test_case_classes.append(obj)

        if not test_case_classes:
            err_msg = "No unittest.TestCase classes were found or defined in the executed code."
            current_stderr_val = redirected_stderr_worker.getvalue()
            if current_stderr_val:
                err_msg += f"\nStderr content at time of check:\n{current_stderr_val}"
            worker_results['worker_execution_error'] = err_msg
        else:
            for test_case_class in test_case_classes:
                try:
                    tests = loader.loadTestsFromTestCase(test_case_class)
                    suite.addTests(tests)
                except Exception as e:
                    tb_str = traceback.format_exc()
                    load_error_msg = f"Error loading tests from {test_case_class.__name__}: {e}\n{tb_str}"
                    if worker_results['worker_execution_error']:
                        worker_results['worker_execution_error'] += f"\n{load_error_msg}"
                    else:
                        worker_results['worker_execution_error'] = load_error_msg

            worker_results['total_tests_in_suite'] = suite.countTestCases()

            if worker_results['total_tests_in_suite'] == 0:
                if not worker_results['worker_execution_error']:
                    worker_results[
                        'worker_execution_error'] = "Test suite was created but contained no tests (or all load attempts failed)."
            else:
                worker_results['all_test_ids_in_suite'] = get_all_test_ids(suite)
                # Use a simpler runner, as TextTestRunner can be verbose even with verbosity=0
                # and can write directly to the original stderr if not careful.
                # Here, we are redirecting, so it's fine.
                runner = unittest.TextTestRunner(stream=redirected_stdout_worker, verbosity=0, failfast=False, resultclass=unittest.TestResult)
                test_run_result_obj = runner.run(suite)


                worker_results['test_result_summary']['was_successful'] = test_run_result_obj.wasSuccessful()
                for test, err_traceback_str in test_run_result_obj.failures:
                    worker_results['test_result_summary']['failures'].append({
                        'name': test.id(), 'type': 'Failure', 'details': err_traceback_str
                    })
                for test, err_traceback_str in test_run_result_obj.errors:
                    worker_results['test_result_summary']['errors'].append({
                        'name': test.id(), 'type': 'Error', 'details': err_traceback_str
                    })

    except Exception: # Catches errors within the worker's try block
        error_type, error_value, tb = sys.exc_info()
        error_details = "".join(traceback.format_exception(error_type, error_value, tb))
        worker_results[
            'worker_execution_error'] = f"An unhandled exception occurred in the test execution worker:\n{error_details}"
    finally:
        sys.stdout = old_stdout_worker
        sys.stderr = old_stderr_worker
        worker_results['stderr_output_worker'] = redirected_stderr_worker.getvalue()
        # worker_results['stdout_output_worker'] = redirected_stdout_worker.getvalue() # If needed

    return worker_results


def check_correctness(dataset, generated_code, timeout_seconds=10):
    """
    Evaluates the generated Python code against the unittest cases in the dataset,
    providing detailed results, with a timeout for execution using pebble for robust termination.
    ... (rest of docstring same)
    """
    result_details = {
        'passed': False,
        'passed_count': 0,
        'failed_count': 0,
        'total_count': 0,
        'passed_tests': [],
        'failed_tests': [],
        'stderr_output': '',
        'result': None  # Renamed from 'execution_error' to 'result' for general status/error message
    }

    test_code = dataset.get('test')
    if not test_code:
        result_details['result'] = 'Dataset does not contain test code ("test" key is missing or empty).'
        return result_details

    full_code_to_run = generated_code + "\n\n" + test_code
    # print(f"full_code_to_run: \n{full_code_to_run}") # Keep for debugging if needed

    # Use pebble.ProcessPool for robust timeouts
    # max_workers=1 as we run one isolated task.
    with pebble.ProcessPool(max_workers=1, max_tasks=1) as executor: # max_tasks=1 can help with process recycling
        try:
            # Use 'schedule' instead of 'submit' for pebble
            future = executor.schedule(_execute_tests_in_worker, args=[full_code_to_run], timeout=timeout_seconds)
            worker_output = future.result() # No timeout here, as pebble handles it in schedule

            result_details['stderr_output'] = worker_output.get('stderr_output_worker', '')
            result_details['total_count'] = worker_output.get('total_tests_in_suite', 0)

            if worker_output.get('worker_execution_error'):
                result_details['result'] = worker_output['worker_execution_error']
                result_details['passed'] = False

                if result_details['total_count'] > 0:
                    result_details['failed_count'] = result_details['total_count']
                    result_details['passed_count'] = 0
                    all_discovered_ids = worker_output.get('all_test_ids_in_suite', set())
                    result_details['failed_tests'] = [
                        {'name': test_id, 'type': 'Error',
                         'details': 'Test did not run to completion due to a worker execution error.'}
                        for test_id in sorted(list(all_discovered_ids))
                    ]
                else:
                    result_details['failed_count'] = 0
                    result_details['passed_count'] = 0
                result_details['passed_tests'] = []
                return result_details

            if result_details['total_count'] == 0:
                result_details['result'] = worker_output.get('worker_execution_error',
                                                                      "Test suite ran but contained no tests.")
                result_details['passed'] = False
                return result_details

            test_summary = worker_output['test_result_summary']
            result_details['passed'] = test_summary.get('was_successful', False)

            failures = test_summary.get('failures', [])
            errors = test_summary.get('errors', [])
            all_failed_or_errored_tests = failures + errors

            result_details['failed_tests'] = all_failed_or_errored_tests
            result_details['failed_count'] = len(all_failed_or_errored_tests)
            result_details['passed_count'] = result_details['total_count'] - result_details['failed_count']

            failed_test_ids = {ft['name'] for ft in all_failed_or_errored_tests}
            all_discovered_test_ids = worker_output.get('all_test_ids_in_suite', set())
            # all_test_ids_in_suite is already a set from get_all_test_ids
            result_details['passed_tests'] = sorted(list(all_discovered_test_ids - failed_test_ids))

        except (ProcessExpired, TimeoutError) as e: # Catch pebble's ProcessExpired and generic TimeoutError
            # ProcessExpired is more specific for pebble timeouts (task took too long and was killed)
            # TimeoutError can be raised if scheduling itself times out, or if pebble re-raises it.
            result_details['passed'] = False
            result_details['result'] = f"Code execution and testing exceeded timeout of {timeout_seconds} seconds and was terminated. Error: {type(e).__name__} - {e}"
            result_details['total_count'] = 0 # Or could try to get from a partially filled worker_output if available, but safer to reset
            result_details['passed_count'] = 0
            result_details['failed_count'] = 0
            result_details['passed_tests'] = []
            result_details['failed_tests'] = []

        except Exception as e: # Catch other errors from ProcessPool or future.result()
            error_type, error_value, tb = sys.exc_info()
            error_details = "".join(traceback.format_exception(error_type, error_value, tb))
            result_details['passed'] = False
            result_details['result'] = f"An error occurred in the evaluation supervisor or during task submission/retrieval:\n{error_details}"
            result_details['total_count'] = 0
            result_details['passed_count'] = 0
            result_details['failed_count'] = 0
            result_details['passed_tests'] = []
            result_details['failed_tests'] = []
    return result_details
