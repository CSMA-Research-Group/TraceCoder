import re
import time
from itertools import islice
import importlib 
from typing import Callable 

# Import functions from sibling modules
from config import DATASET_PATHS 
from reporting import format_check_correctness_result
from src.generation import generator 
from src.traceRunner import execute_code_and_capture_prints_last  # <<< New import
from src.postprocessing import remove_main_block  # <<< New import for _parse_llm_response


# --- Helper Functions ---
def _parse_llm_output(llm_response: str):
    """ 
    Parses the LLM's response string to extract the repair plan and 
    instrumentation suggestions. 
    The function looks for specific start and end markers to delimit the sections.
    It also cleans up common artifacts like surrounding whitespace and code 
    fences (e.g., ```python ... ```). 
    Args: 
        llm_response: The full string output from the language model. 
    Returns: 
        A tuple containing the extracted 'repair_plan' and 
        'instrumentation_suggestions'. If a section is not found, 
        its value will be None. 
    """ 
    def extract_and_clean_section(text: str, start_marker: str, end_marker: str) -> str | None: 
        """Helper function to extract content between two markers.""" 
        # Use re.DOTALL to make '.' match newlines, which is crucial for multiline content. 
        # Use a non-greedy match '.*?' to find the shortest possible block. 
        pattern = re.compile(f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", re.DOTALL) 
        match = pattern.search(text) 
        if not match: 
            return None 
        # Extract the content from the first capturing group 
        content = match.group(1) 
        # Clean the extracted content: 
        # 1. Remove leading/trailing whitespace and newlines. 
        content = content.strip() 
        # 2. Remove common code fences if they exist. 
        if content.startswith("```python"): 
            content = content[len("```python"):].strip() 
        if content.startswith("```"): 
            content = content[len("```"):].strip() 
        if content.endswith("```"): 
            content = content[:-len("```")].strip() 
        return content 
 
    # Define the markers for each section 
    repair_plan_start = "REPAIR_PLAN_START" 
    repair_plan_end = "REPAIR_PLAN_END" 
    suggestions_start = "Instrumentation_Suggestions_START" 
    suggestions_end = "Instrumentation_Suggestions_END" 
 
    # Extract each section using the helper function 
    repair_plan = extract_and_clean_section(llm_response, repair_plan_start, repair_plan_end) 
    instrumentation_suggestions = extract_and_clean_section(llm_response, suggestions_start, suggestions_end) 
 
    return repair_plan, instrumentation_suggestions



def _build_instrumentation_prompt(code_to_debug_next: str, instrumentation_suggestions: str, failure_info_for_instrumentation: str) -> str:
    """Builds the prompt for code instrumentation."""
    return f"""The Python code below failed tests. Add `print()` statements to trace its execution and variable states, focusing on areas related to the 'Test Failure Feedback'. 
 
 Key Instrumentation Rules: 
 
 1.  Identify several logical blocks/steps in the code. 
 
 2.  For each block/step, print: 
 
 *   Key inputs. 
 
 *   Key outputs/results. 
 
 *   Entry/exit of major functions/loops if helpful. 
 
 3.  ONLY add print statements(Don't comment it out). DO NOT change the original code logic or fix errors. 
 
 4.  Prints should be informative, like: `print(f"[BlockName] Input: {{var}}")`. 
 
 Code to Instrument: 
 
 ```python 
 
 {code_to_debug_next} 
 
 ``` 
 
 Instrument suggestions: 
 
 ``` 
 
 {instrumentation_suggestions} 
 
 ``` 
 
 Feedback: 
 
 ``` 
 
 {failure_info_for_instrumentation} 
 
 ``` 
 
 Please provide ONLY the instrumented Python code. 
 
 """


def _get_failed_history_str(history: list) -> str:
    """Formats historical failure records."""
    if not history:
        return "There have been no previous attempts to fix the failure for this specific problem instance."

    formatted_failed_list = []
    for i, attempt_info in enumerate(islice(reversed(history), 2)):
        plan_summary = attempt_info.get('plan', "N/A")
        eval_result = attempt_info['eval_result']
        formatted_failed_list.append(
            f"Historical Failed Fix Attempt {i + 1} (Passed {eval_result.get('passed_count', 0)}/{eval_result.get('total_count', 1)} tests):\n"
            f"Previous Analysis & Plan (which FAILED to lead to a full fix):\n'''\n{plan_summary}\n'''\n"
            f"That attempt resulted in these errors (on the generated code):\n{format_check_correctness_result(eval_result)}\n---"
        )
    return "\n".join(formatted_failed_list)


def _build_analysis_planning_prompt(problem_data, code_with_prints, captured_prints_output, history) -> str:
    """Builds the prompt for the first step (analysis and planning) in the two-step method."""
    failed_attempts_feedback_str = _get_failed_history_str(history)
    return f"""You are a debugging assistant. 
 
 Given the following information about a Python function that failed tests: 
 
 1. Original Problem Description: 
 
 ```python 
 
 {problem_data['complete_prompt']} 
 
 ``` 
 
 2. Instrumented Code (this version produced the prints and test feedback below): 
 
 ```python 
 
 {code_with_prints} 
 
 ``` 
 
 3. Captured Print Output and Execution Error Messages (from executing the Instrumented Code with test cases): 
 
 ```text 
 
 {str(captured_prints_output)} 
 
 ``` 
 
 4. MISTAKES TO AVOID (Review these previous failed attempts for this problem instance. Identify why they failed and how your current analysis suggests a different, better approach.): 
 
 ``` 
 
 {failed_attempts_feedback_str} 
 
 ``` 
 
 Your tasks are: 
 
 Step 1. Analyze the `Captured Print Output` in conjunction with the `Instrumented Code` line-by-line to pinpoint the exact location and nature of the bug(s). 
 
 Step 2. For each bug in `Execution Error Messages`, explain its root cause. 
 
 Step 3. Review the `MISTAKES TO AVOID`. Analyze the reasons for each failed attempt, explicitly state what lessons are learned and how your new plan avoids repeating those errors. 
 
 Step 4. Propose a concise, step-by-step, actionable plan to fix the bug(s) in the *original code structure*. 
 
 Output your response as clear text, covering points A, B, C, and D. 
 
 Your response MUST be in the specified format: 
 
 REPAIR_PLAN_START 
 
 ```python 
 
 [give the repair plan] 
 
 ``` 
 
 REPAIR_PLAN_END 
 
 Instrumentation_Suggestions_START 
 
 ```python 
 
 [Do not give the code] 
 
 ``` 
 
 Instrumentation_Suggestions_END 
 
 """


def _build_code_implementation_prompt(problem_data, code_to_fix, captured_prints, repair_plan) -> str:
    """Builds the prompt for the second step (code implementation) in the two-step method."""
    return f"""You are a code generation assistant.
You need to fix the following Python code based on the provided repair plan.

Original Problem Description:
```python
{problem_data['complete_prompt']}
```
Faulty Code to Fix:
```python
{code_to_fix}
```
Test Failure Feedback:
```
{captured_prints}
```
Detailed Repair Plan (Follow this plan carefully!):
```text
{repair_plan}
```
Your response MUST be in the specified format:
Repaired CODE Part:
```python
[The part of the code you modified/added.]
```
CODE:
```python
[Your complete corrected Python code here.]
```
END_CODE
"""


def _parse_implementation_output(response_text: str) -> str:
    """
    Parses the LLM's response from the implementation prompt to extract the full code.
    This is a simplified parser for the "CODE:" block.
    """
    # Look for the CODE: block with ```python
    full_code_match = re.search(r"CODE:\s*```python\n(.*?)\n```", response_text, re.DOTALL)
    if full_code_match:
        return full_code_match.group(1).strip()
    
    # Fallback for ``` without language tag
    full_code_match = re.search(r"CODE:\s*```\n(.*?)\n```", response_text, re.DOTALL)
    if full_code_match:
        return full_code_match.group(1).strip()

    # Fallback for cases where the code block is not perfectly formatted but follows CODE:
    code_marker_match = re.search(r"CODE:(.*)", response_text, re.DOTALL | re.IGNORECASE)
    if code_marker_match:
        potential_code = code_marker_match.group(1).strip()
        # Clean up potential markdown fences that might still be there
        if potential_code.startswith("```python"):
            potential_code = potential_code[len("```python"):].strip()
        if potential_code.startswith("```"):
            potential_code = potential_code[len("```"):].strip()
        if potential_code.endswith("```"):
            potential_code = potential_code[:-len("```")].strip()
        return potential_code

    return "" # Return empty string if no code is found


def _load_check_correctness_func(dataset_name: str) -> Callable:
    """
    Dynamically loads and returns the corresponding check_correctness function based on the dataset name.
    """
    try:
        module_path = DATASET_PATHS[dataset_name]["eval_module"]
        eval_module = importlib.import_module(module_path)
        check_correctness_func = getattr(eval_module, "check_correctness")
        return check_correctness_func
    except KeyError:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_PATHS in config.py.")
    except AttributeError:
        raise ImportError(f"'check_correctness' function not found in module '{module_path}'.")
    except ImportError as e:
        raise ImportError(f"Could not import evaluation module '{module_path}': {e}. Ensure __init__.py files exist in package directories.")


# --- Core Processing Functions ---
def _run_direct_generation(problem_data, args, check_correctness_func_param: Callable): # New parameter
    """Executes the first phase: direct code generation and evaluation."""
    prompt = problem_data.get('complete_prompt', '')
    generated_code, p_tokens, c_tokens = generator(prompt, "code3_generate", args.model)
    # Use the passed-in function
    eval_result = check_correctness_func_param(problem_data, generated_code, args.timeout)
    return {"code": generated_code, "eval_result": eval_result, "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens}


def _run_self_debugging(initial_code, initial_eval, problem_data, args, check_correctness_func_param: Callable):
    """Executes the second phase: self-debugging loop (updated to include complete logic)."""
    code_to_debug, last_eval, best_code, best_eval = initial_code, initial_eval, initial_code, initial_eval
    log, history, p_tokens, c_tokens, streak = [], [], 0, 0, 0
    instrumentation_suggestions = ""  # Initialize suggestions

    for attempt in range(1, args.max_debug_attempts + 1):
        print(
            f"--- Debug attempt: {attempt}/{args.max_debug_attempts} | Best: {best_eval.get('passed_count', 0)}/{best_eval.get('total_count', 1)} ---")

        # 1. Instrumentation
        captured_prints = "Instrumentation skipped or not applicable."
        instrumented_code = code_to_debug  # Default to original if instrumentation fails
        if not args.no_instrumentation:
            failure_info = format_check_correctness_result(last_eval)
            instr_prompt = _build_instrumentation_prompt(code_to_debug, instrumentation_suggestions, failure_info)
            instrumented_code_gen, p, c = generator(instr_prompt, "code3_generate", args.model)
            p_tokens, c_tokens = p_tokens + p, c_tokens + c

            if instrumented_code_gen.strip():
                instrumented_code = instrumented_code_gen
                exec_result = execute_code_and_capture_prints_last(
                    instrumented_code + "\n\n" + problem_data.get('test', ''), timeout_seconds=args.timeout)
                captured_prints = exec_result.get('display_output', 'No instrumentation output captured.')

        # 2. Analysis and Planning (This is now the unified first step)
        analysis_prompt = _build_analysis_planning_prompt(problem_data, instrumented_code, captured_prints, history)
        llm_response, p, c = generator(analysis_prompt, "code4_generate", args.model)
        p_tokens, c_tokens = p_tokens + p, c_tokens + c
        
        repair_plan, instrumentation_suggestions = _parse_llm_output(llm_response)
        candidate_code = ""

        # 3. Code Implementation (This is the unified second step, skipped if no plan)
        if repair_plan and repair_plan.strip():
            # The two-step repair is now the default if a plan is returned.
            # The `no_two_step_repair` flag is implicitly handled by whether a plan is generated.
            impl_prompt = _build_code_implementation_prompt(problem_data, code_to_debug, captured_prints, repair_plan)
            llm_response_impl, p_impl, c_impl = generator(impl_prompt, "code4_generate", args.model)
            p_tokens, c_tokens = p_tokens + p_impl, c_tokens + c_impl
            
            candidate_code = _parse_implementation_output(llm_response_impl)
        else:
            # If there's no repair plan, we can't proceed with implementation.
            # This also handles the "one-step" case where the analysis prompt might not return a plan.
            print("LLM did not provide a repair plan, terminating debugging.")
            break

        # 3. Evaluation and decision-making
        if not candidate_code.strip():
            print("LLM did not provide valid code, terminating debugging.")
            break

        eval_candidate = check_correctness_func_param(problem_data, candidate_code, args.timeout) # Use the passed-in function
        print(f"Candidate repair result: {format_check_correctness_result(eval_candidate)}")

        attempt_log = {"attempt": attempt, "plan": repair_plan, "eval_result": eval_candidate}
        log.append(attempt_log)

        if eval_candidate.get('passed'):
            best_code, best_eval = candidate_code, eval_candidate
            break

        if eval_candidate.get('passed_count', 0) > best_eval.get('passed_count', 0):
            best_code, best_eval, code_to_debug, last_eval, streak = candidate_code, eval_candidate, candidate_code, eval_candidate, 0
        else:
            streak += 1

        history.append(attempt_log)
        if streak >= args.max_no_improvement_streak:
            print(f"{streak} consecutive attempts with no improvement, terminating debugging.")
            break

    return {"final_code": best_code, "final_eval": best_eval, "session_log": log, "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens}


# The `process_problem` function remains unchanged as it only calls the core functions above
def process_problem(problem_data, task_id, args, check_correctness_func_param: Callable): # New parameter
    # (This function does not need modification; it is copied from the previous version)
    print(f"\n{'=' * 25} Processing problem: {task_id} {'=' * 25}")
    start_time = time.time()

    gen_res = _run_direct_generation(problem_data, args, check_correctness_func_param) # Pass parameter

    result = {
        "task_id": task_id, "model": args.model, "dataset": args.dataset,
        "method_instrumentation": not args.no_instrumentation,
        "method_two_step_repair": not args.no_two_step_repair,
        "direct_gen_code": gen_res["code"], "direct_gen_eval": gen_res["eval_result"],
        "direct_gen_passed": gen_res["eval_result"].get("passed", False),
    }

    total_p, total_c = gen_res["prompt_tokens"], gen_res["completion_tokens"]

    if result["direct_gen_passed"]:
        print("Direct generation passed successfully.")
        result.update({"final_code": gen_res["code"], "final_eval": gen_res["eval_result"], "final_passed": True,
                       "stopped_reason": "Passed on direct generation"})
    else:
        print("Direct generation failed, entering self-debugging phase.")
        debug_res = _run_self_debugging(gen_res["code"], gen_res["eval_result"], problem_data, args, check_correctness_func_param) # Pass parameter
        result.update({
            "final_code": debug_res["final_code"], "final_eval": debug_res["final_eval"],
            "final_passed": debug_res["final_eval"].get("passed", False),
            "debug_session_log": debug_res["session_log"], "stopped_reason": "Debug cycle completed"
        })
        total_p += debug_res["prompt_tokens"]
        total_c += debug_res["completion_tokens"]

    result.update({"total_prompt_tokens": total_p, "total_completion_tokens": total_c,
                   "processing_time_seconds": time.time() - start_time})

    print(f"Problem {task_id} processing completed. Final result: {'Passed' if result['final_passed'] else 'Failed'}")
    return result
