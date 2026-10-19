"""Prompt bodies derived verbatim from the official main branch.
Source: CSMA-Research-Group/TraceCoder @ cbd536eb5a02b53ee835cae2f07f5b7ed8bb694d.
Only history formatting is replaced: ALL failed candidates include plan, source,
feedback and score, per paper Algorithm 1 / HLLM. See provenance/UPSTREAM.json.
"""
import json
import re


def _get_failed_history_str(history):
    if not history:
        return "There have been no previous attempts to fix the failure for this specific problem instance."
    return "Every failed candidate, in chronological order (no history truncation):\n" + json.dumps(history, ensure_ascii=False, indent=2)


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
