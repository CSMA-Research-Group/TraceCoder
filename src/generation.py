import time
import re
import ast
from typing import Tuple, Any, List
import torch
from openai import OpenAI, APIError, Timeout, RateLimitError
from transformers import AutoTokenizer
import logging # Added import

# Import function from a sibling module
from .postprocessing import extract_python_code

# --- OpenAI/Compatible API Client Configuration ---
client = OpenAI(
    api_key="your_api_key_here",
    base_url="your_base_url_here"
)

# Configure logger
logger = logging.getLogger(__name__)
# You can add more detailed logging configuration as needed, for example, setting the log level and format:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Promoted inline helper functions to top-level functions ---

def get_split_point_after_last_return(code_string: str) -> int:
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return len(code_string)
    last_return_end_line_num = -1
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            node_effective_end_line = getattr(node, 'end_lineno', node.lineno)
            if node_effective_end_line is not None and node_effective_end_line > last_return_end_line_num:
                last_return_end_line_num = node_effective_end_line
    if last_return_end_line_num == -1:
        return len(code_string)
    lines = code_string.splitlines(True)
    char_offset = sum(len(lines[i]) for i in range(min(last_return_end_line_num, len(lines))))
    return char_offset


def remove_content_after_last_return(code_string: str) -> str:
    if not code_string.strip(): return ""
    split_point = get_split_point_after_last_return(code_string)
    if split_point >= len(code_string): return code_string.strip()
    code_to_keep = code_string[:split_point]
    code_to_process = code_string[split_point:]
    print_pattern = r'(?m)^\s*print\s*\(.*?\)\s*(?:#.*)?\n?'
    processed_after = re.sub(print_pattern, '', code_to_process)
    comment_pattern = r'(?m)^\s*#.*?\n?'
    processed_after = re.sub(comment_pattern, '', processed_after)
    processed_after = re.sub(r'(\s*\n\s*){2,}', '\n', processed_after).strip()
    final_code = code_to_keep.rstrip('\n')
    if processed_after:
        final_code += '\n' + processed_after
    return final_code.strip()


def extract_python_code_with_logic(text: str, remove_main: bool = True, remove_prints: bool = True,
                                   remove_repl_examples: bool = True) -> str:
    pattern_explicit = r"```python\s*\n(.*?)\n?```"
    pattern_generic = r"```\s*\n(.*?)\n?```"
    explicit_matches = re.findall(pattern_explicit, text, re.DOTALL | re.IGNORECASE)
    extracted_code_blocks_raw = [block.strip() for block in explicit_matches]
    if not extracted_code_blocks_raw:
        generic_matches = re.findall(pattern_generic, text, re.DOTALL)
        for block in generic_matches:
            if any(keyword in block.lower() for keyword in ['def ', 'import ', 'class ']):
                extracted_code_blocks_raw.append(block.strip())
    processed_code_blocks = []
    for raw_block in extracted_code_blocks_raw:
        processed_block = raw_block
        if remove_repl_examples:
            processed_block = re.sub(r'(?m)^\s*(>>>|\.\.\.)\s.*$\n?', '', processed_block)
        if remove_main:
            main_block_pattern = r'(?m)^\s*if\s+__name__\s*==\s*("__main__"|\'__main__\'):.*?(\n\s*\n|\Z)'
            processed_block = re.sub(main_block_pattern, '', processed_block, flags=re.DOTALL)
        if remove_prints:
            processed_block = remove_content_after_last_return(processed_block)
        processed_block = re.sub(r'(?m)^\s*\n', '', processed_block).strip()
        if processed_block:
            processed_code_blocks.append(processed_block)
    return "\n\n".join(processed_code_blocks).strip()


# --- API Call Function ---
def call_api(prompt, model, max_tokens_response=4096):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant specialized in code generation and debugging."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        result_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        time.sleep(1) # Wait for 1 second after each successful call
        return result_text, prompt_tokens, completion_tokens
    except RateLimitError as e:
        logger.warning(f"API rate limit error: {e}. May need to wait longer before retrying.") # logger is now defined
        # A longer wait time or a specific retry strategy can be added here
        return None, 0, 0
    except APIError as e:
        logger.error(f"An error occurred during the API call: {e}") # logger is now defined
        return None, 0, 0
    except Timeout as e:
        logger.error(f"API call timed out: {e}") # logger is now defined
        return None, 0, 0
    except Exception as e: # Catch all other unexpected exceptions
        logger.error(f"An unknown error occurred when calling the API: {e}") # logger is now defined
        return None, 0, 0


# --- Main Generator Function ---
def generator(text: str, status: str, model_name: str, models=None, tokenizers=None) -> Tuple[Any, int, int]:
    """
    Calls different models or APIs to generate content based on the status.
    Returns: (generated content, prompt_tokens, completion_tokens)
    """
    # If local models and tokenizers are provided
    if models and tokenizers:
        # Local model generation logic is preserved here
        messages = [{"role": "user", "content": text}]
        inputs = tokenizers.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            models.device)
        outputs = models.generate(inputs, max_new_tokens=4096, eos_token_id=tokenizers.eos_token_id)

        prompt_tokens = len(inputs[0])
        completion_tokens = len(outputs[0]) - prompt_tokens

        generation = tokenizers.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)

        if status == "code1_generate":
            generation = extract_python_code_with_logic(generation)
        # More status handling logic can be added for local models

        torch.cuda.empty_cache()
        return generation, prompt_tokens, completion_tokens

    # If using an API model
    else:
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            raw_generation, p_tokens, c_tokens = call_api(prompt=text, model=model_name)

            if raw_generation is not None:
                # Post-process the raw output based on the status
                if status in ["code3_generate", "code5_generate"]:
                    processed_generation = extract_python_code(raw_generation)
                    print(f"################# PROCESSED ({status}) #################\n{processed_generation}")
                    return processed_generation, p_tokens, c_tokens

                elif status in ["code4_generate", "code6_generate"]:
                    # These statuses need to return the original text containing analysis and code
                    print(f"################# RAW ({status}) #################\n{raw_generation}")
                    return raw_generation, p_tokens, c_tokens

                else:
                    # Default behavior: return the raw text
                    return raw_generation, p_tokens, c_tokens

            retry_count += 1
            print(f"API call failed, retrying... (Attempt {retry_count}/{max_retries})")
            if retry_count < max_retries:
                time.sleep(5)

        # All retries failed
        error_message = f"Error: API call still failed after {max_retries} retries (status: '{status}')."
        print(error_message)
        return error_message, 0, 0
