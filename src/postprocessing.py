import re
from typing import List

# Note: The 'os' and 'csv' imports from the original file were unused, 
# but they can be kept or removed for consistency.

def remove_code_block_markers(code: str) -> str:
    """
    Removes potential residual ```python and ``` markers from a code string.
    """
    code = re.sub(r'^\s*```(python|py|python3)?\s*', '', code, flags=re.IGNORECASE)
    code = re.sub(r'\s*```\s*$', '', code, flags=re.IGNORECASE)
    return code.strip()

def remove_main_block(code_string: str) -> str:
    """
    Removes the `if __name__ == "__main__":` block and its content from a Python code string.
    """
    main_block_match = re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]:", code_string, re.MULTILINE)
    if main_block_match:
        return code_string[:main_block_match.start()].rstrip()
    return code_string

def remove_test_section(input_str: str) -> str:
    """
    Deletes lines starting with # Test or # test and all subsequent content from the string.
    """
    match = re.search(r"^#\s*(test|example).*$", input_str, re.IGNORECASE | re.MULTILINE)
    if match:
        return input_str[:match.start()].rstrip()
    return input_str

def remove_after_last_return(code: str) -> str:
    """
    Deletes all content after the last 'return' statement in the given code.
    """
    pattern = r'(.*)(return[\s\S]*)'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        # Reconstruct the string up to the end of the line containing the last return
        return match.group(1) + match.group(2).split('\n')[0]
    return code

def remove_main_function_cpp(cpp_code: str) -> str:
    """Removes the main function from a C++ code string."""
    pattern = r"int\s+main\s?\(.*?\)\s?\{.*\}"
    cleaned_code = re.sub(pattern, "", cpp_code, flags=re.DOTALL)
    return cleaned_code.strip()

def remove_main_function_go(go_code: str) -> str:
    """Removes the main function from a Go code string."""
    go_code_no_main = re.sub(r'func main\([^\)]*\)\s*\{.*\}', '', go_code, flags=re.DOTALL)
    return go_code_no_main.strip()

def remove_main_function_java(java_code: str) -> str:
    """Removes the main method from a Java code string."""
    pattern = r"public\s+static\s+void\s+main\s*\(.*\)\s*\{[\s\S]*"
    cleaned_code = re.sub(pattern, "", java_code, flags=re.DOTALL)
    return cleaned_code.strip()

def extract_python_code(text: str) -> str:
    """
    Extracts code blocks from text and processes them based on language type.
    It prioritizes extracting the last Python code block that contains a function or class definition.
    """
    pattern = r"```(\w+)\s*?([\s\S]*?)```"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    all_code_blocks = [(m.group(1).lower(), m.group(2).strip()) for m in matches]

    if not all_code_blocks:
        return text

    last_suitable_python_block_content = None
    # Iterate in reverse to find the last suitable Python block
    for lang, block_content in reversed(all_code_blocks):
        if lang in {'python', 'py', 'python3'}:
            # Check for function or class definitions
            if re.search(r"\b(def|class)\s+[\w_]+\s*[:\(]", block_content):
                last_suitable_python_block_content = block_content
                break

    if last_suitable_python_block_content:
        # Process the found Python block
        processed = remove_test_section(last_suitable_python_block_content)
        processed = remove_main_block(processed)
        processed = remove_code_block_markers(processed)
        return processed.strip()
    else:
        # Fallback to the last block if no suitable Python block is found
        lang_last, content_last = all_code_blocks[-1]
        if lang_last in {'python', 'py', 'python3'}:
            # If the last block is Python but didn't meet the criteria, return it as-is (or maybe the whole text?)
            # Returning the original text as per the original logic
            return text
        elif lang_last in {'cpp', 'c++', 'c'}:
            return remove_main_function_cpp(content_last)
        elif lang_last == 'go':
            return remove_main_function_go(content_last)
        elif lang_last == 'java':
            return remove_main_function_java(content_last)
        else:
            # Return the content of the last block for any other language
            return content_last
