import argparse

# --- Dataset Path Configuration ---
DATASET_PATHS = {
    "humaneval": {
        "data_path": "./datasets/human_eval/data/test.jsonl", # Please replace with the actual local path
        "eval_module": "datasets.human_eval.execution" # Python import path for the evaluation module
    },
    "humanevalplus": {
        "data_path": "./datasets/human_eval_plus/data/test-00000-of-00001-5973903632b82d40.parquet", # Please replace with the actual local path
        "eval_module": "datasets.human_eval.execution" # Python import path for the evaluation module
    },
    "bigcodebench": {
        "data_path": "./datasets/BigCodeBench/data/v0.1.4-00000-of-00001.parquet",   # Please replace with the actual local path
        "eval_module": "datasets.BigCodeBench.evaluation" # Python import path for the evaluation module
    },
    "classeval": { # Added eval_module for ClassEval
        "data_path": "./datasets/ClassEval/data/test-00000-of-00001-5c45fa6e45572491.parquet",      # Please replace with the actual local path
        "eval_module": "datasets.ClassEval.evaluation" # Python import path for the evaluation module
    }
    # Add other datasets here, ensure they have 'data_path' and 'eval_module'
}

# Note: Please ensure the above paths point to the actual location of your dataset files on your local machine.


def setup_arg_parser():
    """
    Sets up and parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="A framework for running code generation and repair experiments")
    parser.add_argument('-m', '--model', type=str, default='gemini-1.5-flash-preview-0514', help='The name of the model to use')
    # Update choices to match the keys in DATASET_PATHS
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=list(DATASET_PATHS.keys()), help='The dataset to use')
    parser.add_argument('--no-instrumentation', action='store_true', help='Disable the code instrumentation step')
    parser.add_argument('--no-two-step-repair', action='store_true', help='Disable the two-step repair process and use a single-step repair instead')
    parser.add_argument('--start-index', type=int, default=0, help='The index of the problem in the dataset to start processing from')
    parser.add_argument('--max-problems', type=int, default=-1, help='The maximum number of problems to process (-1 means process all)')
    parser.add_argument('--max-debug-attempts', type=int, default=5, help='The maximum number of debugging attempts per problem')
    parser.add_argument('--max-no-improvement-streak', type=int, default=2, help='The threshold for stopping debugging after consecutive attempts with no improvement')
    parser.add_argument('--timeout', type=int, default=10, help='The timeout for code execution in seconds')
    parser.add_argument('-o', '--output-file', type=str, default=None, help='Specify the path and name for the output CSV file')
    return parser
