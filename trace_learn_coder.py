import sys
import time
from itertools import islice
from tqdm import tqdm

# Import our split-out modules
from config import setup_arg_parser, DATASET_PATHS
from reporting import save_results, print_summary
# Import process_problem and _load_check_correctness_func from problem_processor
from problem_processor import process_problem, _load_check_correctness_func
from src.dataset_loader import load_dataset


def main():
    """
    The main execution function for the experiment.
    """
    # 1. Setup and parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # 2. Generate experiment ID and output filename
    EXPERIMENT_ID = f"{args.dataset}_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_CSV_FILENAME = args.output_file or f"results/experiment_{EXPERIMENT_ID}.csv"

    # 3. Load the dataset
    if args.dataset not in DATASET_PATHS:
        print(f"Error: Path for dataset '{args.dataset}' is not defined in config.py.")
        sys.exit(1)

    try:
        # Get 'data_path' from the DATASET_PATHS[args.dataset] dictionary
        actual_dataset_path = DATASET_PATHS[args.dataset]["data_path"]
        datasets = load_dataset(args.dataset, actual_dataset_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    except KeyError: # Added KeyError catch in case the "data_path" key is missing
        print(f"Error: The configuration for dataset '{args.dataset}' is missing 'data_path'. Please check config.py.")
        sys.exit(1)

    # <<< New code: Load the check_correctness function >>>
    try:
        check_correctness_func = _load_check_correctness_func(args.dataset)
    except (ValueError, ImportError) as e:
        print(f"Error loading the evaluation function check_correctness: {e}")
        sys.exit(1)
    # <<< End new code >>>

    print(f"--- Experiment Start: {EXPERIMENT_ID} ---")
    print(f"Arguments: {vars(args)}")

    # 4. Set up the problem iterator
    problem_keys = list(datasets.keys())
    start_index = args.start_index
    max_problems = len(problem_keys) if args.max_problems == -1 else args.max_problems
    end_index = min(start_index + max_problems, len(problem_keys))

    problem_iterator = islice(problem_keys, start_index, end_index)

    results_data = []

    # 5. Main loop
    try:
        for task_id in tqdm(problem_iterator, total=(end_index - start_index), desc="Processing problems"):
            # Pass the loaded check_correctness_func down
            problem_result = process_problem(datasets[task_id], task_id, args, check_correctness_func)
            results_data.append(problem_result)
    except KeyboardInterrupt:
        print("\nManual interruption detected. Saving existing results...")
    finally:
        # 6. Save results and print summary
        save_results(results_data, OUTPUT_CSV_FILENAME)
        print_summary(results_data)


if __name__ == '__main__':
    main()
