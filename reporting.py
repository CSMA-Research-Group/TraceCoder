import os
import json
import pandas as pd


def format_check_correctness_result(result_dict: dict) -> str:
    """
    Formats the evaluation result dictionary into a human-readable string.
    """
    if not isinstance(result_dict, dict):
        return "Invalid evaluation result format."
    
    if result_dict.get('passed', False):
        return "All tests passed."

    failure_detail = result_dict.get('result', "No specific failure details provided.")
    passed_count = result_dict.get('passed_count', -1)
    total_count = result_dict.get('total_count', -1)

    return f"Failed ({passed_count}/{total_count} passed): {failure_detail}"


def save_results(results_data, filename):
    """
    Saves the list of experiment results to a CSV file.
    """
    if not results_data:
        print("No results to save.")
        return

    try:
        df = pd.DataFrame(results_data)
        # Convert dictionary and list columns to JSON strings for safe storage
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nExperiment results have been successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def print_summary(results_data: list):
    """
    Prints a final statistical summary at the end of the experiment.
    """
    if not results_data:
        print("\n--- Experiment Summary ---")
        print("No problems were processed.")
        return

    df = pd.DataFrame(results_data)
    direct_pass_rate = df['direct_gen_passed'].mean() * 100
    final_pass_rate = df['final_passed'].mean() * 100
    
    print("\n--- Experiment Summary ---")
    print(f"Total problems processed: {len(df)}")
    print(f"Direct Pass Rate (Pass@1): {direct_pass_rate:.2f}%")
    print(f"Final Pass Rate (after debugging): {final_pass_rate:.2f}%")
