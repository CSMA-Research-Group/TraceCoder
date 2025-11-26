import os
import json
import pandas as pd

def load_dataset(dataset_name: str, dataset_path: str):
    """
    Loads a dataset based on its name and path.
    """
    print(f"Loading dataset '{dataset_name}' from path: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_name == 'bigcodebench':
        return load_parquet_dataset(dataset_path)
    elif dataset_name == 'humaneval':
        # HumanEval is typically a jsonl file
        return load_jsonl_dataset(dataset_path)
    # More dataset loading logic can be added here
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

def load_jsonl_dataset(path):
    datasets_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            task_id = data.get('task_id', f"jsonl_task/{i}")
            data['task_id'] = task_id
            # Ensure key fields exist
            data.setdefault('entry_point', data.get('name', f'func_{i}'))
            data.setdefault('prompt', data.get('signature', f"def {data['entry_point']}():\n    pass"))
            data.setdefault('test', data.get('tests', "assert True # Dummy test"))
            data.setdefault('complete_prompt', data.get('prompt') + data.get('docstring', ''))
            datasets_dict[task_id] = data
    print(f"Loaded {len(datasets_dict)} problems from {path}")
    return datasets_dict

def load_parquet_dataset(path):
    datasets_dict = {}
    df = pd.read_parquet(path)
    if 'task_id' not in df.columns:
        df['task_id'] = [f"parquet_task/{i}" for i in range(len(df))]
    for _, row in df.iterrows():
        data = row.to_dict()
        task_id = data['task_id']
        data.setdefault('entry_point', data.get('name', f'func_{task_id.split("/")[-1]}'))
        data.setdefault('test', data.get('tests', "assert True # Dummy test"))
        # The 'prompt' field in BigCodeBench is usually the complete prompt
        data.setdefault('complete_prompt', data.get('prompt', ''))
        datasets_dict[task_id] = data
    print(f"Loaded {len(datasets_dict)} problems from {path}")
    return datasets_dict
