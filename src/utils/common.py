import os
import json
import random
import numpy as np

from framework.data_return import *
from registry.dataset_map import LOAD_TEST_DATASET


def exponential_backoff(retry_count: int,
                        base_delay: int = 5,
                        max_delay: int = 65,
                        jitter: bool = True) -> float:
    delay = min(base_delay * (2 ** retry_count), max_delay)
    if jitter:
        delay = random.uniform(delay * 0.8, delay * 1.2)
    return delay


def _save_openended_jsonl(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    """
    Writes JSONL preserving original schema and adding a new field with the transformed result.
    Output file: {save_path}/{file_name}.jsonl

    For:
      - ifeval: adds 'prompt_transformed'
      - alpacafarm: adds 'instruction_transformed'
      - mt-bench: adds 'turns_transformed' (list[str])
    """
    os.makedirs(save_config.save_path, exist_ok=True)
    out_path = os.path.join(save_config.save_path, f"{save_config.file_name}.jsonl")

    # Build a map index->transformed item for rerun updates
    # (to_save['question'] is ordered by run order; if rerun_index exists, it corresponds to those indices)
    transformed_by_index = {}

    if rerun_index is None:
        for i, output in enumerate(to_save['question']):
            transformed_by_index[i] = output
    else:
        for i, output in enumerate(to_save['question']):
            idx = int(rerun_index[i])
            transformed_by_index[idx] = output

    # Load existing output if rerun and file exists, then update indices
    existing_rows = None
    if rerun_index is not None and os.path.exists(out_path):
        existing_rows = []
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_rows.append(json.loads(line))

    # Determine dataset name from save_config (your config may store it; fallback to infer)
    dataset_name = getattr(save_config, "dataset_name", None)
    if dataset_name is None:
        # fallback: try derive from path or file_name (not perfect)
        dataset_name = "openended"

    # Build output rows
    rows = []
    for i in range(len(test_dataset)):
        base = dict(test_dataset[i])

        if i in transformed_by_index:
            out = transformed_by_index[i]

            # mt-bench: we expect our pipeline to store turns as list under out['final_turns']
            if 'final_turns' in out:
                base['turns_transformed'] = out['final_turns']
            else:
                # default single sentence case
                base['text_transformed'] = out['final_sentence']

            # Optional: also keep rules used
            base['applied_rules'] = out.get('applied_rules', out.get('applied_rule', []))

        # If rerun and we had an existing row, update that one instead of rewriting everything
        rows.append(base)

    # If rerun updating existing file, merge: keep old lines for untouched indices
    if existing_rows is not None and len(existing_rows) == len(rows):
        for idx in range(len(rows)):
            if idx not in transformed_by_index:
                rows[idx] = existing_rows[idx]

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_func(to_save, save_config, dataset_config, generation_config, task_config):
    save_mapping = {
        'mmlu': return_mmlu,
        'gsm8k': return_gsm8k,
        'arc': return_arc,
        'hellaswag': return_hellaswag,
        'truthfulqa': return_truthfulqa,
        'winogrande': return_winogrande,

        # Open-ended datasets -> JSONL saver
        'ifeval': _save_openended_jsonl,
        'alpacafarm': _save_openended_jsonl,
        'mt-bench': _save_openended_jsonl,
    }

    rerun_index = None
    cefr_index = None

    test_dataset = LOAD_TEST_DATASET[dataset_config.dataset_name]

    if generation_config.rerun is not None:
        rerun_index = list(np.load(generation_config.rerun))

    # Keep original behavior for existing CEFR/L1 benchmark codepaths
    if task_config.task_name in ['cefr', 'L1']:
        try:
            cefr_index = list(np.load(f'assets/vocab_processed/{dataset_config.dataset_name}_{(task_config.cefr_level).lower()}.npy'))
            cefr_index = [int(index) for index in cefr_index]
        except Exception:
            cefr_index = None

    return save_mapping[dataset_config.dataset_name](test_dataset, to_save, save_config, rerun_index, cefr_index)