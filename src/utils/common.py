import os
import json
import random
import numpy as np

from framework.data_return import *
from registry.dataset_map import LOAD_TEST_DATASET, load_openended_dataset


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


def _openended_row_has_transform(row: dict, dataset_name: str) -> bool:
    """True if this JSONL row was already merged with a transformation (matches _save_openended_jsonl)."""
    if dataset_name == "mt-bench":
        v = row.get("turns_transformed")
        return isinstance(v, list)
    return "text_transformed" in row and row.get("text_transformed") is not None


def _jsonl_row_to_question_item(row: dict, dataset_name: str) -> dict:
    """Rebuild one `to_save['question']` entry from a merged JSONL row (for resume without .pk)."""
    applied = row.get("applied_rules", row.get("applied_rule", []))
    if dataset_name == "mt-bench":
        turns = row.get("turns") or []
        ft = list(row.get("turns_transformed") or [])
        return {
            "orig_sentence": turns,
            "whole_response": [],
            "mid_transformed_sentences": [],
            "judge_repsonse": [],
            "applied_rules": applied if isinstance(applied, list) else [],
            "transformed_sentences": [],
            "final_turns": ft,
            "final_sentence": "\n".join(ft),
        }
    key_orig = "prompt" if dataset_name == "ifeval" else "instruction"
    orig = row.get(key_orig, "")
    return {
        "orig_sentence": orig,
        "whole_response": [],
        "mid_transformed_sentences": [],
        "judge_repsonse": [],
        "applied_rules": applied if isinstance(applied, list) else [],
        "transformed_sentences": [],
        "final_sentence": row.get("text_transformed", ""),
    }


def try_resume_openended_from_jsonl(save_config, dataset_config, generation_config):
    """
    When no `{file_name}.pk` exists, infer how many examples were already written by scanning
    `{file_name}.jsonl` for a leading prefix of rows that contain transformation fields.

    Skipped when `sampling=True` (dataloader length may not match full JSONL line count) or when
    `rerun` is set.
    """
    from utils import colorstr, log

    if generation_config.rerun is not None:
        return [], 0
    if dataset_config.dataset_name not in {"ifeval", "alpacafarm", "mt-bench"}:
        return [], 0
    if getattr(dataset_config, "sampling", False):
        return [], 0

    jsonl_path = os.path.join(save_config.save_path, f"{save_config.file_name}.jsonl")
    if not os.path.isfile(jsonl_path):
        return [], 0

    rows = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    rows.append(json.loads(t))
    except (json.JSONDecodeError, OSError) as e:
        log(colorstr("yellow", f"Resume: could not read {jsonl_path}: {e}"))
        return [], 0

    try:
        ref_ds = load_openended_dataset(dataset_config.dataset_name, sampling=dataset_config.sampling)
        n_full = len(ref_ds)
    except Exception as e:
        log(colorstr("yellow", f"Resume: could not load dataset to verify JSONL length: {e}"))
        return [], 0

    if len(rows) != n_full:
        log(
            colorstr(
                "yellow",
                f"Resume: JSONL has {len(rows)} lines but dataset has {n_full}; "
                f"not inferring resume from {jsonl_path}.",
            )
        )
        return [], 0

    start_idx = 0
    for r in rows:
        if _openended_row_has_transform(r, dataset_config.dataset_name):
            start_idx += 1
        else:
            break

    if start_idx == 0:
        return [], 0

    to_save = [_jsonl_row_to_question_item(rows[i], dataset_config.dataset_name) for i in range(start_idx)]
    log(
        colorstr(
            "green",
            f"Resume: no .pk found; continuing from JSONL with {start_idx} completed example(s).",
        )
    )
    return to_save, start_idx


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

    # Ensure open-ended save uses the same sampled dataset as the run.
    if dataset_config.dataset_name in {"ifeval", "alpacafarm", "mt-bench"}:
        test_dataset = load_openended_dataset(dataset_config.dataset_name, sampling=dataset_config.sampling)
    else:
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