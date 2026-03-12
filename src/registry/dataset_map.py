import os
from datasets import load_dataset
from torch.utils.data import DataLoader

from benchmark.mmlu import (
    mmlu_dataloader,
    load_mmlu_test_dataset,
    load_test_data as mmlu_load_test_data,
    extract_answer as mmlu_extract_answer,
)
from benchmark.gsm8k import (
    gsm8k_dataloader,
    load_test_data as gsm8k_load_test_data,
    extract_answer as gsm8k_extract_answer,
)
from benchmark.arc import (
    arc_dataloader,
    load_test_data as arc_load_test_data,
    extract_answer as arc_extract_answer,
)
from benchmark.hellaswag import (
    hellaswag_dataloader,
    load_test_data as hellaswag_load_test_data,
    extract_answer as hellaswag_extract_answer,
)
from benchmark.truthful_qa import (
    truthfulqa_dataloader,
    load_test_data as truthfulqa_load_test_data,
    extract_answer as truthfulqa_extract_answer,
)
from benchmark.winogrande import (
    winogrande_dataloader,
    load_test_data as winogrande_load_test_data,
    extract_answer as winogrande_extract_answer,
)


def _select(ds, rerun_index=None, start_idx=None):
    if start_idx is not None and start_idx > 0:
        ds = ds.select(range(start_idx, len(ds)))
    if rerun_index is not None:
        ds = ds.select(rerun_index)
    return ds


def _openended_collate_fn(batch):
    """
    Open-ended JSONL rows often include variable-length list fields (e.g., instruction_id_list, kwargs).
    The default PyTorch collator tries to stack/align them and can fail.
    This collator keeps everything as simple Python lists.
    """
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: [ex.get(k) for ex in batch] for k in keys}


def _pick_openended_jsonl(dataset_name: str, sampling: bool) -> str:
    """
    Resolve the JSONL path for an open-ended dataset.

    If sampling=True, prefer an existing small sample file if present.
    """
    data_path = os.environ.get("DATA_PATH", ".")
    base_dir = os.path.join(data_path, "datasets", dataset_name)

    if dataset_name == "alpacafarm":
        # Repo includes multiple sample files; prefer the smallest first.
        if sampling:
            for fname in ("sample3.jsonl", "sample50.jsonl", "sample.jsonl", "test.jsonl"):
                p = os.path.join(base_dir, fname)
                if os.path.exists(p):
                    return p
        return os.path.join(base_dir, "sample50.jsonl")

    # ifeval / mt-bench: default is test.jsonl; if sampling, use sample*.jsonl if user created it
    if sampling:
        for fname in ("sample.jsonl", "sample10.jsonl", "sample3.jsonl", "sample50.jsonl", "test.jsonl"):
            p = os.path.join(base_dir, fname)
            if os.path.exists(p):
                return p

    return os.path.join(base_dir, "test.jsonl")


def load_openended_dataset(dataset_name: str, sampling: bool):
    path = _pick_openended_jsonl(dataset_name, sampling=sampling)
    return load_dataset("json", data_files={"test": path}, split="test")


def load_ifeval(batch_size, rerun_index=None, start_idx=None, sampling: bool = False):
    data_path = os.environ.get("DATA_PATH", ".")
    path = _pick_openended_jsonl("ifeval", sampling=sampling)
    ds = load_dataset("json", data_files={"test": path}, split="test")
    ds = _select(ds, rerun_index, start_idx)
    # If user asked for sampling but only test.jsonl exists, still keep runs small.
    if sampling and ("test.jsonl" in os.path.basename(path)) and len(ds) > 10:
        ds = ds.select(range(10))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_openended_collate_fn)


def load_alpacafarm(batch_size, rerun_index=None, start_idx=None, sampling: bool = False):
    data_path = os.environ.get("DATA_PATH", ".")
    path = _pick_openended_jsonl("alpacafarm", sampling=sampling)
    ds = load_dataset("json", data_files={"test": path}, split="test")
    ds = _select(ds, rerun_index, start_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_openended_collate_fn)


def load_mt_bench(batch_size, rerun_index=None, start_idx=None, sampling: bool = False):
    data_path = os.environ.get("DATA_PATH", ".")
    path = _pick_openended_jsonl("mt-bench", sampling=sampling)
    ds = load_dataset("json", data_files={"test": path}, split="test")
    ds = _select(ds, rerun_index, start_idx)
    if sampling and ("test.jsonl" in os.path.basename(path)) and len(ds) > 10:
        ds = ds.select(range(10))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_openended_collate_fn)


# Function definition of each benchmark
MAIN_FUNCS = {
    "mmlu": (mmlu_load_test_data, mmlu_extract_answer, 13436, "question"),
    "gsm8k": (gsm8k_load_test_data, gsm8k_extract_answer, 1319, "question"),
    "arc": (arc_load_test_data, arc_extract_answer, 1172, "question"),
    "hellaswag": (hellaswag_load_test_data, hellaswag_extract_answer, 10042, "ctx"),
    "truthful_qa": (
        truthfulqa_load_test_data,
        truthfulqa_extract_answer,
        817,
        "question",
    ),
    "winogrande": (
        winogrande_load_test_data,
        winogrande_extract_answer,
        1267,
        "sentence",
    ),
}


# Test datasets
LOAD_TEST_DATASET = {
    'mmlu': load_mmlu_test_dataset(),
    'gsm8k': load_dataset('openai/gsm8k', 'main', split='test', cache_dir=os.environ.get("DATA_DIR", None)),
    'arc': load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test', cache_dir=os.environ.get("DATA_DIR", None)),
    'hellaswag': load_dataset('Rowan/hellaswag', split='validation', cache_dir=os.environ.get("DATA_DIR", None)),
    'truthfulqa': load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation', cache_dir=os.environ.get("DATA_DIR", None)),
    'winogrande': load_dataset('allenai/winogrande', split='validation', trust_remote_code=True, name='winogrande_m', cache_dir=os.environ.get("DATA_DIR", None)),

    # Open-ended (local JSONL)
    # NOTE: sampling-specific loading is handled in save_func via load_openended_dataset().
    'ifeval': load_openended_dataset("ifeval", sampling=False),
    'alpacafarm': load_openended_dataset("alpacafarm", sampling=False),
    'mt-bench': load_openended_dataset("mt-bench", sampling=False),
}


# Dataloaders
DATASET_MAPPING = {
    'mmlu': mmlu_dataloader,
    'gsm8k': gsm8k_dataloader,
    'arc': arc_dataloader,
    'hellaswag': hellaswag_dataloader,
    'truthfulqa': truthfulqa_dataloader,
    'winogrande': winogrande_dataloader,

    # Open-ended
    'ifeval': load_ifeval,
    'alpacafarm': load_alpacafarm,
    'mt-bench': load_mt_bench,
}