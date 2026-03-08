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


def load_ifeval(batch_size, rerun_index=None, start_idx=None):
    data_path = os.environ.get("DATA_PATH", ".")
    path = os.path.join(data_path, "datasets", "ifeval", "test.jsonl")
    ds = load_dataset("json", data_files={"test": path}, split="test")
    ds = _select(ds, rerun_index, start_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_alpacafarm(batch_size, rerun_index=None, start_idx=None):
    data_path = os.environ.get("DATA_PATH", ".")
    path = os.path.join(data_path, "datasets", "alpacafarm", "test.jsonl")
    ds = load_dataset("json", data_files={"test": path}, split="test")
    ds = _select(ds, rerun_index, start_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_mt_bench(batch_size, rerun_index=None, start_idx=None):
    data_path = os.environ.get("DATA_PATH", ".")
    path = os.path.join(data_path, "datasets", "mt-bench", "test.jsonl")
    ds = load_dataset("json", data_files={"test": path}, split="test")
    ds = _select(ds, rerun_index, start_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# Optional: placeholders if you accidentally try to run benchmark_eval with open-ended datasets.
# Keep these here only to fail with a clear error message.
def openended_load_test_data(*args, **kwargs):
    raise NotImplementedError(
        "Open-ended dataset eval is not implemented via MAIN_FUNCS. "
        "Use a dedicated evaluator (e.g., judge model / rubric / task-specific scoring)."
    )


def openended_extract_answer(*args, **kwargs):
    raise NotImplementedError(
        "Open-ended dataset answer extraction is not applicable. "
        "Use a dedicated evaluator (e.g., judge model / rubric / task-specific scoring)."
    )


# Function definition of each benchmark
# Tuple: (load_test_data_func, extract_answer_func, dataset_size, question_key)
MAIN_FUNCS = {
    "mmlu": (mmlu_load_test_data, mmlu_extract_answer, 13436, "question"),
    "gsm8k": (gsm8k_load_test_data, gsm8k_extract_answer, 1319, "question"),
    "arc": (arc_load_test_data, arc_extract_answer, 1172, "question"),
    "hellaswag": (hellaswag_load_test_data, hellaswag_extract_answer, 10042, "ctx"),
    "truthful_qa": (truthfulqa_load_test_data, truthfulqa_extract_answer, 817, "question"),
    "winogrande": (winogrande_load_test_data, winogrande_extract_answer, 1267, "sentence"),

    # Open-ended datasets (placeholders only; see functions above)
    "ifeval": (openended_load_test_data, openended_extract_answer, None, "prompt"),
    "alpacafarm": (openended_load_test_data, openended_extract_answer, None, "instruction"),
    "mt-bench": (openended_load_test_data, openended_extract_answer, None, "turns"),
}


# Test datasets
LOAD_TEST_DATASET = {
    "mmlu": load_mmlu_test_dataset(),
    "gsm8k": load_dataset("openai/gsm8k", "main", split="test", cache_dir=os.environ.get("DATA_DIR", None)),
    "arc": load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", cache_dir=os.environ.get("DATA_DIR", None)),
    "hellaswag": load_dataset("Rowan/hellaswag", split="validation", cache_dir=os.environ.get("DATA_DIR", None)),
    "truthfulqa": load_dataset(
        "truthfulqa/truthful_qa", "multiple_choice", split="validation", cache_dir=os.environ.get("DATA_DIR", None)
    ),
    "winogrande": load_dataset(
        "allenai/winogrande",
        split="validation",
        trust_remote_code=True,
        name="winogrande_m",
        cache_dir=os.environ.get("DATA_DIR", None),
    ),

    # Open-ended datasets (local JSONL)
    "ifeval": load_dataset(
        "json",
        data_files={"test": os.path.join(os.environ.get("DATA_PATH", "."), "datasets", "ifeval", "test.jsonl")},
        split="test",
    ),
    "alpacafarm": load_dataset(
        "json",
        data_files={"test": os.path.join(os.environ.get("DATA_PATH", "."), "datasets", "alpacafarm", "test.jsonl")},
        split="test",
    ),
    "mt-bench": load_dataset(
        "json",
        data_files={"test": os.path.join(os.environ.get("DATA_PATH", "."), "datasets", "mt-bench", "test.jsonl")},
        split="test",
    ),
}


# Dataloaders
DATASET_MAPPING = {
    "mmlu": mmlu_dataloader,
    "gsm8k": gsm8k_dataloader,
    "arc": arc_dataloader,
    "hellaswag": hellaswag_dataloader,
    "truthfulqa": truthfulqa_dataloader,
    "winogrande": winogrande_dataloader,

    # Open-ended datasets
    "ifeval": load_ifeval,
    "alpacafarm": load_alpacafarm,
    "mt-bench": load_mt_bench,
}
