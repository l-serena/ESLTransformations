from dataclasses import dataclass, field



# =========================
# Generation Config
# =========================
@dataclass
class GenerationConfig:
    temperature: float = field(default=0.8, metadata={"help": "temperature value for generation"})
    top_p: float = field(default=0.95, metadata={"help": "top_p value for generation"})
    batch_size: int = field(default=30, metadata={"help": "batch size for generation"})
    max_tokens: int = field(default=2000, metadata={"help": "max tokens for generation"})
    rerun: str = field(default=None, metadata={"help": "Path to rerun indices (.npy)"})
    one_transform: bool = field(default=False, metadata={"help": "Apply only one transformation", "action": "store_true"})
    max_rules: int = field(
        default=0,
        metadata={
            "help": "Max guideline rules to try per example. Default 0 = no limit (full guideline list). "
            "Set a positive integer (e.g. 200) to cap for shorter runs."
        },
    )
    max_workers: int = field(
        default=25,
        metadata={
            "help": "Max concurrent API calls per rule (OpenAI path). Raise to 50–100 if your tier allows; set 1 for sequential."
        },
    )
    passthrough_mt_bench: bool = field(
        default=False,
        metadata={
            "help": "If set, do not transform mt-bench turns; keep original prompts as turns_transformed (avoids mangling instructions)."
        },
    )
    max_chain_depth: int = field(
        default=0,
        metadata={
            "help": "Max successful transforms stacked per sentence (0 = unlimited). "
            "Use 3–8 for open-ended text so many rules do not turn prompts into gibberish."
        },
    )
    skip_semantic_check: bool = field(
        default=False,
        metadata={
            "help": "If set, skip the semantic checker (SS) after each transform. "
            "Default (off) matches Trans-EnV: keep a transform only if SS says meaning is preserved."
        },
    )


# =========================
# Model Config
# =========================
@dataclass
class ModelConfig:
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "Model to use",
            "choices": [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Meta-Llama-3-8B",
                "google/gemma-2-27b-it",
                "google/gemma-2-9b-it",
                "gpt-4o-mini"
            ]
        }
    )
    port_num: int = field(default=8000, metadata={"help": "vLLM / OpenAI-compatible server port"})
    tokenizer: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")


# =========================
# Dataset Config
# =========================
@dataclass
class DatasetConfig:
    dataset_name: str = field(
        metadata={
            "help": "Dataset name",
            "choices": [
                "mmlu",
                "gsm8k",
                "arc",
                "hellaswag",
                "truthfulqa",
                "winogrande",
                # NEW: open-ended datasets
                "ifeval",
                "alpacafarm",
                "mt-bench",
            ],
        }
    )
    sampling: bool = field(default=False, metadata={"help": "Use sampling subset", "action": "store_true"})


# =========================
# Task Config
# =========================
@dataclass
class TaskConfig:
    task_name: str = field(
        default="english_dialect",
        metadata={
            "help": "Transformation task",
            "choices": [
                "english_dialect",
                "cefr",
                "L1",
                # NEW TASK MODE
                "openended_cefr",
                "openended_l1",
                # Open-ended ESL variety: CEFR + L1 combined
                "openended_esl",
            ],
        },
    )

    dialect: str = field(
        default="Urban African American Vernacular English",
        metadata={"help": "English Dialect (used only for english_dialect task)"}
    )

    l1: str = field(
        default="Arabic",
        metadata={
            "help": "L1 language (used only for L1 task)",
            "choices": [
                "Arabic",
                "French",
                "German",
                "Italian",
                "Japanese",
                "Mandarin",
                "Portuguese",
                "Russian",
                "Spanish",
                "Turkish",
            ],
        },
    )

    cefr_level: str = field(
        default="A",
        metadata={
            "help": "CEFR Level (A, B, C). Used for cefr and openended_cefr tasks."
        },
    )


# =========================
# Save Config
# =========================
@dataclass
class SaveConfig:
    save_path: str = field(default="./outputs", metadata={"help": "Directory to save outputs"})
    file_name: str = field(default="tmp", metadata={"help": "Output file name (without extension)"})
    data_path: str = field(default="./", metadata={"help": "Project root path (used for loading datasets)"})