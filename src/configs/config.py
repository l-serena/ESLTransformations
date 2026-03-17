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
        default=200,
        metadata={
            "help": "Max guideline rules to try per example (helps avoid very long runs). Set 0 to disable limit."
        },
    )
    max_workers: int = field(
        default=10,
        metadata={"help": "Max concurrent API calls per rule (OpenAI path). Set 1 to disable parallelism."},
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