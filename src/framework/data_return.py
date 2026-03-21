import os
import numpy as np
import pandas as pd

from registry.dataset_map import DATASET_MAPPING

# Open-ended JSONL loaders take a 4th `sampling` flag; HuggingFace benchmark loaders do not
# (except winogrande — see winogrande_dataloader).
_OPEN_ENDED_DATASETS = frozenset({"ifeval", "alpacafarm", "mt-bench"})


def return_dataloader(dataset_config, generation_config, start_idx=None):
    rerun_index = None
    if generation_config.rerun is not None:
        rerun_index = list(np.load(generation_config.rerun))
    batch_size = generation_config.batch_size
    name = dataset_config.dataset_name
    loader_fn = DATASET_MAPPING[name]
    if name in _OPEN_ENDED_DATASETS:
        return loader_fn(batch_size, rerun_index, start_idx, dataset_config.sampling)
    if name == "winogrande":
        return loader_fn(batch_size, rerun_index, start_idx, dataset_config.sampling)
    return loader_fn(batch_size, rerun_index, start_idx)


def return_openended(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    """
    Open-ended datasets saver for: ifeval, alpacafarm, mt-bench

    Writes CSV with columns:
      - index
      - source_text (prompt/instruction/turns joined)
      - transformed (final_sentence)
    """
    if rerun_index is None:
        rows = []
        get_idx = (lambda i: cefr_index[i]) if cefr_index is not None else (lambda i: i)

        for i, output in enumerate(to_save["question"]):
            idx = get_idx(i)
            src = test_dataset[idx]

            # mt-bench: turns is list[str]
            if "turns" in src and isinstance(src["turns"], list):
                source_text = "\n".join(src["turns"])
            else:
                # alpacafarm: instruction, ifeval: prompt (often), fallback keys included
                source_text = (
                    src.get("instruction")
                    or src.get("prompt")
                    or src.get("input")
                    or src.get("question")
                    or ""
                )

            rows.append(
                {
                    "index": idx,
                    "source_text": source_text,
                    "transformed": output["final_sentence"],
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    else:
        out_rerun = os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")
        out_base = os.path.join(save_config.save_path, f"{save_config.file_name}.csv")

        if os.path.isfile(out_rerun):
            df = pd.read_csv(out_rerun)
        elif os.path.isfile(out_base):
            df = pd.read_csv(out_base)
        else:
            df = pd.DataFrame(columns=["index", "source_text", "transformed"])

        for i, output in enumerate(to_save["question"]):
            idx = int(rerun_index[i])
            src = test_dataset[idx]

            if "turns" in src and isinstance(src["turns"], list):
                source_text = "\n".join(src["turns"])
            else:
                source_text = (
                    src.get("instruction")
                    or src.get("prompt")
                    or src.get("input")
                    or src.get("question")
                    or ""
                )

            new_row = {
                "index": idx,
                "source_text": source_text,
                "transformed": output["final_sentence"],
            }

            if "index" in df.columns and (df["index"] == idx).any():
                df.loc[df["index"] == idx, ["source_text", "transformed"]] = (source_text, output["final_sentence"])
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(out_rerun, index=False)


def return_mmlu(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:

        columns = list(test_dataset.features.keys())

        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "question": output["final_sentence"].replace("<blank>", "_______"),
                        "subject": test_dataset[i]["subject"],
                        "choices": [test_dataset[i]["choices"]],
                        "answer": test_dataset[i]["answer"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "question": output["final_sentence"].replace("<blank>", "_______"),
                        "subject": test_dataset[cefr_index[i]]["subject"],
                        "choices": [test_dataset[cefr_index[i]]["choices"]],
                        "answer": test_dataset[cefr_index[i]]["answer"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"))

        for i, output in enumerate(to_save["question"]):
            index = int(rerun_index[i])
            new_row = {
                "question": output["final_sentence"].replace("<blank>", "_______"),
                "subject": test_dataset[index]["subject"],
                "choices": [test_dataset[index]["choices"]],
                "answer": test_dataset[index]["answer"],
            }

            df.loc[index] = new_row

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"), index=False)


def return_gsm8k(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "question": output["final_sentence"],
                        "answer": test_dataset[i]["answer"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "question": output["final_sentence"],
                        "answer": test_dataset[cefr_index[i]]["answer"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"))

        for i, output in enumerate(to_save["question"]):
            index = int(rerun_index[i])
            new_row = {
                "question": output["final_sentence"],
                "answer": test_dataset[index]["answer"],
            }

            df.loc[index] = new_row

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"), index=False)


def return_arc(test_dataset, to_save_dict, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())

        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save_dict["question"]):
                new_row = pd.DataFrame(
                    {
                        "id": test_dataset[i]["id"],
                        "question": output["final_sentence"],
                        "choices": [test_dataset[i]["choices"]],
                        "answerKey": test_dataset[i]["answerKey"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save_dict["question"]):
                new_row = pd.DataFrame(
                    {
                        "id": test_dataset[cefr_index[i]]["id"],
                        "question": output["final_sentence"],
                        "choices": [test_dataset[cefr_index[i]]["choices"]],
                        "answerKey": test_dataset[cefr_index[i]]["answerKey"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"))

        for i, output in enumerate(to_save_dict["question"]):
            index = int(rerun_index[i])
            new_row = {
                "id": test_dataset[index]["id"],
                "question": output["final_sentence"],
                "choices": [test_dataset[index]["choices"]],
                "answerKey": test_dataset[index]["answerKey"],
            }

            df.loc[index] = new_row

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"), index=False)


def return_hellaswag(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "ind": test_dataset[i]["ind"],
                        "activity_label": test_dataset[i]["activity_label"],
                        "ctx_a": output["final_sentence"],
                        "ctx_b": test_dataset[i]["ctx_b"],
                        "ctx": output["final_sentence"] + " " + test_dataset[i]["ctx_b"],
                        "endings": [test_dataset[i]["endings"]],
                        "source_id": test_dataset[i]["source_id"],
                        "split": test_dataset[i]["split"],
                        "split_type": test_dataset[i]["split_type"],
                        "label": test_dataset[i]["label"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "ind": test_dataset[cefr_index[i]]["ind"],
                        "activity_label": test_dataset[cefr_index[i]]["activity_label"],
                        "ctx_a": test_dataset[cefr_index[i]]["ctx_a"],
                        "ctx_b": test_dataset[cefr_index[i]]["ctx_b"],
                        "ctx": output["final_sentence"],
                        "endings": [test_dataset[cefr_index[i]]["endings"]],
                        "source_id": test_dataset[cefr_index[i]]["source_id"],
                        "split": test_dataset[cefr_index[i]]["split"],
                        "split_type": test_dataset[cefr_index[i]]["split_type"],
                        "label": test_dataset[cefr_index[i]]["label"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"))

        for i, output in enumerate(to_save["question"]):
            index = int(rerun_index[i])
            new_row = {
                "ind": test_dataset[index]["ind"],
                "activity_label": test_dataset[index]["activity_label"],
                "ctx_a": output["final_sentence"],
                "ctx_b": test_dataset[index]["ctx_b"],
                "ctx": output["final_sentence"],
                "endings": test_dataset[index]["endings"],
                "source_id": test_dataset[index]["source_id"],
                "split": test_dataset[index]["split"],
                "split_type": test_dataset[index]["split_type"],
                "label": test_dataset[index]["label"],
            }

            df.loc[index] = new_row

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"), index=False)


def return_truthfulqa(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "question": output["final_sentence"],
                        "mc1_targets": [test_dataset[i]["mc1_targets"]],
                        "mc2_targets": [test_dataset[i]["mc2_targets"]],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "question": output["final_sentence"],
                        "mc1_targets": [test_dataset[cefr_index[i]]["mc1_targets"]],
                        "mc2_targets": [test_dataset[cefr_index[i]]["mc2_targets"]],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"))

        for i, output in enumerate(to_save["question"]):
            index = int(rerun_index[i])
            new_row = {
                "question": output["final_sentence"],
                "mc1_targets": test_dataset[index]["mc1_targets"],
                "mc2_targets": test_dataset[index]["mc2_targets"],
            }

            df.loc[index] = new_row

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"), index=False)


def return_winogrande(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "sentence": output["final_sentence"],
                        "option1": test_dataset[i]["option1"],
                        "option2": test_dataset[i]["option2"],
                        "answer": test_dataset[i]["answer"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save["question"]):
                new_row = pd.DataFrame(
                    {
                        "sentence": output["final_sentence"],
                        "option1": test_dataset[cefr_index[i]]["option1"],
                        "option2": test_dataset[cefr_index[i]]["option2"],
                        "answer": test_dataset[cefr_index[i]]["answer"],
                    },
                    index=[0],
                )

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"), index=False)

    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv")) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f"{save_config.file_name}.csv"))

        for i, output in enumerate(to_save["question"]):
            index = int(rerun_index[i])
            new_row = {
                "sentence": output["final_sentence"],
                "option1": test_dataset[index]["option1"],
                "option2": test_dataset[index]["option2"],
                "answer": test_dataset[index]["answer"],
            }

            df.loc[index] = new_row

        df.to_csv(os.path.join(save_config.save_path, f"{save_config.file_name}_rerun.csv"), index=False)
