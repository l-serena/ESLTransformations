import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer
from dataclasses import replace

from torch.utils.data import DataLoader

from configs.parse_arguments import parse_args
from framework.guideline import return_guideline
from framework.data_return import return_dataloader
from framework.transformation import transformation, openai_transformation
from registry.framework import QUESTION_KEY_ID
from utils import log, colorstr
from utils.common import save_func, try_resume_openended_from_jsonl
from utils.model_utils import return_model
from utils.filesys_utils import pickle_load, pickle_save

choice_transform_dataset = []

OPEN_ENDED_DATASETS = {"ifeval", "alpacafarm", "mt-bench"}


def _is_openai_model(model_name: str) -> bool:
    """
    Heuristic: treat OpenAI hosted models (gpt-*, o1-*, o3-*) as OpenAI API models.
    """
    if model_name is None:
        return False
    return model_name.startswith(("gpt-", "o1-", "o3-"))


def main():
    # Initialize arguments
    generation_config, model_config, dataset_config, task_config, save_config = parse_args()

    if dataset_config.dataset_name is None:
        raise AssertionError(colorstr('red', 'Dataset name should be specified!'))

    # Compatibility: allow closed-end CEFR task flag on open-ended datasets.
    # Users often run "cefr_a" (task_name=cefr, cefr_level=A) and expect it to work on jsonl datasets too.
    if (dataset_config.dataset_name in OPEN_ENDED_DATASETS) and (task_config.task_name == "cefr"):
        task_config = replace(task_config, task_name="openended_cefr")

    # Compatibility: allow benchmark L1 task flag on open-ended datasets.
    # This keeps the same L1 rewrite behavior, but avoids requiring CEFR benchmark CSV assets.
    if (dataset_config.dataset_name in OPEN_ENDED_DATASETS) and (task_config.task_name == "L1"):
        task_config = replace(task_config, task_name="openended_l1")

    # NOTE:
    # - Keep original L1 constraint ONLY for the benchmark L1 mode (task_name == 'L1').
    # - openended_l1 should NOT require cefr_level.
    if task_config.task_name == 'L1' and task_config.cefr_level is None:
        raise AssertionError(colorstr('red', 'You should specify cefr level in order to change L1.'))

    # Logging
    if task_config.task_name == 'L1':
        log(
            f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, '
            f'l1: {colorstr(task_config.l1)}, cefr: {colorstr(task_config.cefr_level)}, '
            f'Rerun: {colorstr(bool(generation_config.rerun))}'
        )
    elif task_config.task_name == 'openended_l1':
        log(
            f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, '
            f'l1: {colorstr(task_config.l1)}, Rerun: {colorstr(bool(generation_config.rerun))}'
        )
    elif task_config.task_name == 'openended_esl':
        log(
            f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, '
            f'CEFR level: {colorstr(task_config.cefr_level)}, l1: {colorstr(task_config.l1)}, '
            f'Rerun: {colorstr(bool(generation_config.rerun))}'
        )
    if dataset_config.dataset_name == 'mt-bench' and generation_config.passthrough_mt_bench:
        log(colorstr('yellow', 'mt-bench passthrough: ON — turns will not be transformed (original prompts kept).'))
    elif task_config.task_name == 'english_dialect':
        log(
            f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, '
            f'dialect: {colorstr(task_config.dialect)}, Rerun: {colorstr(bool(generation_config.rerun))}'
        )
    elif task_config.task_name in ['cefr', 'openended_cefr']:
        log(
            f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, '
            f'CEFR level: {colorstr(task_config.cefr_level)}, Rerun: {colorstr(bool(generation_config.rerun))}'
        )

    os.makedirs(save_config.save_path, exist_ok=True)

    if dataset_config.sampling is True:
        save_config.file_name += '_sampling'

    # Initialize model client
    log("Connecting to model server...")
    client = return_model(model_config=model_config)

    # Decide routing (OpenAI API vs HF/vLLM-style)
    use_openai_api = _is_openai_model(model_config.model_name)
    is_hf_model_id = (model_config.model_name is not None) and ("/" in model_config.model_name)

    # Tokenizer only needed for HF/vLLM-style transformation() (tokenizer.apply_chat_template)
    tokenizer = None
    if (not use_openai_api) and is_hf_model_id and (model_config.model_name.split('/')[0] != 'azure'):
        log("Loading tokenizer (may download on first run)...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name,
            cache_dir=os.environ.get("MODEL_DIR", None)
        )
        log("Tokenizer ready.")

    # Guideline
    log("Loading guidelines...")
    guideline = return_guideline(
        task_config=task_config,
        dataset_name=dataset_config.dataset_name,
        data_path=save_config.data_path
    )

    to_save = list()
    to_save_choice = defaultdict(list)

    # Resume: prefer .pk; for open-ended JSONL runs, infer completed prefix from .jsonl if no .pk
    start_idx = 0
    pk_path = os.path.join(save_config.save_path, f'{save_config.file_name}.pk')
    if os.path.exists(pk_path):
        log('Found existing file! Loading progress...')
        resume_dict = pickle_load(pk_path)
        to_save = resume_dict['question']
        start_idx = len(to_save)
    elif dataset_config.dataset_name in OPEN_ENDED_DATASETS and generation_config.rerun is None:
        to_save, start_idx = try_resume_openended_from_jsonl(
            save_config, dataset_config, generation_config
        )

    # Dataloader
    if task_config.task_name == 'cefr':
        dataset = load_dataset(
            "csv",
            data_files={"test": f'{save_config.data_path}/assets/vocab_processed/{dataset_config.dataset_name}_{(task_config.cefr_level).lower()}.csv'},
            split="test",
        )

        if generation_config.rerun is not None:
            rerun_index = list(np.load(generation_config.rerun))
            dataset = dataset.select(rerun_index)
        elif start_idx > 0:
            dataset = dataset.select(range(start_idx, len(dataset)))

        dataloader = DataLoader(dataset, generation_config.batch_size, shuffle=False)

    elif task_config.task_name == 'L1':
        # Benchmark L1 mode (depends on CEFR CSV assets)
        dataset = load_dataset(
            "csv",
            data_files={"test": f'{save_config.data_path}/assets/cefr/{dataset_config.dataset_name}/{task_config.cefr_level}.csv'},
            split='test',
        )

        if generation_config.rerun is not None:
            rerun_index = list(np.load(generation_config.rerun))
            dataset = dataset.select(rerun_index)
        elif start_idx > 0:
            dataset = dataset.select(range(start_idx, len(dataset)))

        dataloader = DataLoader(dataset, generation_config.batch_size, shuffle=False)

    elif task_config.task_name in ['english_dialect', 'openended_cefr', 'openended_l1', 'openended_esl']:
        # Open-ended / dialect mode (jsonl datasets) — no CEFR assets needed
        dataloader = return_dataloader(
            dataset_config=dataset_config,
            generation_config=generation_config,
            start_idx=start_idx
        )
    else:
        raise NotImplementedError(f"Unknown task_name: {task_config.task_name}")

    log("Dataset ready. Starting transformation (first batch may take 1–3 min)...")
    if generation_config.skip_semantic_check:
        log(colorstr('yellow', 'Semantic checker (SS) is OFF — Trans-EnV paper default is ON (meaning must be preserved).'))
    use_semantic_check = not generation_config.skip_semantic_check
    # Sampling Parameters
    sampling_params = {
        'temperature': generation_config.temperature,
        'top_p': generation_config.top_p,
        'max_tokens': generation_config.max_tokens,
    }

    for it, sample in enumerate(tqdm(dataloader)):
        key = QUESTION_KEY_ID[dataset_config.dataset_name]

        if dataset_config.dataset_name == 'mt-bench':
            # batch_turns: list[list[str]]
            batch_turns = sample[key]

            if generation_config.passthrough_mt_bench:
                # Do not transform: keep original prompts (ESL rules mangle instructions).
                iter_result = []
                for ex_i, turns in enumerate(batch_turns):
                    kept = ["" if t is None else re.sub(r'_{2,}', '<blank>', t) for t in turns]
                    iter_result.append({
                        'orig_sentence': batch_turns[ex_i],
                        'whole_response': [],
                        'mid_transformed_sentences': [],
                        'judge_repsonse': [],
                        'applied_rules': [],
                        'transformed_sentences': [],
                        'final_sentence': "\n".join(kept),
                        'final_turns': kept,
                    })
            else:
                # One transformation() call per turn (batch size 1), same pathway as close-ended vLLM:
                # each turn gets its own guideline shuffle and rule loop — no shared multi-turn batch.
                flat_results = []
                owner = []  # (example_index, turn_index)
                for ex_i, turns in enumerate(batch_turns):
                    for t_i, t in enumerate(turns):
                        t = "" if t is None else t
                        t = re.sub(r'_{2,}', '<blank>', t)
                        owner.append((ex_i, t_i))
                        if use_openai_api:
                            flat_results.append(
                                openai_transformation(
                                    [t],
                                    guideline,
                                    client,
                                    sampling_params,
                                    task_config,
                                    model_config,
                                    generation_config.one_transform,
                                    generation_config.max_rules,
                                    generation_config.max_workers,
                                    generation_config.max_chain_depth,
                                    use_semantic_check=use_semantic_check,
                                )[0]
                            )
                        else:
                            flat_results.append(
                                transformation(
                                    [t],
                                    guideline,
                                    client,
                                    tokenizer,
                                    sampling_params,
                                    task_config,
                                    model_config,
                                    generation_config.one_transform,
                                    generation_config.max_rules,
                                    generation_config.max_chain_depth,
                                    use_semantic_check=use_semantic_check,
                                )[0]
                            )

                # Regroup final_sentence and applied_rules per (ex_i, t_i)
                regrouped = [[] for _ in range(len(batch_turns))]
                regrouped_rules = [[] for _ in range(len(batch_turns))]
                for (ex_i, t_i), out in zip(owner, flat_results):
                    while len(regrouped[ex_i]) <= t_i:
                        regrouped[ex_i].append("")
                        regrouped_rules[ex_i].append([])
                    regrouped[ex_i][t_i] = out['final_sentence']
                    regrouped_rules[ex_i][t_i] = out.get('applied_rules', out.get('applied_rule', []))

                # Store one record per example; one transformed string + rule list per turn (parallel to turns / turns_transformed).
                iter_result = []
                for ex_i in range(len(batch_turns)):
                    iter_result.append({
                        'orig_sentence': batch_turns[ex_i],
                        'whole_response': [],
                        'mid_transformed_sentences': [],
                        'judge_repsonse': [],
                        'applied_rules': regrouped_rules[ex_i],
                        'transformed_sentences': [],
                        'final_sentence': "\n".join(regrouped[ex_i]),
                        'final_turns': regrouped[ex_i],
                    })

        else:
            sentence = sample[key]
            # sanitize None
            sentence = [("" if s is None else s) for s in sentence]
            sentence = [re.sub(r'_{2,}', '<blank>', s) for s in sentence]

            # ifeval / alpacafarm: one sentence per transformation() — same vLLM pathway as a single
            # close-ended item (own guideline shuffle per row). Benchmark cefr/L1 stays batched.
            openended_jsonl = dataset_config.dataset_name in {"ifeval", "alpacafarm"}

            if openended_jsonl:
                iter_result = []
                for single in sentence:
                    if use_openai_api or (model_config.model_name.split('/')[0] == 'azure'):
                        iter_result.append(
                            openai_transformation(
                                [single],
                                guideline,
                                client,
                                sampling_params,
                                task_config,
                                model_config,
                                generation_config.one_transform,
                                generation_config.max_rules,
                                generation_config.max_workers,
                                generation_config.max_chain_depth,
                                use_semantic_check=use_semantic_check,
                            )[0]
                        )
                    else:
                        iter_result.append(
                            transformation(
                                [single],
                                guideline,
                                client,
                                tokenizer,
                                sampling_params,
                                task_config,
                                model_config,
                                generation_config.one_transform,
                                generation_config.max_rules,
                                generation_config.max_chain_depth,
                                use_semantic_check=use_semantic_check,
                            )[0]
                        )
            elif use_openai_api or (model_config.model_name.split('/')[0] == 'azure'):
                iter_result = openai_transformation(
                    sentence,
                    guideline,
                    client,
                    sampling_params,
                    task_config,
                    model_config,
                    generation_config.one_transform,
                    generation_config.max_rules,
                    generation_config.max_workers,
                    generation_config.max_chain_depth,
                    use_semantic_check=use_semantic_check,
                )
            else:
                iter_result = transformation(
                    sentence,
                    guideline,
                    client,
                    tokenizer,
                    sampling_params,
                    task_config,
                    model_config,
                    generation_config.one_transform,
                    generation_config.max_rules,
                    generation_config.max_chain_depth,
                    use_semantic_check=use_semantic_check,
                )

        to_save.extend(iter_result)

        if dataset_config.dataset_name in choice_transform_dataset:
            # choices transform (kept as-is; may require dataset-specific handling)
            for choice_num, sentence in enumerate(sample['choices']['text']):
                if use_openai_api or (model_config.model_name.split('/')[0] == 'azure'):
                    iter_choice = openai_transformation(
                        sentence, guideline, client, sampling_params, task_config, model_config,
                        generation_config.one_transform,
                        generation_config.max_rules,
                        generation_config.max_workers,
                        generation_config.max_chain_depth,
                        use_semantic_check=use_semantic_check,
                    )
                else:
                    iter_choice = transformation(
                        sentence, guideline, client, tokenizer, sampling_params, task_config, model_config,
                        generation_config.one_transform,
                        generation_config.max_rules,
                        generation_config.max_chain_depth,
                        use_semantic_check=use_semantic_check,
                    )
                to_save_choice[choice_num].extend(iter_choice)

            to_save_dict = {
                'question': to_save,
                'choices': to_save_choice
            }
        else:
            to_save_dict = {'question': to_save}

        if generation_config.rerun is None:
            pickle_save(os.path.join(save_config.save_path, f'{save_config.file_name}.pk'), to_save_dict)
        else:
            pickle_save(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.pk'), to_save_dict)

        save_func(to_save_dict, save_config, dataset_config, generation_config, task_config)


if __name__ == "__main__":
    main()