import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from registry.prompt import *
from utils.guidline_utils import *


def _normalize_for_compare(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in {"'", '"'}):
        s = s[1:-1].strip()
    return " ".join(s.split())


def _should_accept_transformation(orig: str, transformed: str, raw_response: str) -> bool:
    """Accept any non-empty transformation that differs from the original. No extra rejection."""
    if transformed is None:
        return False

    t = _normalize_for_compare(transformed)
    o = _normalize_for_compare(orig)

    if not t:
        return False
    if t.lower() == "no change" or "(no change)" in t.lower():
        return False
    if t == o:
        return False

    return True


def framework_application(guideline, task):
    guideline = guideline[1]

    guideline_instruction, example = extract_guideline_examples(guideline, task)
    # Keep strict identification for CEFR/ESL guidelines to avoid false positives.
    # For openended_l1 we keep actionable-only to encourage edits when possible.
    if task in {"openended_l1"}:
        system_message = return_actionable_system_message(guideline_instruction)
    else:
        system_message = return_system_message(guideline_instruction)

    message = [
        {"role": "user", "content": system_message},
        {"role": "assistant", "content": "Well Understood."},
        {"role": "user", "content": example[0]['input']},
        {"role": "assistant", "content": example[0]['output']}
        ]

    return message



def transformation(
    sentence,
    guideline,
    client,
    tokenizer,
    sampling_params,
    task_config,
    model_config,
    one_transform: bool = False,
    max_rules: int = 0,
    max_chain_depth: int = 0,
):
    """
    sentence (list of string) where list size is equal to batch size
    """

    if type(sentence) is tuple:     # tuple이면 list로 바꾸기
        sentence = list(sentence)

    orig_sentence = copy.deepcopy(sentence)

    whole_responses = [[] for _ in range(len(sentence))]
    applied_rules = [[] for _ in range(len(sentence))]  # rules that are answered yes to all identification questions
    mid_transformed_sentences = [[] for _ in range(len(sentence))] # transformed sentences that are transformed by applied rules
    judge_responses = [[] for _ in range(len(sentence))] # judge response to each transformed sentence
    transformed_sentences = [[] for _ in range(len(sentence))] # final transformed sentence
    done = [False for _ in range(len(sentence))]  # stop early if one_transform
    chain_count = [0 for _ in range(len(sentence))]  # successful transforms per sentence

    # shuffle guideline
    random.shuffle(guideline)

    rules = guideline
    if max_rules and max_rules > 0:
        rules = guideline[:max_rules]

    total_rules = len(rules)
    rule_iter = tqdm(range(total_rules), desc="Transformations", unit="rule")
    for i in rule_iter:
        rule_iter.set_postfix_str(f"{i + 1}/{total_rules}")
        if one_transform and all(done):
            break
        n = len(sentence)
        active = [
            j
            for j in range(n)
            if not (one_transform and done[j])
            and (max_chain_depth == 0 or chain_count[j] < max_chain_depth)
        ]
        if not active:
            break

        feature = rules[i][0]
        input_prompt = framework_application(guideline=rules[i], task=task_config.task_name)

        batch_input = [
            input_prompt + [{"role": "user", "content": f"**Original Sentence:** {sentence[j]}"}]
            for j in active
        ]
        chat_batch_input = list()

        for input in batch_input:
            text = tokenizer.apply_chat_template(
                input,
                tokenize=False,
                add_generation_prompt=True
            )
            chat_batch_input.append(text)

        responses = client.completions.create(
            model=model_config.model_name,
            prompt=chat_batch_input,
            **sampling_params
            )
        
        for k, num in enumerate(active):
            response = responses.choices[k]
            # save all responses
            whole_responses[num].append(response.text)

            if response.text is None:
                continue

            transformed_sentence = extract_transformed_sentence(response.text)
            if ('no change' in transformed_sentence.lower()) or transformed_sentence.lower() is None:
                continue

            else:
                # save the transformed sentences
                mid_transformed_sentences[num].append(transformed_sentence)

                semantic_input_prompt = semantic_check(orig_sentence[num], transformed_sentence)
                semantic_response = client.chat.completions.create(
                    model=model_config.model_name,
                    messages=[{'role': 'user', 'content': semantic_input_prompt}]
                )

                # save judge response
                judge_responses[num].append(semantic_response.choices[0].message.content.lower())

                # Accept whenever we have a valid transformation (no rejection from judge)
                if _should_accept_transformation(orig_sentence[num], transformed_sentence, response.text):
                    sentence[num] = transformed_sentence
                    applied_rules[num].append(feature)
                    transformed_sentences[num].append(transformed_sentence)
                    chain_count[num] += 1
                    if one_transform:
                        done[num] = True
        
    iter_result = list()

    for num in range(len(sentence)):
        iter_result.append({
            'orig_sentence': orig_sentence[num],
            'whole_response': whole_responses[num],
            'mid_transformed_sentences': mid_transformed_sentences[num],
            'judge_repsonse': judge_responses[num],
            'applied_rules': applied_rules[num],
            'transformed_sentences': transformed_sentences[num],
            'final_sentence': sentence[num]
        })

    return iter_result



def openai_framework_application(guideline, task):
    guideline = guideline[1]

    guideline_instruction, example = extract_guideline_examples(guideline, task)
    if task in {"openended_l1"}:
        system_message = return_actionable_system_message(guideline_instruction)
    else:
        system_message = return_system_message(guideline_instruction)

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example[0]['input']},
        {"role": "assistant", "content": example[0]['output']}
    ]

    return message



def _openai_transform_one(client, model_name, sampling_params, ex_i, messages, orig_sentence):
    """One API call for (rule, example). Returns (ex_i, response_text, transformed_or_none, exception_or_none)."""
    try:
        responses = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **sampling_params
        )
        response = responses.choices[0].message.content
        transformed = extract_transformed_sentence(response) if response else None
        return (ex_i, response, transformed, None)
    except Exception as e:
        return (ex_i, None, None, e)


def openai_transformation(
    sentence,
    guideline,
    client,
    sampling_params,
    task_config,
    model_config,
    one_transform: bool = False,
    max_rules: int = 0,
    max_workers: int = 10,
    max_chain_depth: int = 0,
):
    """
    sentence: list[str] where list size equals batch size
    Returns: list[dict] with one result per input sentence (same shape as transformation()).
    max_workers: run up to this many API calls in parallel per rule (1 = sequential).
    """
    if type(sentence) is tuple:
        sentence = list(sentence)

    orig_sentences = copy.deepcopy(sentence)

    # Per-example accumulators
    whole_responses = [[] for _ in range(len(sentence))]
    applied_rules = [[] for _ in range(len(sentence))]
    transformed_sentences = [[] for _ in range(len(sentence))]

    # Shuffle once per batch for reproducibility of "which rules get tried first"
    random.shuffle(guideline)

    exception_counts = [0 for _ in range(len(sentence))]
    done = [False for _ in range(len(sentence))]
    chain_count = [0 for _ in range(len(sentence))]

    rules = guideline
    if max_rules and max_rules > 0:
        rules = guideline[:max_rules]

    total_rules = len(rules)
    rule_iter = tqdm(range(total_rules), desc="Transformations", unit="rule")
    parallel = max_workers is not None and max_workers > 1

    for i in rule_iter:
        rule_iter.set_postfix_str(f"{i + 1}/{total_rules}")
        if one_transform and all(done):
            break
        feature = rules[i][0]
        base_prompt = openai_framework_application(guideline=rules[i], task=task_config.task_name)

        # Build work items: (ex_i, messages) for examples not yet done / under chain cap
        work = []
        for ex_i in range(len(sentence)):
            if one_transform and done[ex_i]:
                continue
            if max_chain_depth and chain_count[ex_i] >= max_chain_depth:
                continue
            messages = base_prompt + [{"role": "user", "content": f"**Original Sentence:** {sentence[ex_i]}"}]
            work.append((ex_i, messages))

        if not work:
            break

        if parallel and len(work) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _openai_transform_one,
                        client,
                        model_config.model_name,
                        sampling_params,
                        ex_i,
                        messages,
                        orig_sentences[ex_i],
                    ): ex_i
                    for ex_i, messages in work
                }
                for future in as_completed(futures):
                    ex_i, response, transformed_sentence, err = future.result()
                    if err is not None:
                        whole_responses[ex_i].append(f"__EXCEPTION__: {type(err).__name__}: {err}")
                        exception_counts[ex_i] += 1
                        continue
                    whole_responses[ex_i].append(response)
                    if response is None:
                        continue
                    if not _should_accept_transformation(orig_sentences[ex_i], transformed_sentence, response):
                        continue
                    sentence[ex_i] = transformed_sentence
                    applied_rules[ex_i].append(feature)
                    transformed_sentences[ex_i].append(transformed_sentence)
                    chain_count[ex_i] += 1
                    if one_transform:
                        done[ex_i] = True
        else:
            for ex_i, messages in work:
                ex_i, response, transformed_sentence, err = _openai_transform_one(
                    client, model_config.model_name, sampling_params, ex_i, messages, orig_sentences[ex_i]
                )
                if err is not None:
                    whole_responses[ex_i].append(f"__EXCEPTION__: {type(err).__name__}: {err}")
                    exception_counts[ex_i] += 1
                    continue
                whole_responses[ex_i].append(response)
                if response is None:
                    continue
                if not _should_accept_transformation(orig_sentences[ex_i], transformed_sentence, response):
                    continue
                sentence[ex_i] = transformed_sentence
                applied_rules[ex_i].append(feature)
                transformed_sentences[ex_i].append(transformed_sentence)
                chain_count[ex_i] += 1
                if one_transform:
                    done[ex_i] = True

    # If a given example errored on every single rule, keep original sentence
    for ex_i in range(len(sentence)):
        if exception_counts[ex_i] == len(rules):
            sentence[ex_i] = orig_sentences[ex_i]

    iter_result = []
    for ex_i in range(len(sentence)):
        iter_result.append({
            'orig_sentence': orig_sentences[ex_i],
            'whole_response': whole_responses[ex_i],
            'applied_rule': applied_rules[ex_i],
            'transformed_sentences': transformed_sentences[ex_i],
            'final_sentence': sentence[ex_i]
        })

    return iter_result
