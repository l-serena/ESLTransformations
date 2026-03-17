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


# Phrases that often indicate model hallucination (template output instead of preserving the original).
_HALLUCINATION_PREFIXES = (
    "the fact is that",
    "the problem is that",
    "the thing is that",
    "the one that",
    "the point is that",
    "it was said that",
    "none ",
    "some were not",
    "some get some",
    "some find it",
    "i thinking that",
    "i am wondering if",
    "i am suggesting not",
)


def _should_accept_transformation(orig: str, transformed: str, raw_response: str) -> bool:
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

    rr = (raw_response or "").lower()
    if any(p in rr for p in (
        "not applicable",
        "no transformation is required",
        "doesn't meet the criteria",
        "does not meet the criteria",
        "cannot be transformed according to the guidelines",
        "therefore, no transformation is required",
    )):
        return False

    # Anti-hallucination: for long/original text, reject if output collapsed or is template-like.
    len_orig = len(o)
    len_t = len(t)
    if len_orig > 150:
        if len_t < 0.5 * len_orig:
            return False
        t_lower = t.lower()
        if any(t_lower.startswith(p) or (" " + p) in t_lower for p in _HALLUCINATION_PREFIXES):
            return False

    return True


def _long_input_instruction(s: str, task_name: str) -> str:
    """For long or instruction-like inputs, ask the model to preserve full text and avoid template phrases."""
    if task_name not in ("openended_esl", "openended_cefr", "openended_l1"):
        return ""
    if len(s) <= 200:
        return ""
    return (
        "\n\n[Important: The text above is long or is an instruction. Your output must be the FULL text with only "
        "the minimal grammatical change from the guideline. Do not summarize, shorten, or replace with a single "
        "clause or template phrase (e.g. avoid starting with 'The fact is that' or 'The problem is that').]"
    )


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



def transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config, one_transform: bool = False, max_rules: int = 0):
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
        feature = rules[i][0]
        input_prompt = framework_application(guideline=rules[i], task=task_config.task_name)

        batch_input = [
            input_prompt + [{"role": "user", "content": f"**Original Sentence:** {s}" + _long_input_instruction(s, task_config.task_name)}]
            for s in sentence
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
        
        for num, response in enumerate(responses.choices):
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

                # Accept only if semantic judge says meaning preserved AND not an obvious hallucination
                if "no" in semantic_response.choices[0].message.content.lower() and _should_accept_transformation(
                    orig_sentence[num], transformed_sentence, response.text
                ):
                    sentence[num] = transformed_sentence
                    applied_rules[num].append(feature)
                    transformed_sentences[num].append(transformed_sentence)
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


def openai_transformation(sentence, guideline, client, sampling_params, task_config, model_config, one_transform: bool = False, max_rules: int = 0, max_workers: int = 10):
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

        # Build work items: (ex_i, messages) for examples not yet done
        work = []
        for ex_i in range(len(sentence)):
            if one_transform and done[ex_i]:
                continue
            content = f"**Original Sentence:** {sentence[ex_i]}" + _long_input_instruction(sentence[ex_i], task_config.task_name)
            messages = base_prompt + [{"role": "user", "content": content}]
            work.append((ex_i, messages))

        if not work:
            continue

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
                    if _should_accept_transformation(orig_sentences[ex_i], transformed_sentence, response):
                        sentence[ex_i] = transformed_sentence
                        applied_rules[ex_i].append(feature)
                        transformed_sentences[ex_i].append(transformed_sentence)
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
                if _should_accept_transformation(orig_sentences[ex_i], transformed_sentence, response):
                    sentence[ex_i] = transformed_sentence
                    applied_rules[ex_i].append(feature)
                    transformed_sentences[ex_i].append(transformed_sentence)
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
