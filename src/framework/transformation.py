import copy
import random

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

    for i in range(len(rules)):
        if one_transform and all(done):
            break
        feature = rules[i][0]
        input_prompt = framework_application(guideline=rules[i], task=task_config.task_name)

        batch_input = [
            input_prompt + [{"role": 'user', "content": f"**Original Sentence:** {s}"}]
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

                if 'no' in semantic_response.choices[0].message.content.lower():
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



def openai_transformation(sentence, guideline, client, sampling_params, task_config, model_config, one_transform: bool = False, max_rules: int = 0):
    """
    sentence: list[str] where list size equals batch size
    Returns: list[dict] with one result per input sentence (same shape as transformation()).
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

    for i in range(len(rules)):
        if one_transform and all(done):
            break
        feature = rules[i][0]

        for ex_i in range(len(sentence)):
            if one_transform and done[ex_i]:
                continue
            input_prompt = openai_framework_application(guideline=rules[i], task=task_config.task_name)
            input_prompt = input_prompt + [{"role": "user", "content": f"**Original Sentence:** {sentence[ex_i]}"}]

            try:
                responses = client.chat.completions.create(
                    model=model_config.model_name,
                    messages=input_prompt,
                    **sampling_params
                )

                response = responses.choices[0].message.content
                whole_responses[ex_i].append(response)

                if response is None:
                    continue

                transformed_sentence = extract_transformed_sentence(response)

                if _should_accept_transformation(orig_sentences[ex_i], transformed_sentence, response):
                    sentence[ex_i] = transformed_sentence
                    applied_rules[ex_i].append(feature)
                    transformed_sentences[ex_i].append(transformed_sentence)
                    if one_transform:
                        done[ex_i] = True

            except Exception as e:
                # Record the error so runs don't look like "nothing happened".
                whole_responses[ex_i].append(f"__EXCEPTION__: {type(e).__name__}: {e}")
                exception_counts[ex_i] += 1
                continue

    # If a given example errored on every single rule, keep original sentence
    for ex_i in range(len(sentence)):
        if exception_counts[ex_i] == len(guideline):
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
