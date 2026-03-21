import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from registry.prompt import (
    OPEN_ENDED_INSTRUCTION_NOTE,
    return_actionable_system_message,
    return_system_message,
    semantic_check,
)
from utils.guidline_utils import *

OPEN_ENDED_TASKS = ("openended_esl", "openended_cefr", "openended_l1")


def _user_content_for_sentence(sentence_text: str, task_name: str) -> str:
    """Build the user message content: for open-ended tasks, stress rephrasing the full instruction only."""
    if task_name in OPEN_ENDED_TASKS:
        return OPEN_ENDED_INSTRUCTION_NOTE.strip() + "\n\n**Original instruction (full task text):**\n" + sentence_text
    return f"**Original Sentence:** {sentence_text}"


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


# Trans-EnV paper: semantic checker (SS) runs after each candidate transform; only keep if meaning preserved.
_SEMANTIC_JUDGE_MAX_TOKENS = 24


def _semantic_meaning_preserved(judge_response: str) -> bool:
    """
    SS answers whether meaning of Sentence 1 is significantly altered/lost in Sentence 2.
    'yes' -> altered/lost -> reject candidate. 'no' -> preserved -> accept.
    """
    if judge_response is None or not str(judge_response).strip():
        return False
    text = str(judge_response).strip().lower()
    first_line = text.split("\n")[0].strip()
    parts = first_line.replace(",", " ").split()
    first_token = parts[0] if parts else ""
    first_token = first_token.strip(".,;:!?\"'")
    if first_token.startswith("no"):
        return True
    if first_token.startswith("yes"):
        return False
    return False


def _run_semantic_checker(client, model_name: str, orig: str, transformed: str) -> tuple[str, bool]:
    """Call SS; return (raw_judge_text, meaning_preserved). On failure, reject candidate."""
    prompt = semantic_check(orig, transformed)
    try:
        r = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=_SEMANTIC_JUDGE_MAX_TOKENS,
        )
        content = (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"__SEMANTIC_ERROR__: {type(e).__name__}: {e}", False
    return content, _semantic_meaning_preserved(content)


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
    use_semantic_check: bool = True,
):
    """
    sentence (list of string) where list size is equal to batch size.
    use_semantic_check: if True (default), match Trans-EnV: SS must approve meaning preservation
    before a transform is retained (paper: only passing SS are used in subsequent steps).
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

    # Fresh copy + shuffle each call so caller's guideline list is not mutated and each
    # run (e.g. one sentence like close-ended) matches the same logic as benchmark batches.
    guideline_work = copy.deepcopy(guideline)
    random.shuffle(guideline_work)

    rules = guideline_work
    if max_rules and max_rules > 0:
        rules = guideline_work[:max_rules]

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
            input_prompt + [{"role": "user", "content": _user_content_for_sentence(sentence[j], task_config.task_name)}]
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
            if transformed_sentence is None or ("no change" in transformed_sentence.lower()):
                continue

            # save candidate (even if SS later rejects — for debugging)
            mid_transformed_sentences[num].append(transformed_sentence)

            if not _should_accept_transformation(orig_sentence[num], transformed_sentence, response.text):
                judge_responses[num].append("(not_candidate)")
                continue

            if use_semantic_check:
                judge_raw, preserved = _run_semantic_checker(
                    client, model_config.model_name, orig_sentence[num], transformed_sentence
                )
                judge_responses[num].append(judge_raw)
                if not preserved:
                    continue
            else:
                judge_responses[num].append("(semantic_check_off)")

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
    use_semantic_check: bool = True,
):
    """
    sentence: list[str] where list size equals batch size
    Returns: list[dict] with one result per input sentence (same shape as transformation()).
    max_workers: run up to this many API calls in parallel per rule (1 = sequential).
    use_semantic_check: Trans-EnV SS gate (default on); set False to match legacy behavior.
    """
    if type(sentence) is tuple:
        sentence = list(sentence)

    orig_sentences = copy.deepcopy(sentence)

    # Per-example accumulators
    whole_responses = [[] for _ in range(len(sentence))]
    applied_rules = [[] for _ in range(len(sentence))]
    transformed_sentences = [[] for _ in range(len(sentence))]
    mid_transformed_sentences = [[] for _ in range(len(sentence))]
    judge_responses = [[] for _ in range(len(sentence))]

    guideline_work = copy.deepcopy(guideline)
    random.shuffle(guideline_work)

    exception_counts = [0 for _ in range(len(sentence))]
    done = [False for _ in range(len(sentence))]
    chain_count = [0 for _ in range(len(sentence))]

    rules = guideline_work
    if max_rules and max_rules > 0:
        rules = guideline_work[:max_rules]

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
            messages = base_prompt + [{"role": "user", "content": _user_content_for_sentence(sentence[ex_i], task_config.task_name)}]
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
                    mid_transformed_sentences[ex_i].append(transformed_sentence)
                    if use_semantic_check:
                        judge_raw, preserved = _run_semantic_checker(
                            client, model_config.model_name, orig_sentences[ex_i], transformed_sentence
                        )
                        judge_responses[ex_i].append(judge_raw)
                        if not preserved:
                            continue
                    else:
                        judge_responses[ex_i].append("(semantic_check_off)")
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
                mid_transformed_sentences[ex_i].append(transformed_sentence)
                if use_semantic_check:
                    judge_raw, preserved = _run_semantic_checker(
                        client, model_config.model_name, orig_sentences[ex_i], transformed_sentence
                    )
                    judge_responses[ex_i].append(judge_raw)
                    if not preserved:
                        continue
                else:
                    judge_responses[ex_i].append("(semantic_check_off)")
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
            'mid_transformed_sentences': mid_transformed_sentences[ex_i],
            'judge_repsonse': judge_responses[ex_i],
            'applied_rule': applied_rules[ex_i],
            'applied_rules': applied_rules[ex_i],
            'transformed_sentences': transformed_sentences[ex_i],
            'final_sentence': sentence[ex_i]
        })

    return iter_result
