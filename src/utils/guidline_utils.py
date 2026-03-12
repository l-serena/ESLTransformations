import re
from utils import log


def extract_transformed_sentence(text):
    """
    Extract the transformed sentence from the provided structured text.

    Returns:
        str: transformed sentence if present, else 'No change'
    """
    pattern = r"\*\*Transformed Sentence:\*\* (.*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return "No change"


def extract_guideline_l1(guideline):
    guideline_instruction = guideline.split('---')[0]

    example = ''.join(guideline.split('---')[1:])
    example = example.replace('**Input Sentence 1:**', 'Input Sentence 1:')
    example = example.replace('**Input Sentence 2:**', 'Input Sentence 2:')
    example = example.replace('**Final broken sentence:**', '**Transformed Sentence:**')

    example1_input_sentence = example.split('Input Sentence 1:')[1].split('\n')[0].strip().strip('"')
    example1_output = ''.join(example.split('Input Sentence 1:')[1].split('\n')[1:]).split('#### Input Sentence 2:')[0]
    example1_output = example1_output.replace('Phase 2: ', '')
    example1_output = example1_output.replace('Actionable Changes:', 'Actionable Changes')

    example2_input_sentence = example.split('Input Sentence 2:')[1].split('\n')[0].strip().strip('"')
    example2_output = ''.join(example.split('Input Sentence 2:')[1].split('\n')[1:])

    few_shot_example = [
        {"input": '**Original Sentence:**' + example1_input_sentence, "output": example1_output},
        {"input": '**Original Sentence:**' + example2_input_sentence, "output": example2_output},
    ]

    return guideline_instruction, few_shot_example


def extract_guideline_dialect(guideline):
    """
    Parses the original dialect/cefr guideline JSON format (the one with '### Example' blocks).
    """
    guideline_instruction = guideline.split('\n### Example\n')[0]

    # example block
    example_block = guideline.split('\n### Example\n')[1][1:]
    example1_input_sentence = example_block.split('\n\n')[0]

    example1_output = '\n\n'.join(example_block.split('\n\n')[1:-1])

    few_shot_example = [
        {"input": example1_input_sentence, "output": example1_output}
    ]

    return guideline_instruction, few_shot_example


def extract_guideline_openended_cefr(guideline: str):
    """
    Parses the built-in openended_cefr guideline strings you defined in framework/guideline.py.

    Expected structure (flexible):
      Feature: ...
      Identification questions:
      ...
      Actionable changes:
      ...
      Example:
      Input: ...
      Output: **Transformed Sentence:** ...
    """
    # Use the whole guideline as instruction (it includes identification + actionable rules)
    guideline_instruction = guideline.strip()

    # Extract Example Input/Output (required by framework_application)
    # We'll be tolerant to spacing/casing.
    input_match = re.search(r"(?im)^Input:\s*(.*)\s*$", guideline)
    output_match = re.search(r"(?im)^Output:\s*(.*)\s*$", guideline)

    if not input_match or not output_match:
        log("openended_cefr guideline missing Example Input/Output. Using fallback dummy example.", level="error")
        few_shot_example = [{
            "input": "**Original Sentence:** hello",
            "output": "**Transformed Sentence:** hello"
        }]
        return guideline_instruction, few_shot_example

    ex_in = input_match.group(1).strip()
    ex_out = output_match.group(1).strip()

    # Your framework_application expects example[0]['input'] and example[0]['output']
    few_shot_example = [{
        "input": ex_in,
        "output": ex_out
    }]

    return guideline_instruction, few_shot_example


def _pick_extractor_for_guideline_text(guideline: str):
    """
    Auto-detect the guideline format from its text.
    This is needed when a task mixes multiple guideline sources (e.g., CEFR + L1 for ESL varieties).
    """
    if guideline is None:
        return extract_guideline_dialect

    # L1 guideline file uses '---' separators + "Input Sentence 1:"
    if ('---' in guideline) and ('Input Sentence 1:' in guideline or '**Input Sentence 1:**' in guideline):
        return extract_guideline_l1

    # Original dialect/cefr guideline JSON uses '### Example'
    if '\n### Example\n' in guideline:
        return extract_guideline_dialect

    # Built-in openended_cefr guidelines use "Input:" / "Output:"
    if re.search(r"(?im)^Input:\s*", guideline) and re.search(r"(?im)^Output:\s*", guideline):
        return extract_guideline_openended_cefr

    # Fallback: treat as dialect-style (instruction-only) if nothing else matches
    return extract_guideline_dialect


def extract_guideline_examples(guideline, task):
    """
    guideline is a string (guideline text).
    task selects the parser.
    """
    if task in {'openended_esl'}:
        extract_func = _pick_extractor_for_guideline_text(guideline)
    elif task == 'L1':
        extract_func = extract_guideline_l1
    elif task == 'openended_l1':
        # Reuse existing L1 guideline format (python_grammar_error.json)
        extract_func = extract_guideline_l1
    elif task == 'english_dialect':
        extract_func = extract_guideline_dialect
    elif task == 'cefr':
        extract_func = extract_guideline_dialect
    elif task == 'openended_cefr':
        # openended_cefr now uses the same generated guideline JSON format ("### Example") as benchmark CEFR.
        extract_func = extract_guideline_dialect
    else:
        log(f'Please double check your taks, we got {task}', level='error')
        raise NotImplementedError

    return extract_func(guideline)