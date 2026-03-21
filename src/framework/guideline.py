import os
import pandas as pd

from registry.guidline import *
from registry.framework import CEFR_GRAMMAR_FEATURE_PATH
from utils import colorstr
from utils.filesys_utils import json_load


def _l1_key_for_registry(l1: str) -> str:
    """Map TaskConfig / CLI L1 label to keys in L1_GRAMMARERROR (python_grammar_error.json)."""
    if l1 is None:
        raise ValueError("l1 is required")
    # Config uses "Mandarin"; registry / paper corpora use "Chinese-Mandarin"
    aliases = {"Mandarin": "Chinese-Mandarin"}
    return aliases.get(l1, l1)


def dialect_feature(dialect, data_path):
    ewave = pd.read_csv(os.path.join(data_path, 'ewave/ewave.csv'))
    linguistic_features = ewave[(ewave['Language_ID'] == dialect) & (ewave['Value'] == 'A')]['Parameter_ID'].tolist()
    return linguistic_features


def cefr_feature(cefr_level):
    online_profile = pd.read_excel(CEFR_GRAMMAR_FEATURE_PATH)

    mapping = {
        'A1': 'A', 'A2': 'A',
        'B1': 'B', 'B2': 'B',
        'C1': 'C', 'C2': 'C'
    }

    online_profile['Level'] = online_profile['Level'].replace(mapping)

    levels = ['A', 'B', 'C']
    input_index = levels.index(cefr_level)
    filtered_levels = levels[input_index + 1:]

    linguistic_features = online_profile[online_profile['Level'].isin(filtered_levels)]['Can-do statement'].tolist()

    return linguistic_features


def _builtin_openended_cefr_guidelines(cefr_level: str):
    """
    Built-in CEFR-like rewrite guidelines that require NO external files.
    Returns: list[(feature_name, guideline_text)]
    """
    lvl = (cefr_level or "A").upper()

    # Keep this small & safe: rules that usually preserve meaning.
    # You can expand later.
    if lvl == "A":
        return [
            ("short_sentences",
             """
Feature: Short sentences

Identification questions:
1) Is the sentence long (e.g., contains multiple clauses joined by commas/semicolons/and/but/or)?
2) Can it be split into 2 shorter sentences without losing information?

Actionable changes:
- Split into shorter sentences.
- Keep all key info, numbers, named entities.
- Do not change <blank>.
Example:
Input: John went to the store, and he bought milk.
Output: **Transformed Sentence:** John went to the store. He bought milk.
"""),
            ("simple_words",
             """
Feature: Simple words

Identification questions:
1) Are there advanced words that have a simpler common alternative (e.g., "purchase" -> "buy")?
2) Can they be replaced without losing meaning?

Actionable changes:
- Replace only where obvious and safe.
- Keep keywords and numbers.
- Do not change <blank>.
Example:
Input: She purchased a book.
Output: **Transformed Sentence:** She bought a book.
"""),
        ]

    if lvl == "B":
        return [
            ("light_simplification",
             """
Feature: Light simplification

Identification questions:
1) Is there at least one complex/rare phrasing that can be made more direct?
2) Can you keep the same meaning and structure mostly intact?

Actionable changes:
- Make small clarity edits only.
- Keep all key info, numbers, named entities.
- Do not change <blank>.
Example:
Input: Determine whether the claim is valid.
Output: **Transformed Sentence:** Decide if the claim is valid.
"""),
            ("contractions_optional",
             """
Feature: Optional contractions

Identification questions:
1) Does the sentence contain "do not/does not/did not/cannot/I am/it is/they are"?
2) Can one contraction be used safely?

Actionable changes:
- Apply at most ONE contraction.
- Do not change <blank>.
Example:
Input: I am sure it is correct.
Output: **Transformed Sentence:** I'm sure it is correct.
"""),
        ]

    # "C" or fallback
    return [
        ("minimal_change",
         """
Feature: Minimal change (CEFR-C)

Identification questions:
1) Can you rephrase slightly to be more formal/precise without changing meaning?

Actionable changes:
- Make minimal stylistic improvements only.
- Keep all key info, numbers, named entities.
- Do not change <blank>.
Example:
Input: Tell me what this means.
Output: **Transformed Sentence:** Please explain what this means.
""")
    ]


def return_guideline(task_config, dataset_name, data_path):
    # -----------------------------
    # Open-ended modes (no external assets)
    # -----------------------------

    # open-ended CEFR: use the SAME extracted CEFR feature set + guideline JSON
    # as the original benchmark CEFR pipeline, but applied to open-ended datasets.
    if (task_config.task_name == "openended_cefr") and (task_config.cefr_level is not None):
        cefr_file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example_grammar_error.json')
        guideline_json = json_load(cefr_file_path)
        cefr_linguistic_features = cefr_feature(task_config.cefr_level)
        guideline = [(g['feature'][1:-1].strip(), g['guideline']) for g in guideline_json if g['feature'][1:-1].strip() in cefr_linguistic_features]
        assert len(guideline) != 0, colorstr("red", "Guideline Empty!")
        return guideline

    # open-ended L1: reuse existing L1 rules WITHOUT requiring CEFR datasets
    if (task_config.task_name == "openended_l1") and (task_config.l1 is not None):
        l1_file_path = os.path.join(data_path, 'assets/guidelines/python_grammar_error.json')
        guideline_json = json_load(l1_file_path)
        l1_linguistic_features = L1_GRAMMARERROR[_l1_key_for_registry(task_config.l1)]
        guideline = [(g['grammar_error'], g['guideline']) for g in guideline_json if g['grammar_error'] in l1_linguistic_features]
        assert len(guideline) != 0, colorstr("red", "Guideline Empty!")
        return guideline

    # open-ended ESL variety: CEFR + L1 combined
    # Feature set = CEFR-level features + L1-specific features (as in Trans-EnV "ESL English" varieties).
    if (task_config.task_name == "openended_esl") and (task_config.cefr_level is not None) and (task_config.l1 is not None):
        # CEFR component (same as benchmark CEFR guideline generation file)
        cefr_file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example_grammar_error.json')
        guideline_json = json_load(cefr_file_path)
        cefr_linguistic_features = cefr_feature(task_config.cefr_level)
        cefr_guidelines = [(g['feature'][1:-1].strip(), g['guideline']) for g in guideline_json if g['feature'][1:-1].strip() in cefr_linguistic_features]

        # L1 component (same L1 guideline file and feature list as the benchmark L1 pipeline)
        l1_file_path = os.path.join(data_path, 'assets/guidelines/python_grammar_error.json')
        guideline_json = json_load(l1_file_path)
        l1_linguistic_features = L1_GRAMMARERROR[_l1_key_for_registry(task_config.l1)]
        l1_guidelines = [(g['grammar_error'], g['guideline']) for g in guideline_json if g['grammar_error'] in l1_linguistic_features]

        guideline = list(cefr_guidelines) + list(l1_guidelines)
        assert len(guideline) != 0, colorstr("red", "Guideline Empty!")
        return guideline

    # -----------------------------
    # Original benchmark modes
    # -----------------------------

    if (task_config.task_name == 'english_dialect') & (task_config.dialect is not None):
        file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example.json')
        guideline_json = json_load(file_path)

        linguistic_features = dialect_feature(dialect=task_config.dialect.strip("'\""), data_path=data_path)

        if dataset_name in ['mmlu', 'hellaswag']:
            linguistic_features = [l for l in linguistic_features if l in DIALECT_FEATURE_LIST]

        guideline = [(g['feature'][3:-3], g['guideline']) for g in guideline_json if g['feature'][3:-3] in linguistic_features]

    elif (task_config.task_name == 'L1') & (task_config.l1 is not None):
        # Benchmark L1: Trans-EnV paper — ESL variety ℒ_vi combines CEFR-level features and L1-specific
        # features at that proficiency (e.g. CEFR A + Arabic L1). When cefr_level is set, merge both pools.
        l1_file_path = os.path.join(data_path, 'assets/guidelines/python_grammar_error.json')
        guideline_json = json_load(l1_file_path)
        l1_linguistic_features = L1_GRAMMARERROR[_l1_key_for_registry(task_config.l1)]
        l1_guidelines = [(g['grammar_error'], g['guideline']) for g in guideline_json if g['grammar_error'] in l1_linguistic_features]

        if task_config.cefr_level is not None:
            cefr_file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example_grammar_error.json')
            cefr_json = json_load(cefr_file_path)
            cefr_linguistic_features = cefr_feature(task_config.cefr_level)
            cefr_guidelines = [
                (g['feature'][1:-1].strip(), g['guideline'])
                for g in cefr_json
                if g['feature'][1:-1].strip() in cefr_linguistic_features
            ]
            guideline = list(cefr_guidelines) + list(l1_guidelines)
        else:
            guideline = l1_guidelines

    elif (task_config.task_name == 'cefr') & (task_config.cefr_level is not None):
        cefr_file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example_grammar_error.json')
        guideline_json = json_load(cefr_file_path)

        cefr_linguistic_features = cefr_feature(task_config.cefr_level)
        guideline = [(g['feature'][1:-1].strip(), g['guideline']) for g in guideline_json if g['feature'][1:-1].strip() in cefr_linguistic_features]

    else:
        raise NotImplementedError(f'Please double check the task name, we got {task_config.task_name}')

    assert len(guideline) != 0, colorstr("red", "Guideline Empty!")
    return guideline