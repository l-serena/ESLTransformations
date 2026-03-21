def return_identification_system_message(guideline_instruction): 
    return f'''
    Your task is to evaluate if the below feature is applicable to the given sentence by following the guideline.

    {guideline_instruction}

    ### Steps to Follow:
    - You must answer **every** identification question in the guideline, in order, before giving a final verdict.
    - For each question, respond on its own line with the question number and exactly "yes" or "no" (e.g. "1. yes", "2. no").
    - After all answers: if every answer is "yes", output "Final answer is 'applicable'"; otherwise output "Final answer is 'inapplicable.'"
    - Do not skip the identification step or jump straight to the final verdict.
    
    ### Mandatory
    - Final line must be either "Final answer is 'applicable'" or "Final answer is 'inapplicable'".
    '''



def return_actionable_system_message(guideline_instruction): 
    return f'''
    Your task is to rephrase the test questions by following the guideline.

    {guideline_instruction}

    - If the guideline includes identification questions, answer each one explicitly (numbered yes/no) before any rewrite; only rewrite if all answers are "yes" per the guideline.
    - Make only the **necessary changes** to apply the linguistic feature, ensuring no loss of information.
    - Provide the final transformed sentence, adhering strictly to the format and structure of the given example.

    ### Mandatory
    - Retain any <blank> without modifications.
    - Preserve the structure of the original sentence as much as possible with no information loss.
    - Follow the guideline, not considering standard English grammar.
    - Final line must start with '**Transformed Sentence:**' (with the transformed text or (No change)).
    '''



# When transforming open-ended prompts (ifeval, mt-bench, alpacafarm), we need the model to
# rephrase the full instruction text and NOT perform the task (e.g. not write the dialogue).
OPEN_ENDED_INSTRUCTION_NOTE = """
Important: You receive ONE instruction only (a single turn). Transform only this text. Do not answer it or perform the task it describes.

The text below is the full task instruction (e.g. "Compose a blog post...", "Rewrite your previous response...").
Your job is to rephrase this entire instruction according to the guideline so the instruction text itself exhibits the linguistic feature.

Preserve the same communicative intent and task type as the original:
- If it is an imperative (Compose, Write, Draft, Rewrite, Ask...), keep it as an imperative-style instruction with the same goal (same topic, format, and constraints).
- If it is already a question to the user, you may keep question form only if the guideline fits; do NOT replace imperatives with unrelated wh-questions.
- Keep all concrete requirements from the original (topics, counts, formats, quoted titles, bullet rules, etc.). Do not drop or replace them with generic wording.

Do NOT:
- Add meta-instructions ("omitting X", "avoid Y in your response") instead of applying the feature in the wording.
- Append extra instructions or constraints beyond what the guideline requires.
- Perform the task (no dialogue, no blog post body, no email body).

Output only the transformed instruction, starting with '**Transformed Sentence:**'.
"""


def return_system_message(guideline_instruction): 
    return f'''
    Your task is to rephrase the test questions by following the guideline.

    {guideline_instruction}

    ### Steps to Follow:
    1. **Identification Phase** (mandatory — do not skip):
    - You **must** work through **every** identification question in the guideline above, in order, for the given sentence.
    - In your reply, first write a short header line: `### Identification`
    - Then list each question by number and answer **exactly** "yes" or "no" on its own line (e.g. `1. yes`, `2. no`). Use the same numbering as in the guideline.
    - Answer the questions in a strict manner, especially for <blank> placeholders.
    - **Only after** you have written all identification lines: if **any** answer is "no", your **last** line must be exactly: `**Transformed Sentence:** (No change)` and you must not perform Actionable Changes.
    - If **all** answers are "yes", continue to step 2 in the same reply (after the identification block).

    2. **Actionable Changes** (only in the same reply, and only if every identification answer was "yes"):
    - Make only the **necessary changes** to apply the linguistic feature, ensuring no loss of information.
    - End your reply with a single line starting with `**Transformed Sentence:**` followed by the transformed text, matching the format and structure of the given example.

    ### Mandatory
    - Retain any <blank> without modifications.
    - Do not output `**Transformed Sentence:**` until the Identification section is complete.
    - Preserve the structure of the original sentence as much as possible with no information loss.
    - Follow the guideline, not considering standard English grammar.
    - The final line of your reply must be `**Transformed Sentence:** ...` or `**Transformed Sentence:** (No change)`.
    '''



def semantic_check(sentence1, sentence2):
    """
    Trans-EnV semantic checker (SS): after each candidate transformation, SS decides whether
    meaning is preserved; only passing candidates are kept for subsequent rule applications.
    """
    return f"""
    You are the semantic checker (SS) in the Trans-EnV pipeline. Sentence 1 is the original text.
    Sentence 2 is a candidate output after applying a linguistic transformation.

    Question: Is the meaning of Sentence 1 **significantly altered, contradicted, or lost** in Sentence 2?

    ### Consideration
    - All keywords from Sentence 1 should appear in Sentence 2 where they carry essential content.
    - All numbers in Sentence 1 should match Sentence 2.
    - Focus on core information and what the reader is asked to believe or do.
    - Ignore grammar and surface form; non-standard grammar alone is not a loss of meaning.
    - Missing or incorrect prepositions should not be considered decisive.
    - Ignore repetition of phrases.
    - If Sentence 2 negates, reverses, or removes an essential requirement from Sentence 1, answer "yes" (meaning altered/lost).

    Respond with **one word only** on the first line:
    - **yes** — meaning is significantly altered, contradicted, or lost (this transformation must be **discarded**).
    - **no** — essential meaning is preserved (this transformation may be **kept**).

    Sentence 1: {sentence1}
    Sentence 2: {sentence2}
    Answer:
    """




"""
### Steps to Follow:
    1. **Identification Phase**: 
    - Answer the identification questions for the linguistic feature with either "yes" or "no."
    - Answer the questions in a very strict manner leaving no room for potential.
    - Proceed to the next step only if **all** answers are "yes."
    - Otherwise, stop in identification phase with generating '**Transformed Sentence:** (No change)'.
    
    2. **Actionable Changes**: 
    - Make only the **necessary changes** to apply the linguistic feature, ensuring no loss of information.
    - Provide the final transformed sentence, adhering strictly to the format and structure of the given example.

"""
"""
    ### Consideration
    - Keywords from sentence 1 should be in sentence 2.
    - Important information in sentence 1 should all be in sentence 2.
    - Sentence 1 and sentence 2 have the same sentence structure (no different negation).
    - Overall structures of sentence 1 and sentence 2 are identical.
""" 
    