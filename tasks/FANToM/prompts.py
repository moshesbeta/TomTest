"""FANToM prompts.

Prompt templates are derived from tasks/FANToM/eval_fantom_origin.py, focusing on the
extra instructions beyond Story/Question (e.g. Target/Information headers, A/B choice wording).

The method key must match tasks/FANToM/config.yaml `method`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


PROMPTS: Dict[str, Dict[str, str]] = {
    "ZS_vanilla": {
        "factQA": """\
{story}

Question: {question}
Answer:""",
        "beliefQAs": """\
{story}

Question: {question}
Answer:""",
        "beliefQAs_choice": """\
{story}

Question: {question}
{choices_text}

Choose an answer from above:
Answer with exactly one letter (A/B):""",
        "answerabilityQA_list": """\
{story}

Target: {fact_question}
Question: {question}

Options:
{options_text}

Answer with option letters only (e.g., ["A", "C"]):""",
        "answerabilityQAs_binary": """\
{story}

Target: {fact_question}
Question: {question}

Options:
{options_text}

Answer with exactly one letter (A/B):""",
        "infoAccessibilityQA_list": """\
{story}

Information: {fact_question} {fact_answer}
Question: {question}

Options:
{options_text}

Answer with option letters only (e.g., ["A", "C"]):""",
        "infoAccessibilityQAs_binary": """\
{story}

Information: {fact_question} {fact_answer}
Question: {question}

Options:
{options_text}

Answer with exactly one letter (A/B):""",
    },
    "ZS_CoT": {
        "factQA": """\
{story}

Question: {question}

Let's think step by step. Then answer.
Answer:""",
        "beliefQAs": """\
{story}

Question: {question}

Let's think step by step. Then answer.
Answer:""",
        "beliefQAs_choice": """\
{story}

Question: {question}
{choices_text}

Choose an answer from above:
Let's think step by step. Then answer with exactly one letter (A/B):""",
        "answerabilityQA_list": """\
{story}

Target: {fact_question}
Question: {question}

Options:
{options_text}

Let's think step by step. Then answer with option letters only (e.g., ["A", "C"]):""",
        "answerabilityQAs_binary": """\
{story}

Target: {fact_question}
Question: {question}

Options:
{options_text}

Let's think step by step. Then answer with exactly one letter (A/B):""",
        "infoAccessibilityQA_list": """\
{story}

Information: {fact_question} {fact_answer}
Question: {question}

Options:
{options_text}

Let's think step by step. Then answer with option letters only (e.g., ["A", "C"]):""",
        "infoAccessibilityQAs_binary": """\
{story}

Information: {fact_question} {fact_answer}
Question: {question}

Options:
{options_text}

Let's think step by step. Then answer with exactly one letter (A/B):""",
    },
}


def _get_story(row: Mapping[str, Any]) -> str:
    story_block = row.get("Story", {})
    if isinstance(story_block, dict):
        return str(story_block.get("full_story", "") or story_block.get("summary", "") or "").strip()
    return str(story_block or "").strip()


def _get_question_type(row: Mapping[str, Any]) -> str:
    qtype = row.get("question_type")
    if qtype:
        return str(qtype)
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    meta_id = str(meta.get("id", ""))
    if "__" in meta_id:
        parts = meta_id.split("__")
        if len(parts) >= 2:
            return parts[1]
    return "unknown"


def _format_choices_paren(choices: Mapping[str, Any]) -> str:
    lines = []
    for k in ("A", "B"):
        if k in choices:
            lines.append(f"({k}) {str(choices[k]).strip()}")
    return "\n".join(lines)


def _format_options_dot(choices: Mapping[str, Any]) -> str:
    lines = []
    for k in sorted(choices.keys()):
        lines.append(f"{k}. {str(choices[k]).strip()}")
    return "\n".join(lines)


def build_prompt(row: Dict[str, Any], method: str = "ZS_vanilla") -> str:
    """Build FANToM prompt.

    Args:
        row: dataset row (may include derived fields like question_type/_group/_mcq)
        method: prompt method key in PROMPTS
    """
    templates = PROMPTS.get(method) or PROMPTS["ZS_vanilla"]

    question_type = _get_question_type(row)
    template = templates.get(question_type) or templates.get("beliefQAs") or ""

    story = _get_story(row)
    question = str(row.get("Question", "") or "").strip()
    fact_question = str(row.get("fact_question", "") or "").strip()
    fact_answer = str(row.get("fact_answer", "") or "").strip()

    mcq = row.get("_mcq", {}) if isinstance(row.get("_mcq"), dict) else {}
    choices = mcq.get("choices", {}) if isinstance(mcq.get("choices"), dict) else {}

    choices_text = _format_choices_paren(choices)
    options_text = _format_options_dot(choices)

    return template.format(
        story=story,
        question=question,
        fact_question=fact_question,
        fact_answer=fact_answer,
        choices_text=choices_text,
        options_text=options_text,
    ).strip()
