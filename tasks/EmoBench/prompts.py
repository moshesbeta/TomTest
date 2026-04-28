"""EmoBench prompts."""
from __future__ import annotations

from typing import Any, Dict, List


PROMPTS = {
    "zero_shot_mcq": """\
Read the scenario and answer the multiple-choice question.

Rules:
- Select exactly one option.
- You may answer with either the option letter or the exact option text.
- Output JSON only with an `answer` field.
- Do not output explanations.

Scenario:
{story}

Question:
{question}

Options:
{options}
""",
}


def _format_options(options: List[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"{labels[idx]}. {option}"
        for idx, option in enumerate(options)
        if idx < len(labels)
    )


def build_prompt(row: Dict[str, Any], method: str = "zero_shot_mcq") -> str:
    """构建 EmoBench prompt。"""
    template = PROMPTS.get(method, PROMPTS["zero_shot_mcq"])

    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}

    story = str(story_info.get("full_story", "") or "").strip()
    question = str(row.get("Question", "") or "").strip()
    options = [str(option).strip() for option in meta.get("choice_texts", []) if str(option).strip()]

    return template.format(
        story=story,
        question=question,
        options=_format_options(options),
    )
