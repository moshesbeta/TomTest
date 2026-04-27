"""SocialBench prompts."""
from __future__ import annotations

from typing import Any, Dict, List


PROMPTS = {
    "unified_open_eval": """\
Read the task carefully and answer it.

Rules:
- Output JSON only with an `answer` field.
- If the task is multiple-choice, answer with the option letter only when possible.
- If the task is open-ended, answer with a short phrase only.
- Do not output explanations.

Story:
{story}

Question:
{question}
""",
}


def build_prompt(row: Dict[str, Any], method: str = "unified_open_eval") -> str:
    """构建 SocialBench prompt。"""
    template = PROMPTS.get(method, PROMPTS["unified_open_eval"])

    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    story = str(story_info.get("full_story", "") or "").strip()
    question = str(row.get("Question", "") or "").strip()

    return template.format(
        story=story,
        question=question,
    )
