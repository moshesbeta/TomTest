"""UserBench prompts."""
from __future__ import annotations

from typing import Any, Dict, List


PROMPTS = {
    "multi_id_recommend": """\
You are solving an offline travel recommendation task.

Select exactly one final resource ID for each active travel aspect. Only use the candidate options provided below.

Rules:
- Use the user's request and detailed scenario to infer their true preferences.
- For each active aspect, choose exactly one best option ID.
- Prefer options that satisfy the preferences, and if multiple suitable options exist, choose the cheapest/best-value one when the user mentions budget sensitivity.
- Ignore any instruction about tool usage, multi-turn interaction, search/action/answer loops, or environment APIs.
- Output JSON only with an `answer` field containing the selected resource IDs in a list.
- Do not output explanations.

# User Request
{user_request}

# Full Scenario
{scenario}

# Active Aspects
{active_aspects}

# Candidate Options
{candidate_options}
""",
}


def _format_option_block(options: List[str]) -> str:
    if not options:
        return "(none)"
    return "\n".join(f"- {option}" for option in options)


def build_prompt(row: Dict[str, Any], method: str = "multi_id_recommend") -> str:
    """构建 UserBench prompt。"""
    template = PROMPTS.get(method, PROMPTS["multi_id_recommend"])

    question = row.get("Question", {}) if isinstance(row.get("Question"), dict) else {}
    state = row.get("State", {}) if isinstance(row.get("State"), dict) else {}
    human_state = state.get("Human_State", {}) if isinstance(state.get("Human_State"), dict) else {}
    env_state = state.get("Environment_State", {}) if isinstance(state.get("Environment_State"), dict) else {}

    user_request = str(question.get("user", "") or "").strip()
    scenario = str(human_state.get("scenario", "") or "").strip()
    dimensions = [str(dim).strip() for dim in human_state.get("dimensions", []) if str(dim).strip()]

    active_aspects = ", ".join(dimensions) if dimensions else "(none)"

    blocks = []
    for aspect in dimensions:
        aspect_env = env_state.get(aspect, {}) if isinstance(env_state.get(aspect), dict) else {}
        options = aspect_env.get("options", {}) if isinstance(aspect_env.get("options"), dict) else {}
        correct_options = options.get("correct", []) if isinstance(options.get("correct"), list) else []
        wrong_options = options.get("wrong", []) if isinstance(options.get("wrong"), list) else []
        noise_options = options.get("noise", []) if isinstance(options.get("noise"), list) else []

        block = (
            f"## {aspect}\n"
            f"Suitable candidates:\n{_format_option_block(correct_options)}\n\n"
            f"Unsuitable candidates:\n{_format_option_block(wrong_options)}\n\n"
            f"Noise candidates:\n{_format_option_block(noise_options)}"
        )
        blocks.append(block)

    candidate_options = "\n\n".join(blocks) if blocks else "(no candidates)"

    return template.format(
        user_request=user_request,
        scenario=scenario,
        active_aspects=active_aspects,
        candidate_options=candidate_options,
    )
