"""Tomato prompts：系统提示 + Transcript/Question/Options 用户块。"""
from typing import Any, Dict

TOMATO_MCQA_SYSTEM = (
    "You are an expert at understanding human communication. Use the transcript and options "
    "to choose the single best answer. Respond with structured output: the answer field must be "
    "exactly one letter A, B, C, or D matching one of the listed options."
)

PROMPTS = {
    "v2_generate": TOMATO_MCQA_SYSTEM,
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板"""
    return PROMPTS.get(method, TOMATO_MCQA_SYSTEM)


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """拼接用户块（不对正文做 str.format，避免 transcript 中花括号干扰）。"""
    mcq = row["_mcq"]
    story_block = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["original_choices"]
    lines_o = [f"[{letter}] {options[letter]}" for letter in sorted(options.keys())]
    options_block = "\n".join(lines_o)
    user = (
        f"# Transcript\n{story_block}\n\n"
        f"# Question\n{question}\n\n"
        f"# Options\n{options_block}"
    )
    return f"{template}\n\n{user}"
