"""ToMi prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

Return the answer JSON using exactly one lowercase word in the `answer` field.
Do not output a sentence, explanation, or punctuation.""",
    "cot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

Think step by step internally, then return the answer JSON using exactly one lowercase word in the `answer` field.
Do not output a sentence, explanation, or punctuation.""",
}


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (zero_shot/cot)

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["zero_shot"])
    story = row.get("Story", {}).get("full_story", "")
    question = row.get("Question", "")
    return template.format(story=story, question=question)
