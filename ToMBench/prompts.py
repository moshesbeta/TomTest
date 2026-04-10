"""ToMBench prompts"""
from typing import Any, Dict, List

PROMPTS = {
    "zero_shot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

output the answer JSON with exactly one letter (A/B/C/D):""",

    "cot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

Let's think step by step...

output the answer JSON with exactly one letter (A/B/C/D):""",
}

def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 prompt

    Args:
        template: prompt 模板
        row: 数据行

    Returns:
        格式化的 prompt
    """
    story = row.get("Story", "")
    question = row.get("Question", "")
    return template.format(story=story, question=question)


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板

    Args:
        method: prompt 方法名称

    Returns:
        prompt 模板字符串
    """
    return PROMPTS.get(method, PROMPTS["zero_shot"])


