"""Tomato prompts：系统提示 + Transcript/Question/Options 用户块。"""
from typing import Any, Dict

TOMATO_MCQA_SYSTEM = (
    "You are an expert at understanding human communication. Please leverage the information provided "
    "and choose the most probable answer to the question from the options. "
    "Output your final answer by strictly following this format: [A], [B], [C], or [D]"
)

TOMATO_USER_TEMPLATE = """# Transcript
{story_block}

# Question
{question}

# Options
{options_block}"""


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def _build_user_block(story_block: str, question: str, options: Dict[str, str]) -> str:
    lines_o = [f"[{letter}] {options[letter]}" for letter in sorted(options.keys())]
    options_block = "\n".join(lines_o)
    fields = _SafeFormatDict(
        story_block=_escape_braces(story_block.strip()),
        question=_escape_braces(question.strip()),
        options_block=_escape_braces(options_block),
    )
    return TOMATO_USER_TEMPLATE.format_map(fields)


PROMPTS = {
    "v2_generate": TOMATO_MCQA_SYSTEM,
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板

    Args:
        method: prompt 方法名称

    Returns:
        prompt 模板字符串（系统提示）
    """
    return PROMPTS.get(method, TOMATO_MCQA_SYSTEM)


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 prompt

    Args:
        template: prompt 模板（系统提示文本）
        row: 数据行，须包含 '_mcq' 字段（由 preprocess_mcq 写入）

    Returns:
        格式化的 prompt
    """
    mcq = row["_mcq"]
    user = _build_user_block(mcq["story"], mcq["question"], mcq["original_choices"])
    return f"{template}\n\n{user}"
