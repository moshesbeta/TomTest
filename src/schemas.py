"""统一的输出 Schema 定义"""
from typing import Literal, List

from pydantic import BaseModel, constr, Field, field_validator


class OneWordAnswer(BaseModel):
    """单词答案 schema（不允许空白字符）"""
    answer: constr(strip_whitespace=True, min_length=1, pattern=r"^\S+$") = Field(
        description="A single word answer with no whitespace characters"
    )


class MCQAnswer(BaseModel):
    """多选题答案 schema（选项字母 A/B/C/D）"""
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The answer to the multiple choice question, must be one of A, B, C, D"
    )


class MCQAnswer5(BaseModel):
    """五选一多选题答案 schema（选项字母 A/B/C/D/E）"""
    answer: Literal["A", "B", "C", "D", "E"] = Field(
        description="The answer to the multiple choice question, must be one of A, B, C, D, E"
    )


class MCQAnswer3(BaseModel):
    """三选一多选题答案 schema（选项字母 A/B/C，用于 SocialIQA）"""
    answer: Literal["A", "B", "C"] = Field(
        description="The answer to the multiple choice question, must be one of A, B, C"
    )


class MCQAnswer3Lower(BaseModel):
    """三选一多选题答案 schema（小写 a/b/c，用于 Belief_R 等）"""
    answer: Literal["a", "b", "c"] = Field(
        description="The answer to the multiple choice question, must be one of a, b, c"
    )


class MCQAnswer2(BaseModel):
    """二选一单选题答案 schema（选项字母 A/B）"""
    answer: Literal["A", "B"] = Field(
        description="The answer to the binary choice question, must be one of A, B"
    )


class OpenAnswer(BaseModel):
    """开放式问答 schema"""
    answer: str = Field(
        default="",
        description="The answer to the open-ended question"
    )


class JudgeAnswer(BaseModel):
    """判断题答案 schema"""
    answer: Literal["True", "False"] = Field(
        description="The answer to the true/false question, must be either True or False"
    )


class MultiLabelAnswer(BaseModel):
    """多标签多选题答案 schema（用于 RecToM，返回标签列表如 ['A'] 或 ['B', 'D']）"""
    answer: List[str] = Field(
        default_factory=list,
        description="A list of option labels only, each must be a single uppercase letter (A-Z), such as ['A'] or ['B', 'D']",
    )

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value):
        if value is None:
            return []

        import re

        if isinstance(value, str):
            items = re.findall(r"[A-Za-z][A-Za-z0-9_]*", value)
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]

        normalized = []
        seen = set()
        for item in items:
            token = str(item).strip().upper()
            if token and len(token) == 1 and token.isalpha() and token not in seen:
                normalized.append(token)
                seen.add(token)
        return normalized
