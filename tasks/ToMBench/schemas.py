"""ToMBench 数据集的输出 schema"""
from pydantic import BaseModel
from typing import Literal


class MCQAnswer(BaseModel):
    """多选题答案 schema"""
    answer: Literal["A", "B", "C", "D"]


# schema 字典：config.yaml 中引用主 schema，其他 schema 供内部调用
SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
