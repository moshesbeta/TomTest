"""Tomato 数据集的输出 schema（与 ToMBench 对齐；推理解析见 run.py）。"""
from typing import Literal

from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """多选题答案 schema（选项字母）。"""
    answer: Literal["A", "B", "C", "D"]


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
