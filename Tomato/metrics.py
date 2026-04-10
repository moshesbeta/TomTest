"""Tomato 数据集的 metrics：整体准确率 + Meta.dimension 三维分层。"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _extract_dimension_slots(row: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = row.get("Meta") or {}
    if not isinstance(meta, dict):
        return "__missing__", "__missing__", "__none__"
    dims = meta.get("dimension", [])
    if not isinstance(dims, list):
        dims = [dims]
    d0 = str(dims[0]) if len(dims) >= 1 and dims[0] not in (None, "") else "__missing__"
    d1 = str(dims[1]) if len(dims) >= 2 and dims[1] not in (None, "") else "__missing__"
    d2 = str(dims[2]) if len(dims) >= 3 and dims[2] not in (None, "") else "__none__"
    return d0, d1, d2


def compute_metrics(
    predictions: List[str],
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """计算 Tomato 的 metrics

    包括整体准确率和按 Meta.dimension 分组的三维分层准确率

    Args:
        predictions: 模型预测答案列表
        data: 原始数据列表，须包含 '_mcq' 字段（含 gold_letter）

    Returns:
        包含基础指标和二级指标的字典
    """
    gold_letters = [row["_mcq"]["gold_letter"] for row in data]
    assert len(predictions) == len(data) == len(gold_letters), "predictions/data/gold 长度须一致"

    correct = 0
    total = len(predictions)

    dim_total: List[Dict[str, int]] = [defaultdict(int) for _ in range(3)]
    dim_correct: List[Dict[str, int]] = [defaultdict(int) for _ in range(3)]

    for pred, gold, row in zip(predictions, gold_letters, data):
        hit = bool(pred and pred == gold)
        correct += int(hit)
        d0, d1, d2 = _extract_dimension_slots(row)
        for i, d in enumerate((d0, d1, d2)):
            dim_total[i][d] += 1
            if hit:
                dim_correct[i][d] += 1

    accuracy = correct / total if total else 0.0

    def _acc_dict(tot: Dict[str, int], cor: Dict[str, int]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in sorted(tot.keys()):
            out[k] = (cor.get(k, 0) / tot[k]) if tot[k] else 0.0
        return out

    dim_acc = [_acc_dict(dim_total[i], dim_correct[i]) for i in range(3)]

    secondary_metrics: Dict[str, float] = {}
    for i in range(3):
        for k, v in dim_acc[i].items():
            secondary_metrics[f"by_dimension_{i+1}.{k}"] = v

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "dimension_metrics": {
            "method_1": {
                "name": "Meta.dimension[0]",
                "category_accuracy": dim_acc[0],
                "category_total": {k: dim_total[0][k] for k in sorted(dim_total[0].keys())},
            },
            "method_2": {
                "name": "Meta.dimension[1]",
                "category_accuracy": dim_acc[1],
                "category_total": {k: dim_total[1][k] for k in sorted(dim_total[1].keys())},
            },
            "method_3": {
                "name": "Meta.dimension[2]",
                "category_accuracy": dim_acc[2],
                "category_total": {k: dim_total[2][k] for k in sorted(dim_total[2].keys())},
            },
        },
    }
