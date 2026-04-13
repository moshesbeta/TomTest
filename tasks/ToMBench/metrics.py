"""ToMBench 数据集的 metrics 计算"""

from typing import Any, Dict, List


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 ToMBench 的 metrics

    包括整体准确率和按 Meta.ability 分组的二级准确率

    Args:
        predictions: 模型预测答案列表
        data: 原始数据列表

    Returns:
        包含基础指标和二级指标的字典
    """
    gold_answers = [row['Answer']['Correct Answer'][0] for row in data]

    # 基础指标
    correct = sum(1 for p, g in zip(predictions, gold_answers) if p == g)
    accuracy = correct / len(predictions) if predictions else 0

    # 二级指标：按 Meta.ability 分组
    ability_metrics = {}
    for pred, gold, row in zip(predictions, gold_answers, data):
        # 获取 ability
        ability = "unknown"
        if row.get("Meta") and isinstance(row["Meta"], dict):
            ability_value = row["Meta"].get("ability")
            if ability_value:
                ability = ability_value

        if ability not in ability_metrics:
            ability_metrics[ability] = {"correct": 0, "total": 0}
        ability_metrics[ability]["total"] += 1
        if pred == gold:
            ability_metrics[ability]["correct"] += 1

    # 计算各 ability 准确率
    secondary_metrics = {
        f"by_ability.{ability}": stats["correct"] / stats["total"]
        for ability, stats in ability_metrics.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        **secondary_metrics,
        "by_ability": {ability: stats["correct"] / stats["total"] for ability, stats in ability_metrics.items()},
        "ability_counts": {ability: stats["total"] for ability, stats in ability_metrics.items()},
    }
