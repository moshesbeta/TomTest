"""PUB metrics."""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, correct: bool) -> None:
    key_str = str(key) if key not in (None, "") else "unknown"
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if correct:
        stats[key_str]["correct"] += 1


def _flatten(group: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}.{key}": (value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(group.items())
    }


def _to_metric_map(group: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    return {
        key: (value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(group.items())
    }


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 PUB 的整体准确率和按 Meta 字段分组的指标。"""
    if judge_client is not None:
        sample_metrics = compute_sample_metrics_with_llm(predictions, gold_answers, judge_client)
    else:
        sample_metrics = compute_sample_metrics(predictions, gold_answers)
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    by_source: Dict[str, Dict[str, int]] = {}
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}
    by_ethics_category: Dict[str, Dict[str, int]] = {}
    by_task_type: Dict[str, Dict[str, int]] = {}
    by_option_count: Dict[str, Dict[str, int]] = {}

    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        source = meta.get("dataset_source", "unknown")
        difficulty = meta.get("difficulty", "unknown")
        ethics_category = meta.get("ethics_category", "unknown")
        task_type = meta.get("task_type", "unknown")
        option_count = row.get("_mcq", {}).get("option_count", meta.get("option_count", "unknown"))
        dims = meta.get("dimension", [])
        if isinstance(dims, list) and dims:
            dimension = dims[0]
        else:
            dimension = dims if dims else "unknown"

        _update_group(by_source, source, is_correct)
        _update_group(by_dimension, dimension, is_correct)
        _update_group(by_difficulty, difficulty, is_correct)
        _update_group(by_ethics_category, ethics_category, is_correct)
        _update_group(by_task_type, task_type, is_correct)
        _update_group(by_option_count, option_count, is_correct)

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **_flatten(by_source, "by_source"),
        **_flatten(by_dimension, "by_dimension"),
        **_flatten(by_difficulty, "by_difficulty"),
        **_flatten(by_ethics_category, "by_ethics_category"),
        **_flatten(by_task_type, "by_task_type"),
        **_flatten(by_option_count, "by_option_count"),
        "by_source": _to_metric_map(by_source),
        "source_counts": {key: value["total"] for key, value in sorted(by_source.items())},
        "by_dimension": _to_metric_map(by_dimension),
        "dimension_counts": {key: value["total"] for key, value in sorted(by_dimension.items())},
        "by_difficulty": _to_metric_map(by_difficulty),
        "difficulty_counts": {key: value["total"] for key, value in sorted(by_difficulty.items())},
        "by_ethics_category": _to_metric_map(by_ethics_category),
        "ethics_category_counts": {
            key: value["total"] for key, value in sorted(by_ethics_category.items())
        },
        "by_task_type": _to_metric_map(by_task_type),
        "task_type_counts": {key: value["total"] for key, value in sorted(by_task_type.items())},
        "by_option_count": _to_metric_map(by_option_count),
        "option_count_counts": {
            key: value["total"] for key, value in sorted(by_option_count.items())
        },
        "per_sample_results": per_sample_results,
    }
