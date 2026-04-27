"""UserBench metrics."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


RESOURCE_ID_PATTERN = re.compile(r"^[AFHRC]\d+$")


def _normalize_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.findall(r"[AFHRCafhrc]\d+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized = []
    seen = set()
    for item in items:
        token = str(item).strip().upper()
        if token and RESOURCE_ID_PATTERN.match(token) and token not in seen:
            normalized.append(token)
            seen.add(token)
    return normalized


def _parse_gold_ids(row: Dict[str, Any]) -> List[str]:
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    raw = answer_block.get("ground_truth", "[]")

    if isinstance(raw, list):
        return _normalize_ids(raw)

    if raw is None:
        return []

    try:
        parsed = ast.literal_eval(str(raw))
    except (SyntaxError, ValueError):
        parsed = raw
    return _normalize_ids(parsed)


def _classify_prediction(pred_ids: Iterable[str], gold_ids: Iterable[str]) -> str:
    pred_set = set(_normalize_ids(list(pred_ids)))
    gold_set = set(_normalize_ids(list(gold_ids)))

    if pred_set == gold_set:
        return "full_correct"
    if pred_set.issubset(gold_set):
        return "partial_no_error"
    return "has_error"


def _init_group(stats: Dict[str, Dict[str, int]], key: Any) -> Dict[str, int]:
    key_str = str(key) if key not in (None, "") else "unknown"
    if key_str not in stats:
        stats[key_str] = {
            "full_correct": 0,
            "partial_no_error": 0,
            "has_error": 0,
            "total": 0,
        }
    return stats[key_str]


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, category: str) -> None:
    group = _init_group(stats, key)
    group["total"] += 1
    group[category] += 1


def _flatten_group(stats: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in sorted(stats.items()):
        total = value["total"]
        flat[f"{prefix}_full_correct.{key}"] = value["full_correct"] / total if total else 0.0
        flat[f"{prefix}_partial_no_error.{key}"] = value["partial_no_error"] / total if total else 0.0
        flat[f"{prefix}_has_error.{key}"] = value["has_error"] / total if total else 0.0
    return flat


def _rate_dict(stats: Dict[str, Dict[str, int]], field: str) -> Dict[str, float]:
    return {
        key: (value[field] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(stats.items())
    }


def compute_metrics(
    predictions: List[List[str]],
    gold_answers: List[List[str]],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 UserBench 指标。"""
    del judge_client

    total = len(data)
    full_correct = 0
    partial_no_error = 0
    has_error = 0

    by_task_type: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}
    by_dataset_source: Dict[str, Dict[str, int]] = {}
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_num_targets: Dict[str, Dict[str, int]] = {}

    per_sample_results = []

    for pred, gold, row in zip(predictions, gold_answers, data):
        pred_ids = _normalize_ids(pred)
        gold_ids = _normalize_ids(gold)
        category = _classify_prediction(pred_ids, gold_ids)

        is_correct = category == "full_correct"
        if is_correct:
            full_correct += 1
        elif category == "partial_no_error":
            partial_no_error += 1
        else:
            has_error += 1

        per_sample_results.append({
            "is_correct": is_correct,
            "error_reason": None if is_correct else category,
        })

        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        task_type = meta.get("task_type") or "unknown"
        difficulty = meta.get("difficulty") or "unknown"
        dataset_source = meta.get("dataset_source") or "unknown"
        dimensions = meta.get("dimension") if isinstance(meta.get("dimension"), list) else []

        _update_group(by_task_type, task_type, category)
        _update_group(by_difficulty, difficulty, category)
        _update_group(by_dataset_source, dataset_source, category)
        _update_group(by_num_targets, len(gold_ids), category)

        if dimensions:
            for dimension in dimensions:
                _update_group(by_dimension, dimension, category)
        else:
            _update_group(by_dimension, "unknown", category)

    accuracy = full_correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": full_correct,
        "total": total,
        "full_correct": full_correct,
        "partial_no_error": partial_no_error,
        "has_error": has_error,
        "full_correct_rate": full_correct / total if total else 0.0,
        "partial_no_error_rate": partial_no_error / total if total else 0.0,
        "has_error_rate": has_error / total if total else 0.0,
        **_flatten_group(by_task_type, "by_task_type"),
        **_flatten_group(by_difficulty, "by_difficulty"),
        **_flatten_group(by_dataset_source, "by_dataset_source"),
        **_flatten_group(by_dimension, "by_dimension"),
        **_flatten_group(by_num_targets, "by_num_targets"),
        "by_task_type_full_correct": _rate_dict(by_task_type, "full_correct"),
        "by_task_type_partial_no_error": _rate_dict(by_task_type, "partial_no_error"),
        "by_task_type_has_error": _rate_dict(by_task_type, "has_error"),
        "task_type_counts": {key: value["total"] for key, value in sorted(by_task_type.items())},
        "by_difficulty_full_correct": _rate_dict(by_difficulty, "full_correct"),
        "by_difficulty_partial_no_error": _rate_dict(by_difficulty, "partial_no_error"),
        "by_difficulty_has_error": _rate_dict(by_difficulty, "has_error"),
        "difficulty_counts": {key: value["total"] for key, value in sorted(by_difficulty.items())},
        "by_dataset_source_full_correct": _rate_dict(by_dataset_source, "full_correct"),
        "by_dataset_source_partial_no_error": _rate_dict(by_dataset_source, "partial_no_error"),
        "by_dataset_source_has_error": _rate_dict(by_dataset_source, "has_error"),
        "dataset_source_counts": {key: value["total"] for key, value in sorted(by_dataset_source.items())},
        "by_dimension_full_correct": _rate_dict(by_dimension, "full_correct"),
        "by_dimension_partial_no_error": _rate_dict(by_dimension, "partial_no_error"),
        "by_dimension_has_error": _rate_dict(by_dimension, "has_error"),
        "dimension_counts": {key: value["total"] for key, value in sorted(by_dimension.items())},
        "by_num_targets_full_correct": _rate_dict(by_num_targets, "full_correct"),
        "by_num_targets_partial_no_error": _rate_dict(by_num_targets, "partial_no_error"),
        "by_num_targets_has_error": _rate_dict(by_num_targets, "has_error"),
        "num_target_counts": {key: value["total"] for key, value in sorted(by_num_targets.items())},
        "per_sample_results": per_sample_results,
    }
