"""EmoBench metrics."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""

    s = str(text).strip()
    s = re.sub(r"^[\"'`]+|[\"'`]+$", "", s)
    s = re.sub(r"^\s*(answer|ans|option|答案)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _extract_letter(text: Any) -> Optional[str]:
    s = str(text or "").strip()
    match = re.match(r"^\s*([A-Za-z])\b", s)
    if not match:
        return None
    return match.group(1).upper()


def _get_choice_texts(row: Dict[str, Any]) -> List[str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    raw = meta.get("choice_texts", [])
    if not isinstance(raw, list):
        return []
    return [str(choice).strip() for choice in raw if str(choice).strip()]


def _resolve_prediction_to_text(prediction: Any, row: Dict[str, Any]) -> str:
    choices = _get_choice_texts(row)
    if not choices:
        return ""

    letter = _extract_letter(prediction)
    if letter is not None:
        idx = OPTION_LABELS.find(letter)
        if 0 <= idx < len(choices):
            return choices[idx]

    pred_norm = _normalize_text(prediction)
    for choice in choices:
        if pred_norm == _normalize_text(choice):
            return choice
    return ""


def _get_gold_answer(row: Dict[str, Any]) -> str:
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct = answer_block.get("Correct_Answer", [])
    if isinstance(correct, list) and correct:
        return str(correct[0]).strip()
    if correct is None:
        return ""
    return str(correct).strip()


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, correct: bool) -> None:
    key_str = str(key) if key not in (None, "") else "unknown"
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if correct:
        stats[key_str]["correct"] += 1


def _flatten_group(stats: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}.{key}": (value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(stats.items())
    }


def _rate_dict(stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    return {
        key: (value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(stats.items())
    }


def _get_dimension_list(row: Dict[str, Any]) -> List[str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    dims = meta.get("dimension", [])
    if not isinstance(dims, list):
        return []
    return [str(dim).strip() for dim in dims if str(dim).strip()]


def compute_metrics(
    predictions: List[Any],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 EmoBench 指标。"""
    del judge_client

    resolved_predictions = [_resolve_prediction_to_text(pred, row) for pred, row in zip(predictions, data)]
    gold_texts = [_get_gold_answer(row) if not gold else gold for row, gold in zip(data, gold_answers)]

    sample_metrics = compute_sample_metrics(
        [_normalize_text(pred) for pred in resolved_predictions],
        [_normalize_text(gold) for gold in gold_texts],
    )
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    by_subset: Dict[str, Dict[str, int]] = {}
    by_language: Dict[str, Dict[str, int]] = {}
    by_question_subtype: Dict[str, Dict[str, int]] = {}
    by_coarse_category: Dict[str, Dict[str, int]] = {}
    by_finegrained_category: Dict[str, Dict[str, int]] = {}
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_num_choices: Dict[str, Dict[str, int]] = {}

    for sample_result, row in zip(per_sample_results, data):
        is_correct = sample_result["is_correct"]
        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}

        _update_group(by_subset, meta.get("subset") or "unknown", is_correct)
        _update_group(by_language, meta.get("language") or "unknown", is_correct)
        _update_group(by_question_subtype, meta.get("question_subtype") or "unknown", is_correct)
        _update_group(by_coarse_category, meta.get("coarse_category") or "unknown", is_correct)
        _update_group(by_finegrained_category, meta.get("finegrained_category") or "unknown", is_correct)
        _update_group(by_num_choices, len(_get_choice_texts(row)), is_correct)

        dimensions = _get_dimension_list(row)
        if dimensions:
            for dimension in dimensions:
                _update_group(by_dimension, dimension, is_correct)
        else:
            _update_group(by_dimension, "unknown", is_correct)

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **_flatten_group(by_subset, "by_subset"),
        **_flatten_group(by_language, "by_language"),
        **_flatten_group(by_question_subtype, "by_question_subtype"),
        **_flatten_group(by_coarse_category, "by_coarse_category"),
        **_flatten_group(by_finegrained_category, "by_finegrained_category"),
        **_flatten_group(by_dimension, "by_dimension"),
        **_flatten_group(by_num_choices, "by_num_choices"),
        "by_subset": _rate_dict(by_subset),
        "subset_counts": {key: value["total"] for key, value in sorted(by_subset.items())},
        "by_language": _rate_dict(by_language),
        "language_counts": {key: value["total"] for key, value in sorted(by_language.items())},
        "by_question_subtype": _rate_dict(by_question_subtype),
        "question_subtype_counts": {key: value["total"] for key, value in sorted(by_question_subtype.items())},
        "by_coarse_category": _rate_dict(by_coarse_category),
        "coarse_category_counts": {key: value["total"] for key, value in sorted(by_coarse_category.items())},
        "by_finegrained_category": _rate_dict(by_finegrained_category),
        "finegrained_category_counts": {key: value["total"] for key, value in sorted(by_finegrained_category.items())},
        "by_dimension": _rate_dict(by_dimension),
        "dimension_counts": {key: value["total"] for key, value in sorted(by_dimension.items())},
        "by_num_choices": _rate_dict(by_num_choices),
        "num_choice_counts": {key: value["total"] for key, value in sorted(by_num_choices.items())},
        "per_sample_results": per_sample_results,
    }
