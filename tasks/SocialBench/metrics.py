"""SocialBench metrics."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _get_task_type(row: Dict[str, Any]) -> str:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    return str(meta.get("task_type") or "unknown")


def _get_choice_map(row: Dict[str, Any]) -> Dict[str, str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    raw = meta.get("original_choices_json", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k).strip().upper(): str(v).strip() for k, v in parsed.items() if str(k).strip()}


def _get_gold_letters(row: Dict[str, Any]) -> List[str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    raw = meta.get("original_label_json", "")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip().upper() for item in parsed if str(item).strip()]


def _get_gold_text_candidates(row: Dict[str, Any]) -> List[str]:
    answer = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct = answer.get("Correct_Answer", [])
    if not isinstance(correct, list):
        correct = [] if correct is None else [correct]

    candidates = []

    if correct:
        joined_space = " ".join(str(token).strip() for token in correct if str(token).strip()).strip()
        joined_compact = "".join(str(token).strip() for token in correct if str(token).strip()).strip()
        for candidate in [joined_space, joined_compact]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

    choice_map = _get_choice_map(row)
    for letter in _get_gold_letters(row):
        choice_text = choice_map.get(letter)
        if choice_text and choice_text not in candidates:
            candidates.append(choice_text)

    for item in correct:
        token = str(item).strip()
        if token and token not in candidates:
            candidates.append(token)

    return candidates


def _resolve_prediction_mcq(prediction: Any, row: Dict[str, Any]) -> str:
    choice_map = _get_choice_map(row)
    if not choice_map:
        return ""

    letter = _extract_letter(prediction)
    if letter is not None and letter in choice_map:
        return choice_map[letter]

    pred_norm = _normalize_text(prediction)
    for label, text in choice_map.items():
        if pred_norm == _normalize_text(text):
            return text
        if pred_norm == label.casefold():
            return text
    return ""


def _resolve_prediction_qa(prediction: Any) -> str:
    return str(prediction or "").strip()


def _is_correct_prediction(prediction: Any, row: Dict[str, Any]) -> bool:
    task_type = _get_task_type(row)
    if task_type == "mcq":
        pred_text = _resolve_prediction_mcq(prediction, row)
    else:
        pred_text = _resolve_prediction_qa(prediction)

    pred_norm = _normalize_text(pred_text)
    if not pred_norm:
        return False

    gold_candidates = _get_gold_text_candidates(row)
    gold_norms = {_normalize_text(candidate) for candidate in gold_candidates if _normalize_text(candidate)}
    return pred_norm in gold_norms


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


def compute_metrics(
    predictions: List[Any],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 SocialBench 指标。"""
    del judge_client
    del gold_answers

    correctness = [_is_correct_prediction(pred, row) for pred, row in zip(predictions, data)]
    sample_metrics = compute_sample_metrics(correctness, [True] * len(correctness))
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    by_task_type: Dict[str, Dict[str, int]] = {}
    by_source_split: Dict[str, Dict[str, int]] = {}
    by_lang: Dict[str, Dict[str, int]] = {}
    by_original_category: Dict[str, Dict[str, int]] = {}
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_num_choices: Dict[str, Dict[str, int]] = {}

    for sample_result, row in zip(per_sample_results, data):
        is_correct = sample_result["is_correct"]
        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        task_type = meta.get("task_type") or "unknown"
        source_split = meta.get("source_split") or "unknown"
        lang = meta.get("lang") or "unknown"
        original_category = meta.get("original_category") or "unknown"
        dimensions = meta.get("dimension", []) if isinstance(meta.get("dimension"), list) else []

        _update_group(by_task_type, task_type, is_correct)
        _update_group(by_source_split, source_split, is_correct)
        _update_group(by_lang, lang, is_correct)
        _update_group(by_original_category, original_category, is_correct)

        if task_type == "mcq":
            _update_group(by_num_choices, len(_get_choice_map(row)), is_correct)

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
        **_flatten_group(by_task_type, "by_task_type"),
        **_flatten_group(by_source_split, "by_source_split"),
        **_flatten_group(by_lang, "by_lang"),
        **_flatten_group(by_original_category, "by_original_category"),
        **_flatten_group(by_dimension, "by_dimension"),
        **_flatten_group(by_num_choices, "by_num_choices"),
        "by_task_type": _rate_dict(by_task_type),
        "task_type_counts": {key: value["total"] for key, value in sorted(by_task_type.items())},
        "by_source_split": _rate_dict(by_source_split),
        "source_split_counts": {key: value["total"] for key, value in sorted(by_source_split.items())},
        "by_lang": _rate_dict(by_lang),
        "lang_counts": {key: value["total"] for key, value in sorted(by_lang.items())},
        "by_original_category": _rate_dict(by_original_category),
        "original_category_counts": {key: value["total"] for key, value in sorted(by_original_category.items())},
        "by_dimension": _rate_dict(by_dimension),
        "dimension_counts": {key: value["total"] for key, value in sorted(by_dimension.items())},
        "by_num_choices": _rate_dict(by_num_choices),
        "num_choice_counts": {key: value["total"] for key, value in sorted(by_num_choices.items())},
        "per_sample_results": per_sample_results,
    }
