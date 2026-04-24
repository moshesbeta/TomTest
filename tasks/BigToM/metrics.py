"""BigToM metrics: overall + condition-wise + belief-type + TB∧FB pair-gated accuracy."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from src.utils import compute_sample_metrics


def _safe_div(correct: int, total: int) -> float:
	return correct / total if total else 0.0


def _normalize_label(value: Any) -> str:
	if value is None:
		return ""
	return str(value).strip().upper()


def _extract_condition_type(row: Dict[str, Any]) -> str:
	meta = row.get("Meta", {})
	if not isinstance(meta, dict):
		return "unknown"
	condition = str(meta.get("condition_type", "unknown")).strip()
	return condition if condition else "unknown"


def _extract_belief_types(row: Dict[str, Any]) -> Set[str]:
	meta = row.get("Meta", {})
	if not isinstance(meta, dict):
		return {"unknown"}

	dims = meta.get("dimension", [])
	if isinstance(dims, str):
		dims = [dims]
	if not isinstance(dims, list):
		dims = [dims]

	tags: Set[str] = set()
	for dim in dims:
		token = str(dim).strip().lower().replace("-", "_").replace(" ", "_")
		tags.add(token)

	return tags


def _build_pair_key(row: Dict[str, Any]) -> str:
	meta = row.get("Meta", {})
	if not isinstance(meta, dict):
		return "__unknown_pair__"

	sample_id = str(meta.get("id", "")).strip()
	if not sample_id:
		return "__unknown_pair__"

	if "__" in sample_id:
		prefix, _ = sample_id.rsplit("__", 1)
		normalized_prefix = prefix
		if normalized_prefix.endswith("_true_belief"):
			normalized_prefix = normalized_prefix[: -len("_true_belief")]
		elif normalized_prefix.endswith("_false_belief"):
			normalized_prefix = normalized_prefix[: -len("_false_belief")]

		# Pair at scenario level (drop sample suffix id).
		return normalized_prefix or "__unknown_pair__"

	# Fallback for ids without separator.
	normalized = sample_id
	if normalized.endswith("_true_belief"):
		normalized = normalized[: -len("_true_belief")]
	elif normalized.endswith("_false_belief"):
		normalized = normalized[: -len("_false_belief")]
	return normalized or "__unknown_pair__"


def _extract_gold_from_row(row: Dict[str, Any]) -> str:
	mcq = row.get("_mcq") if isinstance(row.get("_mcq"), dict) else {}
	gold = _normalize_label(mcq.get("gold_letter", ""))
	if gold in {"A", "B"}:
		return gold

	# Fallback: by convention this task starts with A as correct.
	return "A"


def compute_metrics(
	predictions: List[Any],
	data: List[Dict[str, Any]],
	judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
	"""Compute BigToM metrics with data-derived gold labels."""

	if len(predictions) != len(data):
		raise ValueError(f"predictions/data length mismatch: {len(predictions)} vs {len(data)}")

	gold_answers = [_extract_gold_from_row(row) for row in data]

	sample_metrics = compute_sample_metrics(
		predictions=predictions,
		gold_answers=gold_answers,
		is_correct_fn=lambda p, g: _normalize_label(p) == _normalize_label(g),
	)

	correct = sample_metrics["correct"]
	total = sample_metrics["total"]
	per_sample_results = sample_metrics["per_sample_results"]

	condition_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
	belief_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
	pair_stats: Dict[str, Dict[str, int]] = defaultdict(
		lambda: {"tb_correct": 0, "tb_total": 0, "fb_correct": 0, "fb_total": 0}
	)

	for row, sample_result in zip(data, per_sample_results):
		is_correct = bool(sample_result.get("is_correct", False))

		condition = _extract_condition_type(row)
		condition_stats[condition]["total"] += 1
		if is_correct:
			condition_stats[condition]["correct"] += 1

		belief_types = _extract_belief_types(row)
		for belief_type in belief_types:
			belief_stats[belief_type]["total"] += 1
			if is_correct:
				belief_stats[belief_type]["correct"] += 1

		pair_key = _build_pair_key(row)
		if "true_belief" in belief_types:
			pair_stats[pair_key]["tb_total"] += 1
			if is_correct:
				pair_stats[pair_key]["tb_correct"] += 1
		if "false_belief" in belief_types:
			pair_stats[pair_key]["fb_total"] += 1
			if is_correct:
				pair_stats[pair_key]["fb_correct"] += 1

	by_condition = {k: _safe_div(v["correct"], v["total"]) for k, v in condition_stats.items()}
	by_belief_type = {k: _safe_div(v["correct"], v["total"]) for k, v in belief_stats.items()}

	tb_fb_total_pairs = 0
	tb_fb_correct_pairs = 0
	for stats in pair_stats.values():
		if stats["tb_total"] <= 0 or stats["fb_total"] <= 0:
			continue
		tb_fb_total_pairs += 1
		if stats["tb_correct"] == stats["tb_total"] and stats["fb_correct"] == stats["fb_total"]:
			tb_fb_correct_pairs += 1

	tb_and_fb_accuracy = _safe_div(tb_fb_correct_pairs, tb_fb_total_pairs)
	by_belief_type["tb_and_fb"] = tb_and_fb_accuracy

	belief_type_counts = {k: v["total"] for k, v in belief_stats.items()}
	belief_type_counts["tb_and_fb"] = tb_fb_total_pairs

	secondary_metrics = {f"by_condition.{k}": v for k, v in by_condition.items()}
	secondary_metrics.update({f"by_belief_type.{k}": v for k, v in by_belief_type.items()})

	return {
		"accuracy": _safe_div(correct, total),
		"correct": correct,
		"total": total,
		**secondary_metrics,
		"by_condition": by_condition,
		"condition_counts": {k: v["total"] for k, v in condition_stats.items()},
		"by_belief_type": by_belief_type,
		"belief_type_counts": belief_type_counts,
		"per_sample_results": per_sample_results,
	}

