"""ToMChallenges metrics：Overall Accuracy + 按 Meta.order 分组的二级准确率。"""
from typing import Any, Dict, List


def _safe_div(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]], judge_client=None) -> Dict[str, Any]:
    """计算 ToMChallenges 的 metrics（直接匹配选项字母 A/B）。"""
    if len(predictions) != len(data):
        raise ValueError(f"predictions/data length mismatch: {len(predictions)} vs {len(data)}")

    total = len(predictions)
    correct = 0
    per_sample_results = []

    by_order_stats: Dict[str, Dict[str, int]] = {}

    for pred, row in zip(predictions, data):
        mcq = row.get("_mcq") or {}
        gold = mcq.get("gold_letter")
        hit = bool(pred) and bool(gold) and pred == gold
        correct += int(hit)
        per_sample_results.append({
            "is_correct": hit,
            "error_reason": None if hit else ("content_none" if not pred else "wrong_answer"),
        })

        meta = row.get("Meta") if isinstance(row.get("Meta"), dict) else {}
        order_key = str((meta.get("order") if isinstance(meta, dict) else None) or "unknown")

        if order_key not in by_order_stats:
            by_order_stats[order_key] = {"correct": 0, "total": 0}
        by_order_stats[order_key]["total"] += 1
        if hit:
            by_order_stats[order_key]["correct"] += 1

    accuracy = _safe_div(correct, total)
    by_order = {k: _safe_div(v["correct"], v["total"]) for k, v in by_order_stats.items()}
    order_counts = {k: v["total"] for k, v in by_order_stats.items()}

    secondary_metrics = {f"by_order.{k}": v for k, v in by_order.items()}

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "by_order": by_order,
        "order_counts": order_counts,
        "per_sample_results": per_sample_results,
    }