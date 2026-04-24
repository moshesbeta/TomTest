"""FANToM metrics.

FANToM mixes 7 task types (6 original + derived beliefQAs_choice) grouped into 3 blocks:
- qa: beliefQAs, factQA (open-ended, judged by LLM judge)
- binary: answerabilityQAs_binary, infoAccessibilityQAs_binary, beliefQAs_choice (A/B choices)
- list: answerabilityQA_list, infoAccessibilityQA_list (multi-label choices)

Return format follows docs/add_new_dataset.md:
- must include accuracy/correct/total/per_sample_results
- detailed metrics are organized under by_category with five major categories:
  overall, belief, answerability, infoaccess, fact
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.schemas import JudgeAnswer
from src.utils import compute_sample_metrics


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _token_f1(a: str, b: str) -> float:
    a_tokens = (a or "").split()
    b_tokens = (b or "").split()
    common = Counter(a_tokens) & Counter(b_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(b_tokens) if b_tokens else 0.0
    recall = num_same / len(a_tokens) if a_tokens else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _snippet_id(row: Dict[str, Any]) -> str:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    meta_id = str(meta.get("id", "") or "")
    return meta_id.split("__")[0] if "__" in meta_id else meta_id or "unknown"


def _question_type(row: Dict[str, Any]) -> str:
    qtype = row.get("question_type")
    if qtype:
        return str(qtype)
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    meta_id = str(meta.get("id", "") or "")
    if "__" in meta_id:
        parts = meta_id.split("__")
        if len(parts) >= 2:
            return parts[1]
    return "unknown"


def _get_correct_texts(row: Dict[str, Any]) -> List[str]:
    ans = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    ca = ans.get("Correct_Answer", [])
    if isinstance(ca, list):
        return [str(x) for x in ca if str(x).strip()]
    if ca is None:
        return []
    return [str(ca)]



def _binary_letter_to_choice_text(row: Dict[str, Any], letter: Optional[str]) -> Optional[str]:
    if not letter:
        return None
    mcq = row.get("_mcq", {}) if isinstance(row.get("_mcq"), dict) else {}
    choices = mcq.get("choices", {}) if isinstance(mcq.get("choices"), dict) else {}
    s = choices.get(str(letter).strip().upper())
    return str(s).strip() if s is not None else None


def _normalize_yesno(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s in {"true", "t"}:
        return "yes"
    if s in {"false", "f"}:
        return "no"
    return None


def _normalize_ab_letter(x: Any) -> Optional[str]:
    """Normalize a model output into 'A'/'B'."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"A", "B"}:
        return s
    if s and s[0] in {"A", "B"}:
        return s[0]
    return None


def _normalize_multilabel_set(x: Any) -> Optional[frozenset]:
    """Normalize multi-label output into an order-insensitive comparable set.

    - None -> None (so compute_sample_metrics reports content_none)
    - Keep single-letter A-Z labels only (consistent with MultiLabelAnswer)
    """
    if x is None:
        return None
    if isinstance(x, str):
        raw = [x]
    elif isinstance(x, (list, tuple, set)):
        raw = list(x)
    else:
        raw = [x]

    labels: List[str] = []
    for item in raw:
        token = str(item).strip().upper()
        if len(token) == 1 and token.isalpha():
            labels.append(token)
            continue
        if token:
            c = token[0]
            if c.isalpha():
                labels.append(c.upper())

    return frozenset(labels)


def _weighted_f1_binary(pred: Sequence[Optional[str]], gold: Sequence[Optional[str]]) -> float:
    labels = ("no", "yes")
    supports: Dict[str, int] = {k: 0 for k in labels}
    tp: Dict[str, int] = {k: 0 for k in labels}
    fp: Dict[str, int] = {k: 0 for k in labels}
    fn: Dict[str, int] = {k: 0 for k in labels}

    for p, g in zip(pred, gold):
        if g not in labels:
            continue
        supports[g] += 1
        if p == g:
            tp[g] += 1
        else:
            fn[g] += 1
            if p in labels:
                fp[p] += 1

    total_support = sum(supports.values())
    if total_support == 0:
        return 0.0

    weighted = 0.0
    for k in labels:
        support = supports[k]
        if support == 0:
            continue
        precision = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) else 0.0
        recall = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        weighted += f1 * support
    return weighted / total_support


def _eval_qa_with_llm_judge(
    predictions: List[Optional[str]],
    data: List[Dict[str, Any]],
    judge_client: Any,
) -> Tuple[List[bool], List[float], List[str]]:
    if judge_client is None:
        raise ValueError("FANToM QA tasks require a judge_client. Enable judge.use_llm_judge or set dataset use_llm_judge.")

    judge_prompts: List[str] = []
    gold_texts: List[str] = []
    for pred, row in zip(predictions, data):
        story_block = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
        context = str(story_block.get("full_story", "") or story_block.get("summary", "") or "").strip()
        question = str(row.get("Question", "") or "").strip()
        correct_list = _get_correct_texts(row)
        gold = correct_list[0] if correct_list else ""
        gold_texts.append(gold)

        pred_str = pred if pred is not None else "(no answer)"
        judge_prompts.append(
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Ground Truth:\n"
            f"{gold}\n\n"
            "Model Answer:\n"
            f"{pred_str}\n\n"
            "Is the model answer correct? Output a JSON with exactly 'True' if correct, 'False' otherwise."
        )

    judge_results = judge_client.batch_generate_structure(judge_prompts, JudgeAnswer)

    is_correct_list: List[bool] = []
    token_f1_list: List[float] = []
    for pred, gold, jr in zip(predictions, gold_texts, judge_results):
        if pred is None:
            is_correct = False
        else:
            is_correct = bool(jr and jr.content and jr.content.answer == "True")
        is_correct_list.append(is_correct)
        token_f1_list.append(_token_f1(str(gold).lower(), str(pred or "").lower()))

    return is_correct_list, token_f1_list, gold_texts


def _eval_fact_token_f1(predictions: List[Optional[str]], data: List[Dict[str, Any]]) -> float:
    f1s: List[float] = []
    for pred, row in zip(predictions, data):
        if pred is None:
            f1s.append(0.0)
            continue
        correct_list = _get_correct_texts(row)
        if not correct_list:
            f1s.append(0.0)
            continue
        pred_norm = str(pred).lower()
        f1s.append(max(_token_f1(str(g).lower(), pred_norm) for g in correct_list))
    return sum(f1s) / len(f1s) if f1s else 0.0


def _aggregate_all_by_snippet(
    rows: List[Dict[str, Any]],
    is_correct: List[bool],
    required_types: Sequence[str],
) -> float:
    by_snip: Dict[str, Dict[str, bool]] = {}
    for row, ok in zip(rows, is_correct):
        snip = row.get("_snippet_id") or _snippet_id(row)
        qtype = _question_type(row)
        if snip not in by_snip:
            by_snip[snip] = {}

        if qtype not in by_snip[snip]:
            by_snip[snip][qtype] = True
        by_snip[snip][qtype] = by_snip[snip][qtype] and bool(ok)

    included = 0
    hits = 0
    req = list(required_types)
    for snip, m in by_snip.items():
        if not all(t in m for t in req):
            continue
        included += 1
        if all(m[t] for t in req):
            hits += 1
    return hits / included if included else 0.0


def compute_metrics(
    predictions: List[Any],
    gold_answers: List[Any],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    if len(predictions) != len(data):
        raise ValueError(f"predictions/data length mismatch: {len(predictions)} vs {len(data)}")

    total = len(data)

    # Global per-sample bookkeeping (filled by buckets)
    global_is_correct: List[bool] = [False] * total
    global_error_reason: List[Optional[str]] = [None] * total

    # --------------------
    # Bucketize (single pass)
    # --------------------
    belief_rows: List[Dict[str, Any]] = []
    belief_preds: List[Optional[str]] = []
    belief_gidx: List[int] = []

    fact_rows: List[Dict[str, Any]] = []
    fact_preds: List[Optional[str]] = []
    fact_gidx: List[int] = []

    choice_pred: List[Optional[str]] = []
    choice_gold: List[str] = []
    choice_gidx: List[int] = []

    ans_bin_pred: List[Optional[str]] = []
    ans_bin_gold: List[str] = []
    ans_bin_pred_yesno: List[Optional[str]] = []
    ans_bin_gold_yesno: List[Optional[str]] = []
    ans_bin_gidx: List[int] = []

    info_bin_pred: List[Optional[str]] = []
    info_bin_gold: List[str] = []
    info_bin_pred_yesno: List[Optional[str]] = []
    info_bin_gold_yesno: List[Optional[str]] = []
    info_bin_gidx: List[int] = []

    ans_list_pred: List[Optional[frozenset]] = []
    ans_list_gold: List[frozenset] = []
    ans_list_gidx: List[int] = []

    info_list_pred: List[Optional[frozenset]] = []
    info_list_gold: List[frozenset] = []
    info_list_gidx: List[int] = []

    for i, (pred, row) in enumerate(zip(predictions, data)):
        qtype = _question_type(row)
        pred_str = pred if isinstance(pred, str) or pred is None else str(pred)

        if qtype == "beliefQAs":
            belief_rows.append(row)
            belief_preds.append(pred_str)
            belief_gidx.append(i)
            continue
        if qtype == "factQA":
            fact_rows.append(row)
            fact_preds.append(pred_str)
            fact_gidx.append(i)
            continue

        mcq = row.get("_mcq", {}) if isinstance(row.get("_mcq"), dict) else {}
        if qtype == "beliefQAs_choice":
            choice_pred.append(_normalize_ab_letter(pred_str))
            choice_gold.append(_normalize_ab_letter(mcq.get("gold_letter")) or "")
            choice_gidx.append(i)
            continue
        if qtype == "answerabilityQAs_binary":
            pred_letter = _normalize_ab_letter(pred_str)
            gold_letter = _normalize_ab_letter(mcq.get("gold_letter")) or ""
            ans_bin_pred.append(pred_letter)
            ans_bin_gold.append(gold_letter)
            ans_bin_pred_yesno.append(_normalize_yesno(_binary_letter_to_choice_text(row, pred_letter)))
            ans_bin_gold_yesno.append(_normalize_yesno(_binary_letter_to_choice_text(row, gold_letter)))
            ans_bin_gidx.append(i)
            continue
        if qtype == "infoAccessibilityQAs_binary":
            pred_letter = _normalize_ab_letter(pred_str)
            gold_letter = _normalize_ab_letter(mcq.get("gold_letter")) or ""
            info_bin_pred.append(pred_letter)
            info_bin_gold.append(gold_letter)
            info_bin_pred_yesno.append(_normalize_yesno(_binary_letter_to_choice_text(row, pred_letter)))
            info_bin_gold_yesno.append(_normalize_yesno(_binary_letter_to_choice_text(row, gold_letter)))
            info_bin_gidx.append(i)
            continue

        if qtype == "answerabilityQA_list":
            gold_labels = mcq.get("gold_labels", [])
            ans_list_pred.append(_normalize_multilabel_set(pred))
            ans_list_gold.append(_normalize_multilabel_set(gold_labels) or frozenset())
            ans_list_gidx.append(i)
            continue
        if qtype == "infoAccessibilityQA_list":
            gold_labels = mcq.get("gold_labels", [])
            info_list_pred.append(_normalize_multilabel_set(pred))
            info_list_gold.append(_normalize_multilabel_set(gold_labels) or frozenset())
            info_list_gidx.append(i)
            continue

        # unknown: mark wrong/content_none so overall accuracy includes them
        global_is_correct[i] = False
        global_error_reason[i] = "content_none" if pred is None else "wrong_answer"

    # ---- QA with judge (beliefQAs, factQA) ----
    belief_qa_acc = 0.0
    belief_qa_token_f1 = 0.0
    belief_qa_token_f1_when_correct = 0.0
    fact_qa_acc = 0.0
    fact_token_f1 = 0.0

    if belief_rows:
        ok_list, f1_list, _ = _eval_qa_with_llm_judge(belief_preds, belief_rows, judge_client)
        for gidx, ok, pred in zip(belief_gidx, ok_list, belief_preds):
            global_is_correct[gidx] = ok
            global_error_reason[gidx] = None if ok else ("content_none" if pred is None else "wrong_answer")
        belief_qa_acc = _safe_div(sum(ok_list), len(ok_list))
        belief_qa_token_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        f1_when_correct = [f for f, ok in zip(f1_list, ok_list) if ok]
        belief_qa_token_f1_when_correct = sum(f1_when_correct) / len(f1_when_correct) if f1_when_correct else 0.0

    if fact_rows:
        ok_list, _, _ = _eval_qa_with_llm_judge(fact_preds, fact_rows, judge_client)
        for gidx, ok, pred in zip(fact_gidx, ok_list, fact_preds):
            global_is_correct[gidx] = ok
            global_error_reason[gidx] = None if ok else ("content_none" if pred is None else "wrong_answer")
        fact_qa_acc = _safe_div(sum(ok_list), len(ok_list))
        fact_token_f1 = _eval_fact_token_f1(fact_preds, fact_rows)

    # ---- Binary A/B accuracy via compute_sample_metrics + weighted f1 for yes/no tasks ----
    belief_choice_acc = 0.0
    if choice_gidx:
        sample = compute_sample_metrics(
            predictions=choice_pred,
            gold_answers=choice_gold,
            is_correct_fn=lambda p, g: bool(p) and bool(g) and p == g,
        )
        belief_choice_acc = _safe_div(sample["correct"], sample["total"])
        for gidx, r in zip(choice_gidx, sample["per_sample_results"]):
            global_is_correct[gidx] = r["is_correct"]
            global_error_reason[gidx] = r["error_reason"]

    ans_bin_acc = 0.0
    ans_bin_wf1 = 0.0
    if ans_bin_gidx:
        sample = compute_sample_metrics(
            predictions=ans_bin_pred,
            gold_answers=ans_bin_gold,
            is_correct_fn=lambda p, g: bool(p) and bool(g) and p == g,
        )
        ans_bin_acc = _safe_div(sample["correct"], sample["total"])
        for gidx, r in zip(ans_bin_gidx, sample["per_sample_results"]):
            global_is_correct[gidx] = r["is_correct"]
            global_error_reason[gidx] = r["error_reason"]
        ans_bin_wf1 = _weighted_f1_binary(ans_bin_pred_yesno, ans_bin_gold_yesno)

    info_bin_acc = 0.0
    info_bin_wf1 = 0.0
    if info_bin_gidx:
        sample = compute_sample_metrics(
            predictions=info_bin_pred,
            gold_answers=info_bin_gold,
            is_correct_fn=lambda p, g: bool(p) and bool(g) and p == g,
        )
        info_bin_acc = _safe_div(sample["correct"], sample["total"])
        for gidx, r in zip(info_bin_gidx, sample["per_sample_results"]):
            global_is_correct[gidx] = r["is_correct"]
            global_error_reason[gidx] = r["error_reason"]
        info_bin_wf1 = _weighted_f1_binary(info_bin_pred_yesno, info_bin_gold_yesno)

    # ---- List tasks: strict set match via compute_sample_metrics (pred_set == gold_set) ----
    ans_list_acc = 0.0
    if ans_list_gidx:
        sample = compute_sample_metrics(predictions=ans_list_pred, gold_answers=ans_list_gold)
        ans_list_acc = _safe_div(sample["correct"], sample["total"])
        for gidx, r in zip(ans_list_gidx, sample["per_sample_results"]):
            global_is_correct[gidx] = r["is_correct"]
            global_error_reason[gidx] = r["error_reason"]

    info_list_acc = 0.0
    if info_list_gidx:
        sample = compute_sample_metrics(predictions=info_list_pred, gold_answers=info_list_gold)
        info_list_acc = _safe_div(sample["correct"], sample["total"])
        for gidx, r in zip(info_list_gidx, sample["per_sample_results"]):
            global_is_correct[gidx] = r["is_correct"]
            global_error_reason[gidx] = r["error_reason"]

    # ---- Aggregate ALL metrics ----
    # overall.ALL: 5 tasks (origin-style) excluding factQA and beliefQAs(qa)
    overall_all = _aggregate_all_by_snippet(
        data,
        global_is_correct,
        required_types=[
            "beliefQAs_choice",
            "answerabilityQA_list",
            "answerabilityQAs_binary",
            "infoAccessibilityQA_list",
            "infoAccessibilityQAs_binary",
        ],
    )

    # overall.ALL_star: 6 tasks excluding factQA, including beliefQAs(qa)
    overall_all_star = _aggregate_all_by_snippet(
        data,
        global_is_correct,
        required_types=[
            "beliefQAs",
            "beliefQAs_choice",
            "answerabilityQA_list",
            "answerabilityQAs_binary",
            "infoAccessibilityQA_list",
            "infoAccessibilityQAs_binary",
        ],
    )

    answerability_all = _aggregate_all_by_snippet(
        data,
        global_is_correct,
        required_types=["answerabilityQA_list", "answerabilityQAs_binary"],
    )
    infoaccess_all = _aggregate_all_by_snippet(
        data,
        global_is_correct,
        required_types=["infoAccessibilityQA_list", "infoAccessibilityQAs_binary"],
    )

    correct = sum(1 for x in global_is_correct if x)
    accuracy = _safe_div(correct, total)

    per_sample_results = []
    for row, ok, err in zip(data, global_is_correct, global_error_reason):
        per_sample_results.append(
            {
                "is_correct": bool(ok),
                "error_reason": None if ok else (err or "wrong_answer"),
                "question_type": _question_type(row),
                "_group": row.get("_group"),
                "_snippet_id": row.get("_snippet_id") or _snippet_id(row),
            }
        )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_sample_results": per_sample_results,
        "by_category": {
            "overall.ALL": overall_all,
            "overall.ALL_star": overall_all_star,
            "belief.qa.accuracy": belief_qa_acc,
            "belief.qa.token_f1": belief_qa_token_f1,
            "belief.qa.token_f1_when_correct": belief_qa_token_f1_when_correct,
            "belief.choice.accuracy": belief_choice_acc,
            "answerability.ALL": answerability_all,
            "answerability.list.accuracy": ans_list_acc,
            "answerability.yn.accuracy": ans_bin_acc,
            "answerability.yn.weighted_f1": ans_bin_wf1,
            "infoaccess.ALL": infoaccess_all,
            "infoaccess.list.accuracy": info_list_acc,
            "infoaccess.yn.accuracy": info_bin_acc,
            "infoaccess.yn.weighted_f1": info_bin_wf1,
            "fact.qa.accuracy": fact_qa_acc,
            "fact.token_f1": fact_token_f1,
        },
    }
