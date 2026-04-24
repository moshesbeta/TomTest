"""FANToM evaluation runner.

This task mixes 7 question types (6 original + derived beliefQAs_choice) across 3 blocks:
- qa (OpenAnswer): beliefQAs, factQA
- binary (MCQAnswer2): answerabilityQAs_binary, infoAccessibilityQAs_binary, beliefQAs_choice
- list (MultiLabelAnswer): answerabilityQA_list, infoAccessibilityQA_list

Only binary and list questions are shuffled (deterministically) per repeat.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from FANToM.prompts import build_prompt
from FANToM.metrics import compute_metrics


_GROUP_MAP = {
    "factQA": "qa",
    "beliefQAs": "qa",
    "beliefQAs_choice": "binary",
    "answerabilityQAs_binary": "binary",
    "infoAccessibilityQAs_binary": "binary",
    "answerabilityQA_list": "list",
    "infoAccessibilityQA_list": "list",
}


def _meta_id(row: Mapping[str, Any]) -> str:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    return str(meta.get("id", "") or "")


def _snippet_id_from_meta_id(meta_id: str) -> str:
    return meta_id.split("__")[0] if "__" in meta_id else meta_id or "unknown"


def _question_type_from_meta_id(meta_id: str) -> str:
    if "__" not in meta_id:
        return "unknown"
    parts = meta_id.split("__")
    return parts[1] if len(parts) >= 2 else "unknown"


def _get_correct_wrong_for_belief_choice(row: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    ans = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    ca = ans.get("Correct_Answer", [])
    wa = ans.get("Wrong_Answer", [])
    if not isinstance(ca, list) or len(ca) != 1:
        return None
    if not isinstance(wa, list) or len(wa) < 1:
        return None
    correct = str(ca[0]).strip()
    wrong = str(wa[0]).strip()
    if not correct or not wrong:
        return None
    return correct, wrong


def _get_fact_q_a(row: Mapping[str, Any]) -> Tuple[str, str]:
    q = str(row.get("Question", "") or "").strip()
    ans = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    ca = ans.get("Correct_Answer", [])
    if isinstance(ca, list) and ca:
        a = str(ca[0]).strip()
    else:
        a = ""
    return q, a


def _preprocess_rows(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in raw:
        out = dict(row)
        mid = _meta_id(out)
        qtype = _question_type_from_meta_id(mid)
        out["question_type"] = qtype
        out["_snippet_id"] = _snippet_id_from_meta_id(mid)
        out["_group"] = _GROUP_MAP.get(qtype, "unknown")
        rows.append(out)

    # Build snippet -> fact info mapping for Target/Information prompts
    snip_to_fact: Dict[str, Tuple[str, str]] = {}
    for r in rows:
        if r.get("question_type") == "factQA":
            snip = str(r.get("_snippet_id") or "unknown")
            snip_to_fact[snip] = _get_fact_q_a(r)

    for r in rows:
        snip = str(r.get("_snippet_id") or "unknown")
        fact_q, fact_a = snip_to_fact.get(snip, ("", ""))
        qtype = str(r.get("question_type") or "unknown")
        if qtype.startswith("answerability"):
            r["fact_question"] = fact_q
            r["fact_answer"] = ""
        elif qtype.startswith("infoAccessibility"):
            r["fact_question"] = fact_q
            r["fact_answer"] = fact_a

    # Inject base MCQ definitions for original binary/list types
    skipped = 0
    for r in rows:
        qtype = str(r.get("question_type") or "unknown")
        if qtype in {"answerabilityQAs_binary", "infoAccessibilityQAs_binary"}:
            ans = r.get("Answer", {}) if isinstance(r.get("Answer"), dict) else {}
            ca = ans.get("Correct_Answer", [])
            correct = str(ca[0]).strip().lower() if isinstance(ca, list) and ca else ""
            if correct.startswith("yes"):
                gold_letter = "A"
            elif correct.startswith("no"):
                gold_letter = "B"
            else:
                gold_letter = ""
            r["_mcq"] = {
                "original_choices": {"A": "yes", "B": "no"},
                "choices": {"A": "yes", "B": "no"},
                "gold_letter": gold_letter,
            }
        elif qtype in {"answerabilityQA_list", "infoAccessibilityQA_list"}:
            correct = r.get("Answer", {}).get("Correct_Answer", []) if isinstance(r.get("Answer"), dict) else []
            wrong = r.get("Answer", {}).get("Wrong_Answer", []) if isinstance(r.get("Answer"), dict) else []
            correct_list = [str(x).strip() for x in correct] if isinstance(correct, list) else []
            wrong_list = [str(x).strip() for x in wrong] if isinstance(wrong, list) else []
            candidates = [x for x in (correct_list + wrong_list) if x]
            if len(candidates) > 26:
                skipped += 1
                r["_skip"] = True
                continue
            r["_mcq"] = {
                "original_options": candidates,
                "gold_texts": correct_list,
            }

    if skipped:
        print(f"Warning: skipped {skipped} list rows (more than 26 options).")

    rows = [r for r in rows if not r.get("_skip")]

    # Derive beliefQAs_choice
    derived: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("question_type") != "beliefQAs":
            continue
        pair = _get_correct_wrong_for_belief_choice(r)
        if pair is None:
            continue
        correct, wrong = pair
        out = dict(r)
        out["question_type"] = "beliefQAs_choice"
        out["_group"] = "binary"

        meta = out.get("Meta", {}) if isinstance(out.get("Meta"), dict) else {}
        meta_id = str(meta.get("id", "") or "")
        snip = _snippet_id_from_meta_id(meta_id)
        suffix = meta_id.split("__")[-1] if "__" in meta_id else str(len(derived))
        meta2 = dict(meta)
        meta2["id"] = f"{snip}__beliefQAs_choice__{suffix}"
        out["Meta"] = meta2
        out["_snippet_id"] = snip

        out["_mcq"] = {
            "original_choices": {"A": correct, "B": wrong},
            "choices": {"A": correct, "B": wrong},
            "gold_letter": "A",
        }
        derived.append(out)

    return rows + derived


def _shuffle_ab(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    swap = rng.random() < 0.5
    original = mcq.get("original_choices", {}) if isinstance(mcq.get("original_choices"), dict) else {}
    if not swap:
        return {**mcq, "choices": dict(original), "gold_letter": str(mcq.get("gold_letter", "") or "").strip().upper()}
    original_gold = str(mcq.get("gold_letter", "") or "").strip().upper()
    new_gold = "B" if original_gold == "A" else ("A" if original_gold == "B" else "")
    return {
        **mcq,
        "choices": {"A": original.get("B", ""), "B": original.get("A", "")},
        "gold_letter": new_gold,
    }


def _shuffle_list(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    options = list(mcq.get("original_options", []) or [])
    gold_texts = set(str(x).strip() for x in (mcq.get("gold_texts", []) or []) if str(x).strip())

    indices = list(range(len(options)))
    rng.shuffle(indices)
    shuffled = [options[i] for i in indices]

    # label as A, B, C, ...
    labels = [chr(ord("A") + i) for i in range(len(shuffled))]
    choices = {lab: txt for lab, txt in zip(labels, shuffled)}
    gold_labels = [lab for lab, txt in zip(labels, shuffled) if str(txt).strip() in gold_texts]

    return {**mcq, "choices": choices, "gold_labels": gold_labels}


def _gold_for_save(row: Dict[str, Any]) -> Any:
    qtype = str(row.get("question_type") or "unknown")
    if qtype in {"answerabilityQAs_binary", "infoAccessibilityQAs_binary", "beliefQAs_choice"}:
        mcq = row.get("_mcq", {}) if isinstance(row.get("_mcq"), dict) else {}
        return mcq.get("gold_letter", "")
    if qtype in {"answerabilityQA_list", "infoAccessibilityQA_list"}:
        mcq = row.get("_mcq", {}) if isinstance(row.get("_mcq"), dict) else {}
        return mcq.get("gold_labels", [])
    correct = row.get("Answer", {}).get("Correct_Answer", []) if isinstance(row.get("Answer"), dict) else []
    if isinstance(correct, list) and correct:
        return str(correct[0])
    return ""


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/FANToM/config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    args = parser.parse_args()
    experiment_config = runner.load_experiment_config(args.experiment_config)
    print(f"Experiment config: {args.experiment_config}")

    prompt_method = dataset_config["method"]
    schema_cfg = dataset_config["schema"]
    schemas = runner.load_schema(schema_cfg)

    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)
    judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)

    raw_data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    base_data = _preprocess_rows(raw_data)
    n = len(base_data)
    print(f"Loaded {len(raw_data)} raw rows from {dataset_config['path']}")
    print(f"After deriving beliefQAs_choice: {n} rows")
    print(f"Prompt method: {prompt_method}")
    print(f"Schemas: {schema_cfg}")
    print(f"Repeats: {experiment_config['repeats']} (binary/list shuffled deterministically)")

    repeats = experiment_config["repeats"]
    all_results: List[List[Any]] = []
    all_prompts: List[List[str]] = []
    all_gold: List[List[Any]] = []
    all_metrics: List[Dict[str, Any]] = []

    for i in range(repeats):
        rows_i: List[Dict[str, Any]] = []
        prompts_i: List[str] = []
        gold_i: List[Any] = []

        prompts_qa: List[str] = []
        idx_qa: List[int] = []
        prompts_bin: List[str] = []
        idx_bin: List[int] = []
        prompts_list: List[str] = []
        idx_list: List[int] = []

        for j, row in enumerate(base_data):
            group = str(row.get("_group") or "unknown")
            out = dict(row)
            if group == "binary":
                out["_mcq"] = _shuffle_ab(out.get("_mcq", {}), seed=42 * (i + 1) + j)
            elif group == "list":
                out["_mcq"] = _shuffle_list(out.get("_mcq", {}), seed=42 * (i + 1) + j)

            rows_i.append(out)
            prompt = build_prompt(out, prompt_method)
            prompts_i.append(prompt)
            gold_i.append(_gold_for_save(out))

            if group == "qa":
                idx_qa.append(j)
                prompts_qa.append(prompt)
            elif group == "binary":
                idx_bin.append(j)
                prompts_bin.append(prompt)
            elif group == "list":
                idx_list.append(j)
                prompts_list.append(prompt)

        # Run inference per block schema and stitch back
        repeat_results = [None] * n
        if prompts_qa:
            res_qa = client.batch_generate_structure(prompts_qa, schemas["OpenAnswer"])
            for k, ridx in enumerate(idx_qa):
                repeat_results[ridx] = res_qa[k]
        if prompts_bin:
            res_bin = client.batch_generate_structure(prompts_bin, schemas["MCQAnswer2"])
            for k, ridx in enumerate(idx_bin):
                repeat_results[ridx] = res_bin[k]
        if prompts_list:
            res_list = client.batch_generate_structure(prompts_list, schemas["MultiLabelAnswer"])
            for k, ridx in enumerate(idx_list):
                repeat_results[ridx] = res_list[k]

        all_results.append(repeat_results)
        all_prompts.append(prompts_i)
        all_gold.append(gold_i)

        preds = [r.content.answer if r is not None and r.content is not None else None for r in repeat_results]
        metrics = compute_metrics(preds, gold_i, rows_i, judge_client)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in base_data],
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
