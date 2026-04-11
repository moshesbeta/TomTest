"""Tomato 评测（结构化 MCQAnswer）。数据：TomDatasets Tomato，1 正 + 3 误；非此形态行跳过。"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from Tomato.prompts import get_template, build_prompt
from Tomato.metrics import compute_metrics
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _story_to_prompt_text(story: Dict[str, Any]) -> str:
    parts: List[str] = []
    if story.get("full_story"):
        parts.append(str(story["full_story"]))
    if story.get("summary"):
        parts.append(f"Summary: {story['summary']}")
    if story.get("background"):
        bg = story["background"]
        parts.append(f"Background: {json.dumps(bg, ensure_ascii=False) if isinstance(bg, (dict, list)) else bg}")
    return "\n".join(parts).strip()


def build_mcq_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """按 TomDatasets 固定 schema 构造 _mcq；不满足 1 正 + 3 误则返回 None。"""
    story = row.get("Story")
    if not isinstance(story, dict):
        return None
    ans = row.get("Answer")
    if not isinstance(ans, dict):
        return None
    ca = ans.get("Correct_Answer")
    wa = ans.get("Wrong_Answer")
    if not isinstance(ca, list) or not isinstance(wa, list):
        return None
    if len(ca) != 1 or len(wa) != 3:
        return None

    correct = str(ca[0]).strip()
    wrong = [str(x).strip() for x in wa]
    letters = ["A", "B", "C", "D"]
    texts = [correct] + wrong
    original_choices = {letters[i]: texts[i] for i in range(4)}

    return {
        "story": _story_to_prompt_text(story),
        "question": str(row.get("Question", "")).strip(),
        "original_choices": original_choices,
        "gold_letter": "A",
    }


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    skipped = 0
    for row in data:
        mcq = build_mcq_from_row(row)
        if mcq is None:
            skipped += 1
            continue
        out = dict(row)
        out["_mcq"] = mcq
        valid.append(out)
    if skipped:
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 3 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：数据需为 TomDatasets Tomato 标准字段与 1+3 选项。")
    return valid


def shuffle_mcq_options(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """返回一份选项顺序被打乱的 _mcq 副本，gold_letter 同步更新。"""
    rng = random.Random(seed)
    letters = sorted(mcq["original_choices"].keys())
    texts = [mcq["original_choices"][l] for l in letters]
    old_gold_idx = letters.index(mcq["gold_letter"])

    indices = list(range(len(letters)))
    rng.shuffle(indices)

    new_choices: Dict[str, str] = {}
    new_gold = mcq["gold_letter"]
    for new_pos, old_idx in enumerate(indices):
        new_choices[letters[new_pos]] = texts[old_idx]
        if old_idx == old_gold_idx:
            new_gold = letters[new_pos]

    return {**mcq, "original_choices": new_choices, "gold_letter": new_gold}


def main() -> None:
    dataset_config = runner.load_dataset_config("Tomato/config.yaml")
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]
    template = get_template(prompt_method)
    client = runner.create_llm_client(experiment_config["llm_config"])

    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['subset']}")
    data = preprocess_mcq(data)

    repeats = experiment_config["repeats"]
    print(f"MCQ samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {repeats} (each with different option shuffle)")

    all_prompts: List[str] = []
    repeat_data: List[List[Dict[str, Any]]] = []

    for i in range(repeats):
        shuffled_rows: List[Dict[str, Any]] = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_mcq_options(row["_mcq"], seed=42 * (i + 1) + j)
            shuffled_row = dict(row)
            shuffled_row["_mcq"] = shuffled_mcq
            shuffled_rows.append(shuffled_row)
            all_prompts.append(build_prompt(template, shuffled_row))
        repeat_data.append(shuffled_rows)

    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    n = len(data)
    all_predictions: List[List[str]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        rows = repeat_data[i]
        predictions = [r.answer for r in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, rows)
        all_metrics.append(metrics)
        all_gold.append([row["_mcq"]["gold_letter"] for row in rows])
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
