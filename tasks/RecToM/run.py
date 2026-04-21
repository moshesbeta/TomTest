"""RecToM 评测脚本（多标签 MCQ）。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from RecToM.prompts import build_prompt
from RecToM.metrics import compute_metrics

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _normalize_prediction(prediction: Any) -> List[str]:
    if prediction is None:
        return []
    if isinstance(prediction, str):
        items = [prediction]
    elif isinstance(prediction, (list, tuple, set)):
        items = list(prediction)
    else:
        items = [prediction]

    normalized = []
    seen = set()
    for item in items:
        token = str(item).strip().upper()
        if token and token not in seen:
            normalized.append(token)
            seen.add(token)
    return sorted(normalized)


def _validate_row(row: Dict[str, Any]) -> bool:
    answer = row.get("Answer")
    if not isinstance(answer, dict):
        return False
    correct = answer.get("Correct_Answer")
    wrong = answer.get("Wrong_Answer")
    question = row.get("Question")
    return (
        isinstance(correct, list)
        and isinstance(wrong, list)
        and bool(correct)
        and isinstance(question, str)
        and question.strip() != ""
    )


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [row for row in data if _validate_row(row)]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"Warning: skipped {skipped} invalid rows.")
    if not valid:
        raise RuntimeError("没有可评测样本：RecToM 数据需要包含 Question 和 Answer.{Correct_Answer, Wrong_Answer}。")
    return valid


def get_gold_answers(data: List[Dict[str, Any]]) -> List[List[str]]:
    """提取标准答案（列表形式）"""
    gold_answers = []
    for row in data:
        answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
        correct_list = answer_block.get("Correct_Answer", [])
        gold = sorted(str(x).strip().upper() for x in correct_list if str(x).strip())
        gold_answers.append(gold)
    return gold_answers


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/RecToM/config.yaml")
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)

    # Judge client (not used for RecToM, but for consistency)
    judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)

    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['path']}")
    data = preprocess_data(data)
    print(f"Valid samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    prompts = [build_prompt(row, prompt_method) for row in data]
    all_prompts = [prompts for _ in range(experiment_config["repeats"])]

    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    results = client.batch_generate_structure(flat_prompts, schema)

    all_predictions: List[List[List[str]]] = []
    all_metrics = []
    all_results = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [_normalize_prediction(r.content.answer) if r.content else [] for r in repeat_results]
        all_predictions.append(predictions)

        gold_answers = get_gold_answers(data)

        metrics = compute_metrics(predictions, gold_answers, data)
        all_metrics.append(metrics)
        print(
            f"Run {i+1}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"FullCorrect={metrics['full_correct']}/{metrics['total']}, "
            f"PartialNoError={metrics['partial_no_error']}, "
            f"HasError={metrics['has_error']}"
        )

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=[[list(g) for g in get_gold_answers(data)] for _ in range(experiment_config["repeats"])],
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(data))


if __name__ == "__main__":
    main()
