"""EmoBench 评测脚本（变长 MCQ）。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
TASKS_DIR = Path(__file__).resolve().parents[1]

for path in (ROOT_DIR, TASKS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src import runner
from EmoBench.metrics import compute_metrics, _get_gold_answer
from EmoBench.prompts import build_prompt

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _validate_row(row: Dict[str, Any]) -> bool:
    question = str(row.get("Question", "") or "").strip()
    answer = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    choice_texts = meta.get("choice_texts", [])
    correct = answer.get("Correct_Answer", [])
    wrong = answer.get("Wrong_Answer", [])

    return (
        bool(question)
        and isinstance(choice_texts, list)
        and len(choice_texts) >= 2
        and isinstance(correct, list)
        and len(correct) == 1
        and isinstance(wrong, list)
        and len(choice_texts) == len(correct) + len(wrong)
    )


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [row for row in data if _validate_row(row)]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"Warning: skipped {skipped} invalid rows.")
    if not valid:
        raise RuntimeError(
            "没有可评测样本：EmoBench 数据需要包含 Question、Answer.{Correct_Answer, Wrong_Answer} 和 Meta.choice_texts。"
        )
    return valid


def get_gold_answers(data: List[Dict[str, Any]]) -> List[str]:
    return [_get_gold_answer(row) for row in data]


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/EmoBench/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    args = parser.parse_args()
    experiment_config = runner.load_experiment_config(args.experiment_config)
    print(f"Experiment config: {args.experiment_config}")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)
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
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {experiment_config['repeats']}")

    prompts = [build_prompt(row, prompt_method) for row in data]
    all_prompts = [prompts for _ in range(experiment_config["repeats"])]

    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    results = client.batch_generate_structure(flat_prompts, schema)

    all_metrics = []
    all_results = []
    gold_answers = get_gold_answers(data)

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [r.content.answer if r.content else None for r in repeat_results]

        metrics = compute_metrics(predictions, gold_answers, data, judge_client)
        all_metrics.append(metrics)
        print(
            f"Run {i+1}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(data))


if __name__ == "__main__":
    main()
