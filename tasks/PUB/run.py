"""PUB 评测（结构化多选，按样本动态支持 2/3/4/5 选项）。"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

TASKS_ROOT = Path(__file__).parent.parent
REPO_ROOT = TASKS_ROOT.parent
sys.path.insert(0, str(TASKS_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src import runner
from src.llm.client import LLMResponse
from PUB.prompts import build_prompt
from PUB.metrics import compute_metrics

SCHEMA_BY_OPTION_COUNT = {
    2: "MCQAnswer2",
    3: "MCQAnswer3",
    4: "MCQAnswer",
    5: "MCQAnswer5",
}

def _format_background(background: Any) -> str:
    if isinstance(background, str):
        return background.strip()
    if isinstance(background, (list, tuple)):
        parts = [str(item).strip() for item in background if str(item).strip()]
        return "\n".join(parts)
    return ""


def _story_to_prompt_text(story: Dict[str, Any]) -> str:
    parts: List[str] = []
    full_story = str(story.get("full_story", "")).strip()
    summary = str(story.get("summary", "")).strip()
    background = _format_background(story.get("background", []))

    if full_story:
        parts.append(full_story)
    if summary:
        parts.append(f"Summary: {summary}")
    if background:
        parts.append(f"Background: {background}")

    return "\n".join(parts).strip()


def build_mcq_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """按 1 正 + N 误（N in [1, 4]）的 schema 构造统一 MCQ。"""
    story = row.get("Story")
    answer = row.get("Answer")
    if not isinstance(story, dict) or not isinstance(answer, dict):
        return None

    correct = answer.get("Correct_Answer")
    wrong = answer.get("Wrong_Answer")
    if not isinstance(correct, list) or not isinstance(wrong, list):
        return None
    if len(correct) != 1 or len(wrong) not in (1, 2, 3, 4):
        return None

    correct_text = str(correct[0]).strip()
    wrong_texts = [str(item).strip() for item in wrong]
    if not correct_text or any(not item for item in wrong_texts):
        return None

    option_count = 1 + len(wrong_texts)
    letters = list("ABCDE")[:option_count]
    texts = [correct_text] + wrong_texts
    original_choices = {letters[i]: texts[i] for i in range(option_count)}

    return {
        "story": _story_to_prompt_text(story),
        "question": str(row.get("Question", "")).strip(),
        "original_choices": original_choices,
        "gold_letter": "A",
        "option_count": option_count,
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
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 1-4 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：数据需包含 1 个正确答案和 1-4 个错误答案。")
    return valid


def shuffle_mcq_options(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """打乱选项，并同步更新 gold_letter。"""
    rng = random.Random(seed)
    letters = sorted(mcq["original_choices"].keys())
    texts = [mcq["original_choices"][letter] for letter in letters]
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


def _run_batched_by_schema(
    client: Any,
    prompts: List[str],
    option_counts: List[int],
    schema_map: Dict[str, Any],
) -> List[LLMResponse]:
    """按样本选项数分组调用不同 schema，并保持原顺序返回。"""
    assert len(prompts) == len(option_counts), "prompts 与 option_counts 长度须一致"

    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, option_count in enumerate(option_counts):
        grouped_indices[option_count].append(idx)

    ordered_results: List[Optional[LLMResponse]] = [None] * len(prompts)
    for option_count in sorted(grouped_indices):
        schema_name = SCHEMA_BY_OPTION_COUNT.get(option_count)
        if schema_name is None or schema_name not in schema_map:
            raise ValueError(f"Unsupported option_count={option_count}")
        indices = grouped_indices[option_count]
        batch_prompts = [prompts[idx] for idx in indices]
        print(f"Running inference for option_count={option_count} ({len(batch_prompts)} prompts)...")
        batch_results = client.batch_generate_structure(batch_prompts, schema_map[schema_name])
        for idx, result in zip(indices, batch_results):
            ordered_results[idx] = result

    if any(result is None for result in ordered_results):
        raise RuntimeError("Some PUB results are missing after batched inference.")

    return [result for result in ordered_results if result is not None]


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/PUB/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default="experiment_config.yaml")
    args = parser.parse_args()
    experiment_config = runner.load_experiment_config(args.experiment_config)
    print(f"Experiment config: {args.experiment_config}")

    prompt_method = dataset_config["method"]
    schemas = runner.load_schema(dataset_config["schema"])

    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)
    judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)

    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['path']}")
    data = preprocess_mcq(data)

    repeats = experiment_config["repeats"]
    n = len(data)
    print(f"MCQ samples: {n}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {repeats} (each with different option shuffle)")
    print(f"Option count distribution: {dict(sorted(Counter(row['_mcq']['option_count'] for row in data).items()))}")

    all_prompts: List[List[str]] = []
    repeat_data: List[List[Dict[str, Any]]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        shuffled_rows: List[Dict[str, Any]] = []
        prompts: List[str] = []
        gold_answers: List[str] = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_mcq_options(row["_mcq"], seed=307 * (i + 1) + j)
            shuffled_row = dict(row)
            shuffled_row["_mcq"] = shuffled_mcq
            shuffled_rows.append(shuffled_row)
            prompts.append(build_prompt(shuffled_row, prompt_method))
            gold_answers.append(shuffled_mcq["gold_letter"])
        repeat_data.append(shuffled_rows)
        all_prompts.append(prompts)
        all_gold.append(gold_answers)

    flat_prompts = [prompt for repeat_prompts in all_prompts for prompt in repeat_prompts]
    flat_option_counts = [
        row["_mcq"]["option_count"]
        for rows in repeat_data
        for row in rows
    ]
    print(f"Running inference ({len(flat_prompts)} prompts total)...")
    results = _run_batched_by_schema(client, flat_prompts, flat_option_counts, schemas)

    all_metrics: List[Dict[str, Any]] = []
    all_results: List[List[LLMResponse]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        rows = repeat_data[i]
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [result.content.answer if result.content else None for result in repeat_results]
        gold_answers = all_gold[i]

        metrics = compute_metrics(predictions, gold_answers, rows, judge_client)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
