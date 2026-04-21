"""FollowBench 评测脚本（基于结构化输出）"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src import runner
from src.dataloader import load_dataset

from FollowBench.prompts import build_prompt
from FollowBench.metrics import compute_metrics


def _build_evolution_paths(all_data: List[Dict]) -> Dict[Tuple, Dict[int, str]]:
    """Build evolution path lookup: {(constraint_type, example_group_id): {level: instruction}}"""
    paths: Dict[Tuple, Dict[int, str]] = {}
    for row in all_data:
        meta = row["Meta"]
        key = (meta["constraint_type"], meta["example_group_id"])
        if key not in paths:
            paths[key] = {}
        paths[key][meta["constraint_level"]] = row["Question"]
    return paths


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("tasks/FollowBench/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)

    # 创建 Judge 客户端（如果配置了）
    judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)
    if judge_client:
        judge_config = experiment_config.get("judge_config", {})
        print(f"Judge model: {judge_config.get('model_name', 'unknown')}")
    else:
        print("No judge config — LLM eval items will be skipped (rule-based only)")

    # Load ALL data (levels 0-5); max_samples applies to eval items (level > 0)
    all_data = load_dataset(
        subset=dataset_config["path"],
        datasets_root=experiment_config["datasets_path"],
    )
    print(f"Loaded {len(all_data)} total samples (all levels) from {dataset_config['path']}")

    # Build evolution paths from full data
    evolution_paths = _build_evolution_paths(all_data)

    # Eval items: level 1-5 only
    eval_data = [row for row in all_data if row["Meta"]["constraint_level"] > 0]
    max_samples = experiment_config["max_samples"]
    if max_samples and max_samples > 0:
        eval_data = eval_data[:max_samples]

    print(f"Eval items (level 1-5): {len(eval_data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {experiment_config['repeats']}")

    # Build prompts
    prompts = [build_prompt(row, prompt_method) for row in eval_data]
    all_prompts = [prompts for _ in range(experiment_config["repeats"])]

    # 使用结构化输出
    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    all_responses = client.batch_generate_structure(flat_prompts, schema)

    # Compute metrics per repeat
    all_metrics = []
    all_results = []

    for i in range(experiment_config["repeats"]):
        start = i * len(eval_data)
        end = start + len(eval_data)
        responses = all_responses[start:end]
        all_results.append(responses)

        metrics = compute_metrics(responses, eval_data, evolution_paths, judge_client)
        all_metrics.append(metrics)

        print(f"\nRun {i+1}:")
        print(f"  Mean HSR: {metrics['accuracy']:.4f} | CSL: {metrics['csl']:.2f}")
        print(f"  HSR by level: " + "  ".join(
            f"L{lv}={v:.1%}" for lv, v in sorted(metrics['hsr_by_level'].items()) if v is not None
        ))
        print(f"  Rule-based: {metrics['rule_eval_count']} | LLM judge: {metrics['llm_eval_count']}")

    # Save results (gold_answers are targets, mostly empty for FollowBench)
    gold_answers = [(row["Answer"]["Correct_Answer"] or [""])[0] for row in eval_data]
    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in eval_data],
    )

    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(eval_data))


if __name__ == "__main__":
    main()
