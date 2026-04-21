"""IFEval 评测脚本

流程：
1. 从 datasets/IFEval/test 加载数据（HuggingFace arrow 格式）
2. 让模型现场生成回答（batch_generate，普通文本生成）
3. 用 instruction_following_eval 检查器计算 strict/loose 指标
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from IFEval.prompts import build_prompt
from IFEval.metrics import compute_metrics

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    dataset_config = runner.load_dataset_config("tasks/IFEval/config.yaml")
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    prompt_method = dataset_config["method"]

    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)

    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['path']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    prompts = [build_prompt(row, prompt_method) for row in data]
    all_prompts = [prompts for _ in range(experiment_config["repeats"])]

    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    responses = client.batch_generate(flat_prompts)

    all_predictions = []
    all_metrics = []
    all_results = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        predictions = responses[start:end]
        all_predictions.append(predictions)
        all_results.append(predictions)

        metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(
            f"Run {i+1}: "
            f"prompt_strict={metrics['prompt_accuracy_strict']:.4f}  "
            f"prompt_loose={metrics['prompt_accuracy_loose']:.4f}  "
            f"instr_strict={metrics['instruction_accuracy_strict']:.4f}  "
            f"instr_loose={metrics['instruction_accuracy_loose']:.4f}"
        )

    gold_answers = [row["Question"] for row in data]

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    avg_strict = sum(m["prompt_accuracy_strict"] for m in all_metrics) / len(all_metrics)
    avg_loose = sum(m["prompt_accuracy_loose"] for m in all_metrics) / len(all_metrics)
    print(f"\n{'='*50}")
    print(f"Results (Average over {experiment_config['repeats']} runs)")
    print(f"{'='*50}")
    print(f"Prompt-level Strict:  {avg_strict:.4f}")
    print(f"Prompt-level Loose:   {avg_loose:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
