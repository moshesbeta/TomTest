"""ToMBench 评测脚本（基于结构化输出）"""
import sys
from pathlib import Path
from typing import Any, Dict, List

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataloader import load_dataset
from src.llm import LLMClient
from src import runner

from ToMBench.prompts import build_prompt
from ToMBench.metrics import compute_metrics


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("tasks/ToMBench/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)

    # 创建 Judge 客户端（如果配置了）
    judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)

    # 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['path']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {experiment_config['repeats']}")

    # 提前准备 gold_answers
    gold_answers = [row['Answer']['Correct Answer'][0] for row in data]

    # 构建 prompts（每个 repeat 构建相同的 prompts）
    prompts = [build_prompt(row, prompt_method) for row in data]
    all_prompts = [prompts for _ in range(experiment_config["repeats"])]

    # 批量推理
    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    results = client.batch_generate_structure(flat_prompts, schema)

    # 使用数据集的 metrics 函数计算
    all_metrics = []
    all_results = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [r.content.answer if r.content else None for r in repeat_results]

        # 调用数据集的 metrics 函数
        metrics = compute_metrics(predictions, gold_answers, data, judge_client)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    # 保存结果
    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(gold_answers))


if __name__ == "__main__":
    main()
