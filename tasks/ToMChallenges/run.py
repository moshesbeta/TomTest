"""ToMChallenges 评测脚本（基于结构化输出）"""
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from ToMChallenges.prompts import build_prompt
from ToMChallenges.metrics import compute_metrics


def _extract_ab_answers(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """提取 (correct, wrong)。不符合 1 正 + 1 误则返回 None。"""
    ans = row.get("Answer")
    ca = ans.get("Correct_Answer")
    wa = ans.get("Wrong_Answer")
    if len(ca) != 1 or len(wa) != 1:
        return None
    return str(ca[0]).strip(), str(wa[0]).strip()


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为每行注入 _mcq.original_choices 与 gold_letter（初始 A 为正确答案）。"""
    valid: List[Dict[str, Any]] = []
    skipped = 0

    for row in data:
        pair = _extract_ab_answers(row)
        if pair is None:
            skipped += 1
            continue
        correct, wrong = pair
        out = dict(row)
        out["_mcq"] = {
            "original_choices": {"A": correct, "B": wrong},
            "choices": {"A": correct, "B": wrong},
            "gold_letter": "A",
        }
        valid.append(out)

    if skipped:
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 1 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：数据需包含 Answer 且为 1 Correct_Answer + 1 Wrong_Answer。")
    return valid


def shuffle_ab_choices(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """对 A/B 两个选项做 deterministic shuffle，并同步 gold_letter。"""
    rng = random.Random(seed)
    swap = rng.random() < 0.5

    original = mcq["original_choices"]
    if not swap:
        return {**mcq, "choices": dict(original), "gold_letter": "A"}

    return {
        **mcq,
        "choices": {"A": original["B"], "B": original["A"]},
        "gold_letter": "B",
    }


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("tasks/ToMChallenges/config.yaml")

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

    print(f"Loaded {len(data)} raw rows from {dataset_config['path']}")
    data = preprocess_mcq(data)

    print(f"MCQ samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {experiment_config['repeats']} (each with deterministic A/B shuffle)")

    n = len(data)
    all_prompts = []
    all_gold = []

    # 为每个 repeat 生成 shuffled 数据和 prompts
    for i in range(experiment_config["repeats"]):
        rows_i = []
        gold_i = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_ab_choices(row["_mcq"], seed=42 * (i + 1) + j)
            out = dict(row)
            out["_mcq"] = shuffled_mcq
            rows_i.append(out)
            all_prompts.append(build_prompt(out, prompt_method))
            gold_i.append(shuffled_mcq["gold_letter"])
        all_gold.append(gold_i)

    # 批量推理
    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    # 计算 metrics
    all_metrics = []
    all_results = []

    for i in range(experiment_config["repeats"]):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [r.content.answer if r.content else None for r in repeat_results]

        # 重新构建当前 repeat 的 shuffled data
        shuffled_data = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_ab_choices(row["_mcq"], seed=42 * (i + 1) + j)
            out = dict(row)
            out["_mcq"] = shuffled_mcq
            shuffled_data.append(out)

        metrics = compute_metrics(predictions, shuffled_data, judge_client)
        all_metrics.append(metrics)
        print(
            f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

    # 保存结果
    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=[all_prompts[i*n:(i+1)*n] for i in range(experiment_config["repeats"])],
        gold_answers=all_gold,
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, experiment_config["repeats"], n)


if __name__ == "__main__":
    main()