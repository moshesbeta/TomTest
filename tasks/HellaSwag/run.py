"""HellaSwag 评测脚本（结构化 MCQAnswer；Overall / In-domain / Zero-shot）。"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from HellaSwag.metrics import compute_metrics
from HellaSwag.prompts import build_prompt

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def build_mcq_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """按 HellaSwag 当前数据格式构造 _mcq；不满足 1 正 + 3 误则返回 None。"""
    story = row.get("Story")
    if not isinstance(story, dict):
        return None

    ans = row.get("Answer")
    if not isinstance(ans, dict):
        return None

    ca = ans.get("Correct_Answer")
    wa = ans.get("Wrong_Answer")
    if len(ca) != 1 or len(wa) != 3:
        return None

    correct = str(ca[0]).strip()
    wrong = [str(x).strip() for x in wa]
    endings = [correct] + wrong
    if len(endings) != 4:
        return None

    return {
        "context": story.get("full_story"),
        "question": row.get("Question", ""),
        "endings": endings,
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
        raise RuntimeError("没有可评测样本：数据需包含 Story/Question/Answer 且 Answer 为 1 Correct_Answer + 3 Wrong_Answer。")
    return valid


def shuffle_endings(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """返回 endings 被打乱的 _mcq 副本，同时更新 gold_letter。"""
    rng = random.Random(seed)
    endings = list(mcq["endings"])
    indices = list(range(len(endings)))
    rng.shuffle(indices)

    new_endings = [endings[i] for i in indices]
    old_gold_idx = 0  # 初始 gold_letter 固定为 A，对应 index 0
    new_gold_idx = indices.index(old_gold_idx)
    new_gold = ["A", "B", "C", "D"][new_gold_idx]

    return {**mcq, "endings": new_endings, "gold_letter": new_gold}


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取标准答案字母"""
    mcq = row.get("_mcq") or {}
    return mcq.get("gold_letter", "")


def main() -> None:
    dataset_config = runner.load_dataset_config("tasks/HellaSwag/config.yaml")
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    prompt_method = dataset_config["method"]
    schema = runner.load_schema(dataset_config["schema"])

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)

    # 创建 Judge 客户端（如果配置了）
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
    print(f"Repeats: {repeats} (each with deterministic option shuffle)")

    all_prompts: List[List[str]] = []
    repeat_data: List[List[Dict[str, Any]]] = []

    for i in range(repeats):
        rows_i: List[Dict[str, Any]] = []
        prompts: List[str] = []
        for j, row in enumerate(data):
            shuffled = shuffle_endings(row["_mcq"], seed=42 * (i + 1) + j)
            out = dict(row)
            out["_mcq"] = shuffled
            rows_i.append(out)
            prompts.append(build_prompt(out, prompt_method))
        repeat_data.append(rows_i)
        all_prompts.append(prompts)

    # 批量推理
    flat_prompts = [p for repeat_prompts in all_prompts for p in repeat_prompts]
    print(f"Running inference ({len(flat_prompts)} prompts)...")
    results = client.batch_generate_structure(flat_prompts, schema)

    all_metrics = []
    all_results = []

    for i in range(repeats):
        start = i * n
        end = start + n
        rows_i = repeat_data[i]
        repeat_results = results[start:end]
        all_results.append(repeat_results)
        predictions = [r.content.answer if r.content else None for r in repeat_results]

        gold_answers = [row["_mcq"]["gold_letter"] for row in rows_i]

        metrics = compute_metrics(predictions, gold_answers, rows_i, judge_client)
        all_metrics.append(metrics)

        print(
            f"Run {i+1}: "
            f"Overall={metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']}), "
        )

    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results,
        all_prompts=all_prompts,
        gold_answers=[[row["_mcq"]["gold_letter"] for row in repeat_data[i]] for i in range(repeats)],
        all_metrics=all_metrics,
        sample_metas=[row.get("Meta") for row in data],
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
