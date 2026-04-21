"""数据集评测运行器公共函数

提供数据集评测脚本之间的共享逻辑，减少重复代码。
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml

from src.dataloader import load_dataset
from src.llm import LLMClient
from src.llm.client import LLMResponse


def _serialize_llm_response(response: LLMResponse) -> Dict[str, Any]:
    """序列化 LLMResponse 对象为字典

    Args:
        response: LLMResponse 对象

    Returns:
        包含 content 和 reasoning 的字典
    """
    content = response.content
    if content is None:
        serialized_content = None
    elif hasattr(content, "model_dump"):
        # Pydantic 模型
        serialized_content = content.model_dump()
    elif isinstance(content, dict):
        serialized_content = content
    else:
        # 字符串或其他类型
        serialized_content = content
    
    return {
        "content": serialized_content,
        "reasoning": response.reasoning,
    }


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """加载数据集配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        数据集配置字典（dataset, path, method, schema, system_prompt）
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

        return {
            "dataset": config["dataset"],
            "path": config["path"],
            "method": config["method"],
            "schema": config["schema"],  # 必须指定 schema
            "system_prompt": config.get("system_prompt", ""),  # 可选的 system prompt
            "use_llm_judge": config.get("use_llm_judge", None),  # 可选，覆盖 experiment_config 的 use_llm_judge
        }


def load_schema(schema_name: Optional[str]) -> Optional[Type]:
    """根据名称动态加载 schema 类

    Args:
        schema_name: schema 类名，如 "MCQAnswer", "OpenAnswer" 等

    Returns:
        Pydantic BaseModel 类，如果 schema_name 为 None 则返回 None
    """
    if schema_name is None:
        return None

    from src.schemas import (
        MCQAnswer,
        MCQAnswer3,
        MCQAnswer3Lower,
        OpenAnswer,
        OneWordAnswer,
        JudgeAnswer,
        MultiLabelAnswer,
    )

    schema_map = {
        "MCQAnswer": MCQAnswer,
        "MCQAnswer3": MCQAnswer3,
        "MCQAnswer3Lower": MCQAnswer3Lower,
        "OpenAnswer": OpenAnswer,
        "OneWordAnswer": OneWordAnswer,
        "JudgeAnswer": JudgeAnswer,
        "MultiLabelAnswer": MultiLabelAnswer,
    }

    if schema_name not in schema_map:
        raise ValueError(
            f"Unknown schema: {schema_name}. "
            f"Available schemas: {list(schema_map.keys())}"
        )

    return schema_map[schema_name]


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """加载实验配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        实验配置字典（llm_config, repeats, max_samples, datasets_path, results_path, judge_config 等）
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return {
            "llm_config": config.get("llm", {}),
            "repeats": config.get("repeats", 1),
            "max_samples": config.get("max_samples", 0),
            "datasets_path": config.get("datasets_path", "datasets"),
            "results_path": config.get("results_path", "results"),
            "judge_config": config.get("judge", {}),  # 覆盖数据集的 judge 配置
        }


def create_llm_client(llm_config: Dict[str, Any], dataset_config: Optional[Dict[str, Any]] = None) -> Any:
    """创建 LLM 客户端（统一使用结构化输出）

    Args:
        llm_config: 配置字典，包含 model_name, api_key, api_url, temperature, max_tokens 等
        dataset_config: 可选的数据集配置，用于覆盖 system_prompt

    Returns:
        StructureClient 实例
    """
    config = llm_config.copy()
    # 优先使用 dataset_config 中的 system_prompt
    if dataset_config and dataset_config.get("system_prompt"):
        config["system_prompt"] = dataset_config["system_prompt"]
    from src.llm import StructureClient
    return StructureClient.from_config(config)


def create_judge_client(judge_config: Dict[str, Any], dataset_config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """创建 Judge 客户端（用于 LLM judge）

    Args:
        judge_config: judge 配置字典，包含 model_name, api_key, api_url, use_llm_judge 等
        dataset_config: 可选的数据集配置，其中 use_llm_judge 字段（非 None 时）会覆盖 judge_config 的设置

    Returns:
        StructureClient 实例，如果 use_llm_judge 为 False 则返回 None
    """
    use_llm_judge = judge_config.get("use_llm_judge", False)
    # dataset 层的 use_llm_judge 非 None 时覆盖 experiment_config 的设置
    if dataset_config is not None and dataset_config.get("use_llm_judge") is not None:
        use_llm_judge = dataset_config["use_llm_judge"]
    if not use_llm_judge:
        return None
    return create_llm_client(judge_config)


def _compute_average_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算所有 metrics 的平均值

    Args:
        all_metrics: 所有重复运行的 metrics 列表

    Returns:
        平均 metrics 字典
    """
    if not all_metrics:
        return {}

    avg_metrics = {}

    for key in all_metrics[0].keys():
        values = [m.get(key) for m in all_metrics if key in m]
        if not values:
            continue

        first_value = values[0]
        if isinstance(first_value, (int, float)):
            # 数值类型直接计算平均
            avg_metrics[key] = sum(v for v in values if isinstance(v, (int, float))) / len(values)
        elif isinstance(first_value, dict):
            # 字典类型递归计算平均
            sub_avg = {}
            sub_keys = set()
            for v in values:
                if isinstance(v, dict):
                    sub_keys.update(v.keys())

            for sub_key in sub_keys:
                sub_values = [v.get(sub_key) for v in values if isinstance(v, dict) and sub_key in v]
                if sub_values and isinstance(sub_values[0], (int, float)):
                    sub_avg[sub_key] = sum(sub_values) / len(sub_values)

            avg_metrics[key] = sub_avg

    return avg_metrics


def save_common_results(
    dataset_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    all_results: List[List[LLMResponse]],
    all_prompts: List[List[str]],
    gold_answers: Union[List[str], List[List[str]]],
    all_metrics: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    sample_metas: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Tuple[Path, Path, Path]:
    """保存评测结果

    Args:
        dataset_config: 数据集配置字典，必须包含 dataset, method, results_path 等字段
        experiment_config: 实验配置字典，必须包含 llm_config 等字段
        all_results: 所有重复运行的 LLMResponse 列表 [repeat][sample]
        all_prompts: 所有重复运行的 prompt 列表 [repeat][sample]
        gold_answers: 标准答案。支持两种格式：
            - List[str]: 所有 repeat 共用同一组 gold（如 ToMBench）
            - List[List[str]]: 每个 repeat 有独立 gold（如 Tomato 选项 shuffle）
        all_metrics: 所有重复运行的 metrics 列表
        metadata: 额外元数据（如 judge_model）

    Returns:
        (config_path, metrics_path, prediction_path) 元组
    """
    # 从配置中提取所需字段
    dataset_name = dataset_config["dataset"]
    model = experiment_config["llm_config"]["model_name"]
    prompt_method = dataset_config["method"]
    results_path = experiment_config["results_path"]

    per_repeat_gold = bool(gold_answers and isinstance(gold_answers[0], list))

    # 创建目录结构: results/dataset_name/model/exp_{timestamp}/
    results_dir = Path(results_path)
    timestamp = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir / dataset_name / model / f"exp_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存 config.json - 包含所有配置信息
    config_data = {
        "dataset": dataset_name,
        "model": model,
        "prompt_method": prompt_method,
        "repeats": len(all_metrics),
    }

    # 添加 dataset_config 内容（排除非 JSON 可序列化对象）
    if dataset_config:
        dataset_config_copy = dict(dataset_config)
        config_data["dataset_config"] = dataset_config_copy

    # 添加 experiment_config 内容（排除敏感信息）
    if experiment_config:
        experiment_config_copy = dict(experiment_config)
        # 过滤敏感信息
        if "llm_config" in experiment_config_copy:
            llm_config_copy = dict(experiment_config_copy["llm_config"])
            llm_config_copy.pop("api_key", None)
            llm_config_copy.pop("api_url", None)
            experiment_config_copy["llm_config"] = llm_config_copy
        if "judge_config" in experiment_config_copy:
            judge_config_copy = dict(experiment_config_copy["judge_config"])
            judge_config_copy.pop("api_key", None)
            judge_config_copy.pop("api_url", None)
            experiment_config_copy["judge_config"] = judge_config_copy
        config_data["experiment_config"] = experiment_config_copy

    if metadata:
        config_data["metadata"] = metadata

    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. 保存 metrics.json
    avg_metrics = _compute_average_metrics(all_metrics)
    metrics_data = {
        "avg_metrics": avg_metrics,
        "all_metrics": all_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 3. 保存 prediction.jsonl
    prediction_path = output_dir / "prediction.jsonl"
    with open(prediction_path, "w", encoding="utf-8") as f:
        for repeat_idx, repeat_results in enumerate(all_results):
            repeat_gold = gold_answers[repeat_idx] if per_repeat_gold else gold_answers
            repeat_prompts = all_prompts[repeat_idx] if repeat_idx < len(all_prompts) else None
            repeat_metrics = all_metrics[repeat_idx]
            per_sample_results = repeat_metrics.get("per_sample_results", [])

            for sample_idx, (result, gold) in enumerate(zip(repeat_results, repeat_gold)):
                record = {
                    "repeat": repeat_idx,
                    "sample_idx": sample_idx,
                    "gold_answer": gold,
                    "pred": _serialize_llm_response(result),
                }

                # 添加 prompt
                if repeat_prompts and sample_idx < len(repeat_prompts):
                    record["prompt"] = repeat_prompts[sample_idx]

                # 添加 meta
                if sample_metas and sample_idx < len(sample_metas) and sample_metas[sample_idx] is not None:
                    record["meta"] = sample_metas[sample_idx]

                # 添加 is_correct 和 error_reason
                if per_sample_results and sample_idx < len(per_sample_results):
                    record["is_correct"] = per_sample_results[sample_idx]["is_correct"]
                    record["error_reason"] = per_sample_results[sample_idx]["error_reason"]

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Results saved to: {output_dir}")
    print(f"  - config.json")
    print(f"  - metrics.json")
    print(f"  - prediction.jsonl")

    return config_path, metrics_path, prediction_path


def print_summary_stats(
    all_metrics: List[Dict[str, Any]],
    repeats: int,
    total_samples: int,
) -> None:
    """打印统计摘要

    Args:
        all_metrics: 所有重复运行的 metrics 列表
        repeats: 重复次数
        total_samples: 总样本数
    """
    avg_accuracy = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
    avg_correct = sum(m["correct"] for m in all_metrics) / len(all_metrics)

    print(f"\n{'='*50}")
    print(f"Results (Average over {repeats} runs)")
    print(f"{'='*50}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Correct: {avg_correct:.1f}/{total_samples}")
    print(f"{'='*50}")


def load_and_limit_data(
    subset: str,
    datasets_path: str = "datasets",
    max_samples: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """加载数据并限制样本数

    Args:
        subset: 数据集子集路径
        datasets_path: 数据集根目录
        max_samples: 最大样本数（0 表示不限制）
        seed: 随机种子，用于可复现的随机抽样

    Returns:
        数据列表
    """
    import random

    data = load_dataset(subset, datasets_root=datasets_path)
    if max_samples > 0:
        random.seed(seed)
        data = random.sample(data, min(max_samples, len(data)))
    return data
