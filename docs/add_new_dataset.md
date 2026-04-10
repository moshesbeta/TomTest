# 新增数据集指南

本指南介绍如何在 TomTest 框架中添加一个新的数据集评测。

## 概述

添加新数据集需要创建 4 个文件：
1. `config.yaml` - 数据集配置（固定参数）
2. `schemas.py` - 输出 Schema
3. `prompts.py` - Prompt 模板
4. `metrics.py` - Metrics 计算（包括二级指标）
5. `run.py` - 评测脚本

然后更新顶层 `run_all.py` 注册数据集。

## 步骤 1：创建数据集目录

```bash
cd /path/to/TomTest
mkdir MyDataset
cd MyDataset
```

## 步骤 2：创建 config.yaml

`config.yaml` 定义数据集的**固定参数**：

```yaml
# MyDataset/config.yaml
dataset: MyDataset
path: MyDataset/test

# 结构化输出 schema（从数据集自己的 schemas.py 导入）
schema: MCQAnswer  # 或 OpenAnswer / YesNoAnswer / JudgeAnswer

# 默认 prompt 方法
default_prompt: zero_shot
```

### Schema 选择

在数据集的 `schemas.py` 中定义需要的 schema。

| Schema | 使用场景 | 字段 |
|---|---|---|
| `MCQAnswer` | 多选题 | `answer: Literal["A", "B", "C", "D"]` |
| `OpenAnswer` | 开放式问答 | `answer: str` |
| `YesNoAnswer` | 是非题 | `answer: Literal["YES", "NO"]` |
| `MultipleChoice` | 多选题（任意数量选项） | `answer: List[Literal["A", "B", "C", "D"]]` |
| `JudgeAnswer` | LLM Judge 答案 | `answer: Literal["CORRECT", "INCORRECT"]` |

如果需要自定义 schema，在数据集的 `schemas.py` 中添加，然后在 `config.yaml` 中引用。

**注意**：实验参数（model、repeats、max_samples 等）统一在 `experiment_config.yaml` 中配置。

## 步骤 2.5：创建 schemas.py

`schemas.py` 定义数据集的输出 schema，使用字典格式：

```python
"""MyDataset schemas"""
from pydantic import BaseModel
from typing import Literal


class MCQAnswer(BaseModel):
    """多选题答案 schema"""
    answer: Literal["A", "B", "C", "D"]


class JudgeAnswer(BaseModel):
    """LLM Judge 答案 schema（可选，供 metrics.py 内部调用）"""
    answer: Literal["CORRECT", "INCORRECT"]


# schema 字典：config.yaml 中引用主 schema，其他 schema 供内部调用
SCHEMAS = {
    "MCQAnswer": MCQAnswer,
    "JudgeAnswer": JudgeAnswer,  # 如需 LLM Judge
}
```

**说明**：
- `config.yaml` 中只指定主 schema（用于模型推理）
- 其他 schema（如 `JudgeAnswer`）可在 `metrics.py` 中直接 import 使用

## 步骤 3：创建 prompts.py

`prompts.py` 只定义 prompt 模板，不包含 schema 相关代码。

```python
"""MyDataset prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": """You are an expert.

Context: {context}
Question: {question}

Options:
{options}

Answer with exactly one letter (A/B/C/D):""",

    "cot": """You are an expert.

Context: {context}
Question: {question}

Options:
{options}

Let's think step by step...
Answer with exactly one letter (A/B/C/D):""",
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板

    Args:
        method: prompt 方法名称

    Returns:
        prompt 模板字符串
    """
    return PROMPTS.get(method, PROMPTS["zero_shot"])


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 prompt

    Args:
        template: prompt 模板
        row: 数据行

    Returns:
        格式化的 prompt
    """
    # 从数据行提取字段
    context = row.get("context", "")
    question = row.get("question", "")
    options = row.get("options", [])

    # 构建选项文本
    options_text = "\n".join([
        f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
    ])

    return template.format(context=context, question=question, options=options_text)
```

## 步骤 4：创建 metrics.py

`metrics.py` 定义指标计算函数，包括基础指标和二级指标。

### 模板 1：直接匹配（MCQAnswer/OpenAnswer）

```python
"""MyDataset metrics 计算"""
from typing import Any, Dict, List


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 MyDataset 的 metrics（直接匹配）

    Args:
        predictions: 模型预测答案列表（从结构化输出提取的 answer 字段）
        data: 原始数据列表

    Returns:
        包含基础指标和二级指标的字典
    """
    gold_answers = [row.get('answer', '') for row in data]

    # 基础指标
    correct = sum(1 for p, g in zip(predictions, gold_answers) if p == g)
    accuracy = correct / len(predictions) if predictions else 0

    # 二级指标：按类别分组（可选）
    category_metrics = {}
    for pred, gold, row in zip(predictions, gold_answers, data):
        category = row.get("category", "unknown")

        if category not in category_metrics:
            category_metrics[category] = {"correct": 0, "total": 0}
        category_metrics[category]["total"] += 1
        if pred == gold:
            category_metrics[category]["correct"] += 1

    # 计算各类别准确率
    secondary_metrics = {
        f"by_category.{cat}": stats["correct"] / stats["total"]
        for cat, stats in category_metrics.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
        **secondary_metrics,
        "by_category": {cat: stats["correct"] / stats["total"] for cat, stats in category_metrics.items()},
        "category_counts": {cat: stats["total"] for cat, stats in category_metrics.items()},
    }
```

### 模板 2：LLM Judge

如果需要 LLM judge（如 ToMi），在自己的 `compute_metrics` 中使用 judge client：

```python
"""MyDataset metrics 计算（LLM Judge 版）"""
from typing import Any, Dict, List
from MyDataset.schemas import JudgeAnswer  # 直接从 schemas.py 导入


def compute_metrics(
    predictions: List[str],
    data: List[Dict[str, Any]],
    judge_client,
) -> Dict[str, Any]:
    """计算 MyDataset 的 metrics（使用 LLM Judge）

    Args:
        predictions: 模型预测答案列表
        data: 原始数据列表
        judge_client: LLM 客户端，用于 judge

    Returns:
        包含基础指标的字典
    """
    # 批量构建 judge prompts
    judge_prompts = []
    for pred, row in zip(predictions, data):
        gold = row.get("gold_answer", "")
        context = row.get("context", "")
        question = row.get("question", "")

        judge_prompt = f"""Determine if the answer is correct.

Context: {context}
Question: {question}
Ground Truth: {gold}
Model Answer: {pred}

Output CORRECT or INCORRECT:"""
        judge_prompts.append(judge_prompt)

    # 批量结构化输出
    judge_results = judge_client.batch_generate_structure(judge_prompts, JudgeAnswer)

    # 统计正确数
    correct = sum(1 for result in judge_results if result.answer == "CORRECT")

    accuracy = correct / len(predictions) if predictions else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions),
    }
```

## 步骤 5：创建 run.py

`run.py` 是评测脚本的主入口。

### 模板 1：直接匹配（MCQAnswer/OpenAnswer）

```python
"""MyDataset 评测脚本"""
import sys
from pathlib import Path

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner

from MyDataset.prompts import get_template, build_prompt
from MyDataset.metrics import compute_metrics


def extract_gold_answers(data):
    """提取标准答案"""
    return [row.get("answer", "") for row in data]


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("MyDataset/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    # 从 config 获取 schema（已由 load_dataset_config 动态加载）
    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]

    # 获取 prompt 模板
    template = get_template(prompt_method)

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"])

    # 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['subset']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    # 构建 prompts（每个 repeat 构建相同的 prompts）
    prompts = [build_prompt(template, row) for row in data]
    all_prompts = prompts * experiment_config["repeats"]

    # 批量结构化推理
    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    # 使用数据集的 metrics 函数计算
    all_predictions = []
    all_metrics = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        predictions = [r.answer for r in repeat_results]
        all_predictions.append(predictions)

        # 调用数据集的 metrics 函数
        metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    # 保存结果
    gold_answers = extract_gold_answers(data)
    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(gold_answers))


if __name__ == "__main__":
    main()
```

### 模板 2：LLM Judge

```python
"""MyDataset 评测脚本（LLM Judge 版）"""
import sys
from pathlib import Path

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner

from MyDataset.prompts import get_template, build_prompt
from MyDataset.metrics import compute_metrics


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("MyDataset/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    # 从 config 获取 schema
    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]

    # 获取 prompt 模板
    template = get_template(prompt_method)

    # 创建主模型客户端
    client = runner.create_llm_client(experiment_config["llm_config"])

    # 创建 judge 客户端
    judge_config = experiment_config.get("judge_config", {})
    if judge_config:
        judge_client = runner.create_llm_client(judge_config)
        judge_model = judge_config.get("model_name") or judge_config.get("model")
        print(f"Judge model: {judge_model}")
    else:
        judge_client = None
        print("Warning: No judge config found, using direct matching")

    # 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['subset']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    # 构建 prompts
    prompts = [build_prompt(template, row) for row in data]
    all_prompts = prompts * experiment_config["repeats"]

    # 批量结构化推理
    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    # 使用数据集的 metrics 函数计算
    all_predictions = []
    all_metrics = []

    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        predictions = [r.answer for r in repeat_results]
        all_predictions.append(predictions)

        # 调用数据集的 metrics 函数（传入 judge_client）
        if judge_client:
            metrics = compute_metrics(predictions, data, judge_client)
        else:
            metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    # 保存结果（包含 judge model）
    gold_answers = [row.get("answer", "") for row in data]
    metadata = {}
    if judge_model:
        metadata["judge_model"] = judge_model

    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
        metadata=metadata,
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(gold_answers))


if __name__ == "__main__":
    main()
```

## 步骤 6：注册数据集

编辑顶层 `run_all.py`，在 `DATASETS` 列表中添加：

```python
DATASETS = [
    "ToMBench",
    "MyDataset",  # 新增
]
```

## 步骤 7：测试

```bash
# 配置实验参数（编辑 experiment_config.yaml）
# 然后运行
python MyDataset/run.py

# 或运行所有数据集
python run_all.py
```

## 常见问题

### Q: 如何添加自定义 schema？

编辑 `src/metrics/schemas.py`，添加新的 Pydantic 模型：

```python
class CustomAnswer(BaseModel):
    """自定义答案 schema"""
    answer: Literal["OPTION_1", "OPTION_2", "OPTION_3"]
    confidence: float  # 可选字段
```

然后在数据集的 `config.yaml` 中引用：

```yaml
schema: CustomAnswer
```

### Q: 如何处理不同的数据格式？

`load_dataset` 返回的是数据列表，每个元素是一个字典。根据实际数据格式调整 `build_prompt` 和 `extract_gold_answers` 函数。

### Q: 如何支持多个 prompt 方法？

在 `PROMPTS` 字典中添加新方法，然后在 `config.yaml` 中修改 `default_prompt`。

### Q: 如何调试？

使用 `experiment_config.yaml` 中的 `max_samples` 参数限制样本数进行快速测试：

```yaml
max_samples: 3  # 只测试前 3 条样本
```
