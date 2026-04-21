# 新增数据集指南

本指南介绍如何在 TomTest 框架中添加一个新的数据集评测任务。

## 目录

- [概述](#概述)
- [步骤 1：准备数据集文件](#步骤-1准备数据集文件)
- [步骤 2：创建任务目录和四个核心文件](#步骤-2创建任务目录和四个核心文件)
  - [2.1 config.yaml](#21-configyaml)
  - [2.2 prompts.py](#22-promptspy)
  - [2.3 metrics.py](#23-metricspy)
  - [2.4 run.py](#24-runpy)
- [步骤 3：注册到 run_all.py](#步骤-3注册到-run_allpy)
- [步骤 4：验证新数据集](#步骤-4验证新数据集)
- [Schema 参考](#schema-参考)
- [常见问题](#常见问题)

---

## 概述

添加一个新数据集需要完成以下工作：

| 步骤 | 内容 | 文件 |
|---|---|---|
| 1 | 准备 Arrow 格式的数据集文件 | `datasets/{DatasetName}/` |
| 2 | 创建任务目录和 4 个核心文件 | `tasks/{DatasetName}/` |
| 3 | 注册到全量运行脚本 | `run_all.py` |
| 4 | 快速验证 | `tasks/{DatasetName}/run.py` |

---

## 步骤 1：准备数据集文件

TomTest 使用 HuggingFace Arrow 格式（`.arrow`）的数据集文件。

### 1.1 获取数据集

```bash
# 方式一：从 HuggingFace Hub 下载（需先设置代理）
export https_proxy=http://agent.baidu.com:8891
python -c "
from datasets import load_dataset
ds = load_dataset('your-org/your-dataset', split='test')
ds.save_to_disk('datasets/MyDataset/test')
"

# 方式二：从本地文件转换
python -c "
from datasets import Dataset
import json

data = [json.loads(l) for l in open('my_data.jsonl')]
ds = Dataset.from_list(data)
ds.save_to_disk('datasets/MyDataset/test')
"
```

### 1.2 验证数据集

```bash
# 在 TomTest 目录下验证数据集可以正常加载
python -c "
from src.dataloader.dataloader import load_dataset
data = load_dataset('MyDataset/test', 'datasets')
print(f'加载了 {len(data)} 条样本')
print('第一条示例:', data[0])
"
```

### 1.3 数据集目录结构

数据集保存后的目录结构类似：

```
datasets/
└── MyDataset/
    └── test/
        ├── dataset_info.json
        ├── dataset_dict.json (如有多个split)
        └── data-00000-of-00001.arrow
```

> **提示**：参考 `datasets_examples/` 目录下的示例 JSON 文件，了解各数据集的字段格式。

---

## 步骤 2：创建任务目录和四个核心文件

```bash
cd /path/to/TomTest/tasks
mkdir MyDataset
```

### 2.1 config.yaml

`config.yaml` 定义数据集的基本配置：

```yaml
# 数据集名称（用于结果保存路径）
dataset: MyDataset

# 数据集加载路径，对应 datasets/ 下的子目录
# 格式：{DatasetName}/{split}
path: MyDataset/test

# Prompt 方法（在 prompts.py 的 PROMPTS 字典中定义）
method: ZS_vanilla

# 结构化输出 Schema（从 src/schemas.py 中选择）
schema: MCQAnswer

# 系统 Prompt（可选，不填则继承 experiment_config.yaml 中的配置）
# system_prompt: "You are an expert in theory of mind."
```

**Schema 选择**（详见 [Schema 参考](#schema-参考)）：

| 题型 | 选用 Schema |
|---|---|
| 四选一（A/B/C/D） | `MCQAnswer` |
| 三选一（A/B/C） | `MCQAnswer3` |
| 三选一小写（a/b/c） | `MCQAnswer3Lower` |
| 开放式问答 | `OpenAnswer` |
| 多标签多选 | `MultiLabelAnswer` |

### 2.2 prompts.py

`prompts.py` 定义如何将原始数据行格式化为模型输入的 Prompt：

```python
"""MyDataset prompts"""
from typing import Any, Dict

# Prompt 模板字典，key 为方法名，与 config.yaml 中的 method 对应
PROMPTS = {
    "ZS_vanilla": """\
Read the following story and answer the question.

Story:
{story}

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Answer with exactly one letter (A/B/C/D):""",

    "ZS_cot": """\
Read the following story carefully and answer the question step by step.

Story:
{story}

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Think step by step, then answer with exactly one letter (A/B/C/D):""",
}


def build_prompt(row: Dict[str, Any], method: str = "ZS_vanilla") -> str:
    """构建 Prompt

    Args:
        row: 数据集中的一行，字段与数据集格式对应
        method: Prompt 方法名，对应 PROMPTS 中的 key

    Returns:
        格式化后的 Prompt 字符串
    """
    template = PROMPTS.get(method, PROMPTS["ZS_vanilla"])

    # 根据实际数据集字段提取内容
    story = row.get("story", row.get("context", ""))
    question = row.get("question", "")
    options = row.get("options", [])

    # 确保选项数量足够
    while len(options) < 4:
        options.append("")

    return template.format(
        story=story,
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )
```

> **注意**：`build_prompt` 函数签名必须是 `(row, method)` 形式，`method` 默认值应与 `config.yaml` 中的 `method` 一致。

### 2.3 metrics.py

`metrics.py` 定义如何计算评测指标。根据题型选择合适的模板。

#### 模板 A：多选题直接匹配（最常见）

```python
"""MyDataset metrics 计算"""
from typing import Any, Dict, List, Optional

from src.utils import compute_sample_metrics


def compute_metrics(
    predictions: List[Any],
    data: List[Dict[str, Any]],
    judge_client=None,
) -> Dict[str, Any]:
    """计算 MyDataset 的指标

    Args:
        predictions: 模型预测答案列表（从结构化输出提取的 answer 字段值）
        data: 原始数据列表（与 predictions 等长）
        judge_client: LLM Judge 客户端（可选，不使用时传 None）

    Returns:
        包含 accuracy / correct / total 及二级指标的字典
        必须包含 "per_sample_results" 键，供 runner.py 保存 prediction.jsonl
    """
    # 提取 gold 答案
    gold_answers = [row.get("answer", "") for row in data]

    # 计算样本级指标（正确/错误分类）
    sample_result = compute_sample_metrics(
        predictions=predictions,
        gold_answers=gold_answers,
        is_correct_fn=lambda p, g: str(p).strip() == str(g).strip(),
    )

    # ---- 二级指标：按能力类别细分（可选但推荐）----
    category_metrics: Dict[str, Dict[str, int]] = {}
    for row, is_correct_info in zip(data, sample_result["per_sample_results"]):
        category = row.get("ability", row.get("category", "unknown"))
        if category not in category_metrics:
            category_metrics[category] = {"correct": 0, "total": 0}
        category_metrics[category]["total"] += 1
        if is_correct_info["is_correct"]:
            category_metrics[category]["correct"] += 1

    by_category = {
        cat: stats["correct"] / stats["total"]
        for cat, stats in category_metrics.items()
        if stats["total"] > 0
    }

    return {
        # 基础指标（必须包含这三项）
        "accuracy": sample_result["correct"] / sample_result["total"],
        "correct": sample_result["correct"],
        "total": sample_result["total"],
        # 二级指标（字典类型会在表格中单独展示为一个 section）
        "by_category": by_category,
        # 样本级结果（必须包含，供 runner.py 使用）
        "per_sample_results": sample_result["per_sample_results"],
    }
```

#### 模板 B：开放式问答 + LLM Judge

```python
"""MyDataset metrics 计算（LLM Judge 版）"""
from typing import Any, Dict, List, Optional

from src.schemas import JudgeAnswer
from src.utils import compute_sample_metrics_with_llm


def compute_metrics(
    predictions: List[Any],
    data: List[Dict[str, Any]],
    judge_client,
) -> Dict[str, Any]:
    """使用 LLM Judge 计算开放式问答指标"""
    gold_answers = [row.get("gold_answer", row.get("answer", "")) for row in data]

    # 构建 Judge Prompt 列表
    judge_prompts = []
    for pred, row in zip(predictions, data):
        gold = row.get("gold_answer", "")
        context = row.get("story", row.get("context", ""))
        question = row.get("question", "")
        judge_prompts.append(
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Ground Truth: {gold}\nModel Answer: {pred}\n\n"
            f"Is the model answer correct? Output True or False:"
        )

    # 批量调用 Judge（返回 List[LLMResponse]）
    judge_results = judge_client.batch_generate_structure(judge_prompts, JudgeAnswer)

    # 构建 per_sample_results
    per_sample = []
    correct = 0
    for result in judge_results:
        is_correct = result.content is not None and result.content.answer == "True"
        if is_correct:
            correct += 1
        per_sample.append({
            "is_correct": is_correct,
            "error_reason": None if is_correct else "wrong_answer",
        })

    return {
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "correct": correct,
        "total": len(predictions),
        "per_sample_results": per_sample,
    }
```

### 2.4 run.py

`run.py` 是数据集评测的主入口，负责加载数据、调用 LLM、计算并保存结果：

```python
"""MyDataset 评测脚本"""
import sys
from pathlib import Path

# 将 TomTest 根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import runner
from tasks.MyDataset.metrics import compute_metrics
from tasks.MyDataset.prompts import build_prompt


def main():
    # 1. 加载配置
    dataset_config = runner.load_dataset_config(
        Path(__file__).parent / "config.yaml"
    )
    experiment_config = runner.load_experiment_config(
        Path(__file__).parent.parent.parent / "experiment_config.yaml"
    )

    # 2. 创建 LLM 客户端
    schema = runner.load_schema(dataset_config["schema"])
    client = runner.create_llm_client(experiment_config["llm"], dataset_config)
    judge_client = runner.create_judge_client(experiment_config.get("judge", {}))

    # 3. 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )
    print(f"已加载 {len(data)} 条样本，数据集: {dataset_config['path']}")

    # 4. 构建 Prompts
    method = dataset_config["method"]
    prompts = [build_prompt(row, method) for row in data]

    # 5. 多轮推理（repeats 次）
    n = len(data)
    repeats = experiment_config["repeats"]
    all_prompts_flat = prompts * repeats
    print(f"开始推理，共 {len(all_prompts_flat)} 个请求（{n} 样本 × {repeats} 轮）...")
    all_results_flat = client.batch_generate_structure(all_prompts_flat, schema)

    # 6. 按轮次计算指标
    all_metrics = []
    all_results_by_repeat = []
    all_prompts_by_repeat = []

    for i in range(repeats):
        repeat_results = all_results_flat[i * n : (i + 1) * n]
        repeat_prompts = all_prompts_flat[i * n : (i + 1) * n]
        predictions = [
            r.content.answer if r.content else None for r in repeat_results
        ]
        metrics = compute_metrics(predictions, data, judge_client)
        all_metrics.append(metrics)
        all_results_by_repeat.append(repeat_results)
        all_prompts_by_repeat.append(repeat_prompts)
        print(
            f"第 {i+1} 轮: accuracy={metrics['accuracy']:.4f}  "
            f"({metrics['correct']}/{metrics['total']})"
        )

    # 7. 保存结果
    gold_answers = [row.get("answer", "") for row in data]
    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=all_results_by_repeat,
        all_prompts=all_prompts_by_repeat,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
    )
    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
```

---

## 步骤 3：注册到 run_all.py

编辑 TomTest 根目录下的 `run_all.py`，在 `DATASETS` 列表中添加新数据集名称：

```python
DATASETS = [
    "Belief_R",
    "FictionalQA",
    "HellaSwag",
    "MyDataset",      # ← 新增
    "RecToM",
    "SimpleTom",
    "SocialIQA",
    "Tomato",
    "ToMBench",
    "ToMChallenges",
    "ToMQA",
    # "FollowBench",  # 已禁用（指标问题）
    # "ToMi",         # 已禁用（异常分数）
]
```

---

## 步骤 4：验证新数据集

### 4.1 快速冒烟测试（3 条样本）

编辑 `experiment_config.yaml`，临时设置：

```yaml
max_samples: 3   # 只取前 3 条样本
repeats: 1       # 只跑 1 轮
```

然后运行：

```bash
cd /path/to/TomTest
python tasks/MyDataset/run.py
```

### 4.2 检查输出

成功后会在 `results/MyDataset/{model_name}/exp_{timestamp}/` 下生成三个文件：

```
results/
└── MyDataset/
    └── Qwen3-8B/
        └── exp_20260422_120000/
            ├── config.json       # 完整实验配置（API key 已脱敏）
            ├── metrics.json      # 指标结果
            └── prediction.jsonl  # 每条样本的预测详情
```

验证 `metrics.json` 格式正确：

```bash
cat results/MyDataset/Qwen3-8B/exp_*/metrics.json | python -m json.tool
```

应包含 `avg_metrics.accuracy`、`avg_metrics.correct`、`avg_metrics.total` 等字段。

### 4.3 全量运行

恢复 `experiment_config.yaml` 的全量配置后运行：

```bash
python tasks/MyDataset/run.py   # 单数据集
# 或
python run_all.py                # 全部数据集
```

---

## Schema 参考

所有 Schema 定义在 `src/schemas.py`：

| Schema 名 | 字段 | 适用场景 |
|---|---|---|
| `MCQAnswer` | `answer: Literal["A","B","C","D"]` | 标准四选一 MCQ |
| `MCQAnswer3` | `answer: Literal["A","B","C"]` | 三选一 MCQ（大写） |
| `MCQAnswer3Lower` | `answer: Literal["a","b","c"]` | 三选一 MCQ（小写） |
| `OpenAnswer` | `answer: str` | 开放式问答 |
| `OneWordAnswer` | `answer: str`（无空白字符） | 单词/短语回答 |
| `MultiLabelAnswer` | `answer: List[str]` | 多标签多选 |
| `JudgeAnswer` | `answer: Literal["True","False"]` | LLM Judge 判断 |

### 自定义 Schema

如果现有 Schema 不满足需求，在 `src/schemas.py` 末尾添加新类：

```python
from pydantic import BaseModel
from typing import Literal

class BinaryAnswer(BaseModel):
    """是/否二元判断"""
    answer: Literal["Yes", "No"]

class RankedAnswer(BaseModel):
    """排名答案（1-5）"""
    answer: Literal["1", "2", "3", "4", "5"]
    confidence: str  # 可选附加字段
```

然后在 `config.yaml` 中引用：

```yaml
schema: BinaryAnswer
```

---

## 常见问题

### Q: 如何查看数据集字段结构？

```bash
python -c "
from src.dataloader.dataloader import load_dataset
data = load_dataset('MyDataset/test', 'datasets')
import json
print(json.dumps(data[0], ensure_ascii=False, indent=2))
"
```

或查看 `datasets_examples/` 目录下对应的示例文件。

### Q: 如何处理多语言数据集？

参考 `tasks/ToMBench/prompts.py`，根据数据行中的语言标记（如 `row.get("lang")`）选择不同语言的 Prompt 模板。

### Q: 如何支持选项随机打乱（减少位置偏差）？

参考 `tasks/Tomato/run.py` 或 `tasks/SocialIQA/run.py`，在构建 Prompt 前对选项顺序按 `(sample_idx, repeat)` 种子随机打乱，并同步更新 gold 答案的选项字母。

### Q: `compute_sample_metrics` 返回什么结构？

```python
{
    "correct": int,           # 正确数
    "total": int,             # 总数
    "per_sample_results": [   # 每条样本
        {
            "is_correct": bool,
            "error_reason": str | None,  # "content_none" 或 "wrong_answer"
        },
        ...
    ]
}
```

### Q: 如何禁用某个数据集？

在 `run_all.py` 的 `DATASETS` 列表中注释掉该数据集，并加上原因说明：

```python
# "MyDataset",  # 暂时禁用（原因说明）
```
