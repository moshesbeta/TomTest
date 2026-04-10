# TomTest

简化评测框架 - 支持多数据集、多模型的 Theory-of-Mind 基准评测。

## 设计理念

基于**结构化输出**的新架构，简洁高效：
- **新增模型时**：只需修改 `experiment_config.yaml` 中的 API 配置
- **新增数据集时**：复用现有加载函数，只需编写自己的 `prompts.py` 和 `metrics.py`
- **不需要复杂的字符串检查**：直接从结构化输出获取答案

## 目录结构

```
TomTest/
├── datasets/              # 数据集（已规范化）
├── ToMBench/              # ToMBench 评测目录
│   ├── config.yaml         # 数据集配置（固定参数）
│   ├── prompts.py          # prompt 方法
│   ├── metrics.py          # metrics 计算（包含二级指标）
│   └── run.py            # ToMBench 主评测脚本
├── results/              # 评测结果输出
├── experiment_config.yaml  # 实验配置（LLM、repeat、路径等）
├── src/
│   ├── llm/               # LLMClient（支持 batch_generate_structure）
│   ├── dataloader/        # DataLoader
│   ├── metrics/
│   │   ├── schemas.py     # 通用的输出 schema
│   │   └── common.py     # 通用 metrics 计算函数
│   └── runner.py          # 评测运行器公共函数
├── docs/                 # 文档目录
└── run_all.py            # 统一运行所有数据集
```

## 核心优势：结构化输出

**之前**（无结构化输出）：
- 需要复杂的正则表达式提取
- 需要文本归一化
- 需要处理各种输出格式

**现在**（有结构化输出）：
- 直接从对象获取答案：`result.answer`
- 不需要字符串检查和提取
- 代码更简洁、更可靠

## 安装依赖

```bash
pip install openai datasets tqdm pyyaml
```

## 数据集下载

数据集托管于 [TomTraining/TomDatasets](https://huggingface.co/datasets/TomTraining/TomDatasets)，下载到本地 `datasets/` 目录：

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TomTraining/TomDatasets",
    repo_type="dataset",
    local_dir="datasets",
)
EOF
```

## 快速开始

### 1. 配置实验参数

编辑 `experiment_config.yaml`：

```yaml
# LLM 配置
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.6
  max_tokens: 8192
  max_workers: 16
  enable_thinking: False

# LLM Judge 配置（可选，用于需要 judge 的数据集）
judge:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.0
  max_tokens: 4096

# 实验参数
repeats: 3
max_samples: 0  # 0 表示使用全部样本

# 路径配置
datasets_path: datasets
results_path: results
```

### 2. 运行评测

```bash
# 运行所有数据集
python run_all.py

# 或单独运行某个数据集
python ToMBench/run.py
```

### 3. 查看结果

评测结果保存在 `results/` 目录：

| 文件 | 内容 |
|---|---|
| `{dataset}_{model}_{prompt_method}_{timestamp}.jsonl` | 详细预测结果（每行一个样本） |
| `{dataset}_{model}_{prompt_method}_{timestamp}.json` | 评测指标汇总 |

## 配置文件说明

### 数据集配置（各数据集目录下的 `config.yaml`）

存储数据集相关的**固定参数**：

```yaml
dataset: ToMBench
path: ToMBench/test
schema: MCQAnswer  # 从数据集自己的 schemas.py 导入
default_prompt: zero_shot
```

### 实验配置（`experiment_config.yaml`）

存储实验相关的**可变参数**：

| 参数 | 说明 |
|---|---|
| `llm.model_name` | 模型名称 |
| `llm.api_url` | API 地址 |
| `llm.api_key` | API 密钥（支持环境变量 `${VAR_NAME}`） |
| `llm.temperature` | 温度参数 |
| `llm.max_tokens` | 最大 token 数 |
| `llm.max_workers` | 最大线程数（默认 32） |
| `llm.enable_thinking` | 是否启用思考模式（默认 True） |
| `judge.model_name` | Judge 模型名称（可选） |
| `judge.api_url` | Judge API 地址（可选） |
| `judge.api_key` | Judge API 密钥（可选） |
| `judge.temperature` | Judge 温度（通常为 0.0） |
| `judge.max_tokens` | Judge 最大 token 数 |
| `repeats` | 重复运行次数 |
| `max_samples` | 最大样本数（0 = 全部） |
| `datasets_path` | 数据集根目录 |
| `results_path` | 结果输出目录 |

## 支持的数据集

| 数据集 | Schema | 二级指标 |
|---|---|---|
| ToMBench | `MCQAnswer` | 按 `Meta.ability` 分组 |

更多数据集请参考 [新增数据集指南](docs/add_new_dataset.md) 添加。

## 可用的 Schema

每个数据集在自己的 `schemas.py` 中通过 `SCHEMAS` 字典定义 schema。

```python
# ToMBench/schemas.py 示例
SCHEMAS = {
    "MCQAnswer": MCQAnswer,
    "JudgeAnswer": JudgeAnswer,  # 可选，供内部调用
}
```

| Schema | 说明 |
|---|---|
| `MCQAnswer` | 多选题答案（A/B/C/D） |
| `OpenAnswer` | 开放式答案（字符串） |
| `YesNoAnswer` | 是非题答案（YES/NO） |
| `MultipleChoice` | 多选题（任意数量选项） |
| `JudgeAnswer` | LLM Judge 答案（CORRECT/INCORRECT） |

## 扩展指南

- [新增数据集指南](docs/add_new_dataset.md) - 如何添加新的评测数据集
- [新增模型指南](docs/add_new_model.md) - 如何使用新模型进行评测

## 公共函数（src/runner.py）

`src/runner.py` 提供了数据集评测脚本之间的共享公共函数：

| 函数 | 说明 |
|---|---|
| `load_dataset_config()` | 加载数据集配置 |
| `load_experiment_config()` | 加载实验配置 |
| `create_llm_client()` | 创建 LLM 客户端 |
| `save_common_results()` | 保存评测结果（JSONL + JSON） |
| `print_summary_stats()` | 打印统计摘要 |
| `load_and_limit_data()` | 加载数据并限制样本数 |

### 使用示例

```python
from src import runner

# 加载配置
dataset_config = runner.load_dataset_config("ToMBench/config.yaml")
experiment_config = runner.load_experiment_config("experiment_config.yaml")

# 创建客户端
client = runner.create_llm_client(experiment_config["llm_config"])

# 加载数据
data = runner.load_and_limit_data(
    subset=dataset_config["subset"],
    datasets_path=experiment_config["datasets_path"],
    max_samples=experiment_config["max_samples"],
)

# 使用数据集的 metrics 函数
from ToMBench.metrics import compute_metrics
metrics = compute_metrics(predictions, data)

# 保存结果
runner.save_common_results(
    dataset_name="ToMBench",
    model=experiment_config["llm_config"]["model_name"],
    prompt_method=prompt_method,
    all_predictions=all_predictions,
    gold_answers=gold_answers,
    all_metrics=all_metrics,
    results_path=experiment_config["results_path"],
)
```

## 许可证

MIT License
