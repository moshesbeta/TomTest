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
├── tasks/                 # 数据集评测代码
│   ├── ToMi/
│   │   ├── config.yaml
│   │   ├── prompts.py
│   │   ├── metrics.py
│   │   ├── schemas.py
│   │   └── run.py
│   ├── ToMBench/
│   ├── Tomato/
│   └── ToMQA/
├── results/              # 评测结果输出
├── tables/               # 评测结果表格
│   ├── SUMMARY.md         # 总览表格（accuracy）
│   └── {dataset_name}/    # 各数据集详细表格
│       ├── 基础指标.md
│       ├── 其他指标.md
│       └── {model}/
│           └── config.json
├── experiment_config.yaml  # 实验配置（LLM、repeat、路径等）
├── src/
│   ├── llm/               # LLMClient（支持 batch_generate_structure）
│   ├── dataloader/        # DataLoader
│   ├── metrics/
│   │   └── common.py     # 通用 metrics 计算函数
│   └── runner.py          # 评测运行器公共函数
├── docs/                 # 文档目录
├── run_all.py            # 统一运行所有数据集
├── generate_dataset_tables.py  # 从 results 生成各数据集表格
└── generate_summary.py         # 从表格生成总览汇总
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
  max_workers: 64
  enable_thinking: true

# LLM Judge 配置（可选，用于需要 judge 的数据集）
judge:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.0
  max_tokens: 4096

# 实验参数
repeats: 3
max_samples: 0  # 0 表示使用全部样本，>0 则随机抽样
seed: 42  # 随机种子（用于可复现的随机抽样）

# 路径配置
datasets_path: datasets
results_path: results
```

### 2. 运行评测

```bash
# 运行所有数据集
python run_all.py

# 或单独运行某个数据集
python tasks/ToMBench/run.py
```

### 3. 生成结果表格

```bash
# 从 results 生成各数据集表格（基础指标.md、其他指标.md）
python generate_dataset_tables.py

# 从表格生成总览汇总（SUMMARY.md）
python generate_summary.py

# 或直接输出到终端
python generate_summary.py --stdout
```

### 4. 查看结果

评测结果保存在 `results/` 目录，表格保存在 `tables/` 目录：

**results/ 结构**：
```
results/
├── {dataset_name}/
│   └── {model}/
│       ├── config.json         # 配置信息（包含 dataset_config 和 experiment_config）
│       ├── metrics.json        # 评测指标（avg_metrics + all_metrics）
│       └── prediction.jsonl    # 详细预测结果（每行一个样本）
```

**tables/ 结构**：
```
tables/
├── SUMMARY.md              # 总览表格（所有数据集 × 模型 × accuracy）
└── {dataset_name}/         # 各数据集详细表格
    ├── 基础指标.md          # accuracy、correct、total
    ├── 其他指标.md           # 其他所有指标
    └── {model}/
        └── config.json       # 复制的配置文件
```

| 文件 | 内容 |
|---|---|
| `config.json` | 所有配置信息（除 api_key 和 api_url 外） |
| `metrics.json` | 评测指标（平均指标 + 各运行详细指标） |
| `prediction.jsonl` | 预测结果（每行包含 repeat、sample_idx、prediction、gold_answer） |
| `SUMMARY.md` | 总览表格（所有数据集 × 模型 × accuracy） |
| `基础指标.md` | 数据集详细指标（accuracy、correct、total） |
| `其他指标.md` | 数据集详细指标（其他所有指标） |

## 配置文件说明

### 数据集配置（各数据集目录下的 `config.yaml`）

存储数据集相关的**固定参数**：

```yaml
dataset: ToMBench
path: ToMBench/test  # 数据集子集路径（相对于 datasets/）
schema: MCQAnswer   # 从数据集自己的 schemas.py 导入
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
| `max_samples` | 最大样本数（0 = 全部，>0 = 随机抽样） |
| `seed` | 随机种子（用于可复现的随机抽样） |
| `datasets_path` | 数据集根目录 |
| `results_path` | 结果输出目录 |

## 支持的数据集

| 数据集 | Schema | 说明 |
|---|---|---|
| ToMBench | `MCQAnswer` | Theory of Mind 基准测试 |
| Tomato | `MCQAnswer` | Theory of Mind 多选题（支持选项 shuffle） |
| ToMQA | `OpenAnswer` | Theory of Mind 问答 |
| ToMi | `OneWordAnswer` | Theory of Mind 单词回答 |

更多数据集请参考 [新增数据集指南](docs/add_new_dataset.md) 添加。

## 可用的 Schema

每个数据集在自己的 `schemas.py` 中通过 `SCHEMAS` 字典定义 schema。

```python
# tasks/ToMBench/schemas.py 示例
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
| `OneWordAnswer` | 单词回答（如 ToMi） |
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
| `save_common_results()` | 保存评测结果（config.json + metrics.json + prediction.jsonl） |
| `print_summary_stats()` | 打印统计摘要 |
| `load_and_limit_data()` | 加载数据并限制样本数（支持随机抽样） |

### 使用示例

```python
from src import runner

# 加载配置
dataset_config = runner.load_dataset_config("tasks/ToMBench/config.yaml")
experiment_config = runner.load_experiment_config("experiment_config.yaml")

# 创建客户端
client = runner.create_llm_client(experiment_config["llm_config"])

# 加载数据（支持随机抽样）
data = runner.load_and_limit_data(
    subset=dataset_config["subset"],
    datasets_path=experiment_config["datasets_path"],
    max_samples=experiment_config["max_samples"],
    seed=experiment_config.get("seed", 42),
)

# 使用数据集的 metrics 函数
from tasks.ToMBench.metrics import compute_metrics
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
    dataset_config=dataset_config,      # 可选，保存完整配置到 config.json
    experiment_config=experiment_config,  # 可选，保存完整配置到 config.json
)

# 返回值: (config_path, metrics_path, prediction_path)
```

### save_common_results() 详细说明

```python
def save_common_results(
    dataset_name: str,
    model: str,
    prompt_method: str,
    all_predictions: List[List[str]],
    gold_answers,
    all_metrics: List[Dict[str, Any]],
    results_path: str = "results",
    metadata: Optional[Dict[str, Any]] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path, Path]:
    """保存评测结果

    结果保存结构: results/{dataset_name}/{model}/
    - config.json: 包含所有配置（dataset_config + experiment_config，排除 api_key 和 api_url）
    - metrics.json: 包含 avg_metrics 和 all_metrics
    - prediction.jsonl: 包含每条样本的预测结果

    返回: (config_path, metrics_path, prediction_path)
    """
```

### load_and_limit_data() 详细说明

```python
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
        seed: 随机种子（用于可复现的随机抽样）

    Returns:
        数据列表
    """
```

当 `max_samples > 0` 时，会随机抽取指定数量的样本，使用 `seed` 保证结果可复现。

## 结果生成工具

### generate_dataset_tables.py

从 `results/` 目录读取 `metrics.json` 文件，为每个数据集生成详细表格：

```bash
python generate_dataset_tables.py --results-dir results --output-dir tables
```

输出：
- `tables/{dataset}/基础指标.md` - accuracy、correct、total
- `tables/{dataset}/其他指标.md` - 其他所有指标
- `tables/{dataset}/{model}/config.json` - 复制的配置文件

### generate_summary.py

从 `tables/` 目录读取已生成的表格文件，生成总览汇总：

```bash
python generate_summary.py --tables-dir tables
# 或直接输出到终端
python generate_summary.py --stdout
```

输出：
- `tables/SUMMARY.md` - 所有数据集 × 模型 × accuracy

## 许可证

MIT License
