# TomTest 报告工具说明

本目录包含三个独立的报告/分析脚本，均采用 **YAML 配置文件驱动**，覆盖从原始评测结果到深度 bad case 分析的完整工作流。

---

## 文件一览

| 文件 | 功能 | 配置文件 |
|---|---|---|
| `generate_dataset_tables.py` | 生成各数据集的多维度指标 Markdown 表格 | `tables_config.yaml` |
| `generate_summary.py` | 汇总所有数据集的 accuracy，生成总览表格 | 无（直接读 `tables/`） |
| `report_client.py` | 指标对比 + 优先级 bad case 抽取 + LLM 批量分析 | `report_config.yaml` |

---

## 工作流

```
results/                         # 评测原始输出
   └── {dataset}/{model}/exp_*/
       ├── prediction.jsonl
       └── metrics.json
        │
        ▼  generate_dataset_tables.py
tables/                          # 聚合 Markdown 表格
   └── {dataset}/
       ├── 基础指标.md
       └── 其他指标.md
        │
        ├──▶ generate_summary.py ──▶ tables/SUMMARY.md
        │
        └──▶ report_client.py   ──▶ analysis/{dataset}/{model}/*.md
```

---

## generate_dataset_tables.py — 生成指标表格

从 `results/` 目录读取各实验的 `metrics.json`，为每个数据集生成：

- **`基础指标.md`**：accuracy / correct / total
- **`其他指标.md`**：按 `## section` 分块，含标量指标和各维度子指标（by_ability、by_dimension 等）

支持增量合并：若表格已存在，新模型列会追加进去，不会覆盖已有数据（会提示确认）。

**用法**

```bash
python generate_dataset_tables.py tables_config.yaml
```

**tables_config.yaml 关键字段**

```yaml
results_dir: results       # 原始结果目录
output_dir: tables         # 输出目录
exp_suffix:                # 指定实验后缀（null = 自动取最新）
dataset:                   # 只处理某个数据集（null = 全部）
models:
  - name: Qwen3-8B         # results/ 中的目录名
    display: Qwen3-8B-Think  # 表格列名（可选，默认同 name）
```

---

## generate_summary.py — 生成总览汇总

读取 `tables/` 下所有数据集的 `基础指标.md`，提取每个模型的 **accuracy**，生成：

- **`tables/SUMMARY.md`**：横轴为数据集、纵轴为模型的总览对比表格

无需额外配置，直接扫描 `tables/` 目录。

**用法**

```bash
python generate_summary.py
```

---

## report_client.py — Bad Case 深度分析

综合三项功能于一体：

1. **多维度指标展示**：从 `tables/` 读取基础指标和细粒度指标，可与基线模型对比并展示差值
2. **优先级 bad case 抽取**：按维度表现排序，分三层（Tier）优先抽取最差维度的典型错误样本
3. **LLM 批量分析**：调用本地或远端 LLM API，对每条 bad case 输出维度归因、错误原因、改善建议

**用法**

```bash
# 仅查看指标和 bad case，跳过 LLM 分析
python report_client.py report_config.yaml   # no_llm_analysis: true

# 完整功能（需 LLM API 可用）
python report_client.py report_config.yaml   # no_llm_analysis: false
```

**report_config.yaml 关键字段**

```yaml
results_dir: results
tables_dir: tables
output_dir: analysis          # 报告保存目录

model:
  name: Qwen3-8B              # results/ 中的目录名
  display: Qwen3-8B-Think     # 表格列名和显示名

baseline:                     # 可选基线（同样支持字符串或字典写法）
  name: Qwen3-7B
  display: Qwen3-7B-Think

dataset: ToMBench             # null = 分析全部数据集

llm:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.0
  max_workers: 8

bad_cases:
  n: 10                       # 抽取条数
  seed: 42

no_llm_analysis: false        # true 时跳过 LLM 分析
output_report: true           # 是否保存 Markdown 报告到 output_dir
```

**Bad Case 抽样分层规则**

| Tier | 条件 |
|---|---|
| Tier 1 | 维度 wrong_rate > 70% **且** 该样本全部 repeat 均答错 |
| Tier 2 | 维度 wrong_rate > 50% **且** 答错次数 ≥ 总 repeat 的 50% |
| Tier 3 | 其余有错的样本，按答错次数降序 |

若提供了基线，维度排序依据为 `模型精度 - 基线精度`（差值越负越优先）；否则按 wrong_rate 降序。

**LLM 分析输出格式**

```
【维度归因】该题核心考查哪种 ToM 能力
【错误原因】模型出现了哪种认知偏差或推理错误
【改善建议】可通过哪种方式提升
```

报告文件保存到 `{output_dir}/{dataset}/{model_display}/{timestamp}.md`。
