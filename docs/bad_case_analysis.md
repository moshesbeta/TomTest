# Bad Case 分析指南

本指南介绍如何使用 `report/report_client.py` 对模型的错误案例进行系统性分析，并生成包含 LLM 自动诊断的分析报告。

## 目录

- [概述](#概述)
- [前提条件](#前提条件)
- [快速开始](#快速开始)
- [配置文件 report_config.yaml](#配置文件-report_configyaml)
- [输出内容说明](#输出内容说明)
- [Bad Case 优先级分层机制](#bad-case-优先级分层机制)
- [分析报告结构](#分析报告结构)
- [常见用法示例](#常见用法示例)
- [常见问题](#常见问题)

---

## 概述

`report_client.py` 是一个全自动的 Bad Case 分析工具，它会：

1. **读取指标**：从 `tables/` 加载基础指标和细粒度指标，与基线模型对比
2. **分层抽取**：按优先级（最差维度优先）抽取代表性错误案例
3. **LLM 诊断**：调用 LLM 对每个 Bad Case 进行维度归因、错误原因分析和改善建议
4. **生成报告**：输出终端摘要 + Markdown 报告文件

---

## 前提条件

运行分析前，需确保以下文件已就绪：

| 所需文件 | 来源 |
|---|---|
| `results/{dataset}/{model}/exp_*/prediction.jsonl` | 运行评测后自动生成 |
| `tables/{dataset}/基础指标.md` | 运行 `generate_dataset_tables.py` 后生成 |
| `tables/{dataset}/其他指标.md` | 运行 `generate_dataset_tables.py` 后生成 |

如果这些文件尚未生成，请先参考 [add_new_model.md](add_new_model.md) 和 [generate_tables.md](generate_tables.md)。

---

## 快速开始

```bash
cd /path/to/TomTest

# 使用默认配置（report/report_config.yaml）
python report/report_client.py

# 使用自定义配置文件
python report/report_client.py /path/to/my_report_config.yaml
```

---

## 配置文件 report_config.yaml

所有参数通过 `report/report_config.yaml` 配置：

```yaml
# ============================
# 输入路径
# ============================
results_dir: results    # prediction.jsonl 所在的 results 目录
tables_dir: tables      # 指标表格所在目录

# ============================
# 输出路径
# ============================
output_dir: analysis    # Markdown 报告保存目录

# ============================
# 待分析模型
# ============================
# 写法1：字符串（results/ 目录名 == 表格列名）
# model: Qwen3-8B

# 写法2：字典（目录名与显示名不同时使用）
model:
  name: Qwen3-8B        # results/ 中的目录名（用于读取 prediction.jsonl）
  display: Qwen3-8B     # tables/ 中的列名（用于读取指标表格）

# ============================
# 可选：基线模型（用于对比）
# ============================
# baseline: Qwen3-4B
baseline:
  name: Qwen3-4B
  display: Qwen3-4B

# ============================
# 数据集
# ============================
# 指定单个数据集（推荐，聚焦分析）
dataset: ToMBench
# 不填或注释掉则分析所有有 results 的数据集
# dataset:

# ============================
# LLM 分析配置（用于自动诊断 Bad Case）
# ============================
llm:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.0        # 分析任务使用确定性输出
  max_tokens: 4096
  max_workers: 8          # Bad Case 数量通常较少，无需过高并发
  enable_thinking: false
  system_prompt: ""

# ============================
# Bad Case 抽取配置
# ============================
bad_cases:
  n: 10       # 最多抽取多少条（按优先级选取最有代表性的）
  seed: 42    # 随机种子（固定可复现）

# ============================
# 功能开关
# ============================
no_llm_analysis: false   # true 时跳过 LLM 分析（仅打印指标和 Bad Case 列表）
output_report: true      # true 时保存 Markdown 报告到 output_dir
```

---

## 输出内容说明

### 终端输出

运行后会在终端依次打印：

```
============================================================
  TomTest 评测报告
  数据集: ToMBench  模型: Qwen3-8B  基线: Qwen3-4B
============================================================

[1/3] 基础指标
------------------------------------------------------------
指标                 Qwen3-8B       Qwen3-4B       差值(↑为好)
accuracy             0.7340         0.6982         +0.0358
correct              743            707            +0.0358
total                1012           1012           N/A

[2/3] 细粒度指标
------------------------------------------------------------
-- by_ability --
  Belief: Content false beliefs          0.7143   0.6521   +0.0622
  Desire: Goal-based predictions         0.6891   0.6234   +0.0657
  Knowledge: Percepts-knowledge links    0.8021   0.7654   +0.0367
  ...

[3/3] Bad Case 分析（共 10 条，按维度表现排序）

[Bad Case 1/10]  [Tier 1 - 最差维度，全错]
错误 repeat: 3/3
维度: Desire: Goal-based predictions
Meta: {"ability": "Desire: Goal-based predictions", "order": "first_order"}
正确答案: B  |  模型回答: C
Prompt（节选）: Sally has a basket...
推理过程（节选）: Let me think about what Sally wants...

[LLM 分析]
【维度归因】该题考查一阶欲望推断（First-Order Desire），要求模型预测 Sally 的行为目标。
【错误原因】模型在推断角色欲望时受到了显著的信念偏差干扰，误将场景中的物品位置信息
            映射为欲望预测依据，产生了"位置-欲望"混淆的认知错误。
【改善建议】可在 Prompt 中显式引导模型区分"人物知道什么"和"人物想要什么"两种不同的
            心智状态，或增加欲望推断专项训练数据。
------------------------------------------------------------
```

### Markdown 报告文件

如果 `output_report: true`，报告保存至：

```
analysis/{DatasetName}/{model_display}/{timestamp}.md
```

例如：

```
analysis/ToMBench/Qwen3-8B/20260422_145230.md
```

报告包含完整的指标表格、细粒度对比表和所有 Bad Case 的详细分析（含完整 Prompt 和推理过程）。

---

## Bad Case 优先级分层机制

系统按以下三层优先级抽取 Bad Case：

| 优先级 | 条件 | 说明 |
|---|---|---|
| **Tier 1**（最高） | 该维度错误率 > 70% **且** 该样本 `repeats` 次全部答错 | 最稳定、最严重的错误 |
| **Tier 2** | 该维度错误率 > 50% **且** 该样本至少一半次数答错 | 较显著的系统性错误 |
| **Tier 3** | 其余有错的样本 | 偶发性错误 |

在同一 Tier 内，系统按**与基线模型的性能差距**（差距越大越优先）排序，没有基线时按维度错误率降序排列。

这样设计的目的是：优先展示最具代表性、最值得关注的错误模式，避免浪费分析精力在随机偶发错误上。

---

## 分析报告结构

LLM 对每个 Bad Case 的分析按固定格式输出：

```
【维度归因】<该题核心考查的 ToM 能力>
【错误原因】<模型为什么会答错，出现了哪种认知偏差>
【改善建议】<提示词工程、思维链引导、数据增强等改进方向>
```

分析依据：
- 样本的 **Meta 信息**（如 `ability`、`order`、`language` 等标签）
- **Prompt 原文**（前 800 字符）
- 模型的**推理过程**（前 600 字符，来自 `prediction.jsonl` 的 `pred.reasoning`）
- **正确答案 vs 模型回答**

---

## 常见用法示例

### 只看指标，不做 LLM 分析（快速浏览）

```yaml
no_llm_analysis: true
output_report: false
```

```bash
python report/report_client.py
```

### 分析所有数据集

```yaml
# dataset: 注释掉或留空
dataset:
```

### 抽取更多 Bad Case

```yaml
bad_cases:
  n: 20    # 抽取 20 条
  seed: 0  # 更换种子
```

### 使用外部 LLM（如 DeepSeek）做分析

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.0
  max_tokens: 2048
  max_workers: 4
```

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python report/report_client.py
```

### 不与基线对比

```yaml
# baseline: 注释掉或留空
# baseline:
```

---

## 常见问题

### Q: prediction.jsonl 在哪里？

在 `results/{DatasetName}/{model_name}/exp_{timestamp}/prediction.jsonl`。脚本会自动找到该模型最新的 `exp_*` 目录。

### Q: 分析报告保存在哪里？

`analysis/{DatasetName}/{model_display}/{timestamp}.md`。如果 `output_dir` 使用相对路径，则相对于 TomTest 根目录。

### Q: 可以分析历史版本的实验结果吗？

可以。`report_config.yaml` 中的 `results_dir` 和 `tables_dir` 支持自定义路径。如果同一模型有多次实验，系统默认取最新的 `exp_*` 目录。目前不支持直接指定特定 `exp_suffix`，如需分析历史实验，可临时将对应 `exp_*` 目录重命名为最新时间戳。

### Q: 没有启用 LLM 时，Bad Case 列表还有用吗？

有用。即使 `no_llm_analysis: true`，仍会打印：
- 完整的指标对比表（基础指标 + 细粒度指标）
- 每条 Bad Case 的：维度标签、错误次数、正确答案 vs 模型回答、Prompt 节选、推理过程节选

这对于手动排查问题已经很有参考价值。

### Q: Tier 1 为空怎么办？

如果 `repeats: 1`（只跑一轮），则每个样本最多只有 1 次结果，"全错"等价于"错了 1 次"，Tier 1 条件（`wrong_count == max_repeat`）仍然成立。但如果模型总体错误率较低，可能确实没有错误率超过 70% 的维度，此时 Tier 1 为空是正常现象，系统会从 Tier 2 和 Tier 3 补充。
