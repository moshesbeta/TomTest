# 生成数据集表格指南

本指南介绍如何使用 `report/generate_dataset_tables.py` 脚本，将 `results/` 目录下的评测结果转换为人类可读的 Markdown 表格。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [配置文件 tables_config.yaml](#配置文件-tables_configyaml)
- [输出文件说明](#输出文件说明)
- [增量更新（新增模型列）](#增量更新新增模型列)
- [常见用法示例](#常见用法示例)

---

## 概述

脚本读取 `results/` 下所有 `metrics.json` 文件，为每个数据集生成两个 Markdown 表格文件：

| 输出文件 | 内容 |
|---|---|
| `tables/{DatasetName}/基础指标.md` | accuracy、correct、total |
| `tables/{DatasetName}/其他指标.md` | 其余所有指标（含按维度细分的字典类指标） |

同时将 `config.json` 复制到 `tables/{DatasetName}/{model_name}/config.json` 作为配置存档。

---

## 快速开始

```bash
cd /path/to/TomTest

# 使用默认配置（处理 results/ 下所有数据集和模型）
python report/generate_dataset_tables.py

# 使用自定义配置文件
python report/generate_dataset_tables.py report/tables_config.yaml
```

---

## 配置文件 tables_config.yaml

脚本通过 `report/tables_config.yaml` 配置所有参数：

```yaml
# results 目录路径（相对于 TomTest 根目录）
results_dir: results

# 表格输出目录路径
output_dir: tables

# 实验时间后缀（可选）
# 填写后只处理 exp_{exp_suffix} 目录的结果
# 不填则自动选择每个数据集/模型下最新的 exp_* 目录
exp_suffix:

# 只处理指定数据集（可选，不填处理全部）
dataset: ToMBench

# 只处理指定模型（可选，不填处理全部）
# 支持两种写法：
models:
  # 写法1：字符串，目录名即为表格列名
  - Qwen3-8B

  # 写法2：字典，name 为 results/ 下的目录名，display 为表格列名
  - name: Qwen3-8B
    display: Qwen3-8B-Think
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `results_dir` | results 目录路径 | `results` |
| `output_dir` | 表格输出目录 | `tables` |
| `exp_suffix` | 指定实验时间戳后缀，如 `20260422_143022` | 空（自动选最新） |
| `dataset` | 只处理该数据集，不填则处理全部 | 空（全部） |
| `models` | 只处理指定模型列表，不填则处理全部 | 空（全部） |

---

## 输出文件说明

### 基础指标.md

包含 accuracy、correct、total 三个基础指标，每行是一个指标，每列是一个模型：

```markdown
# ToMBench - 基础指标

| 指标 \ 模型 | Qwen3-0.6B | Qwen3-4B | Qwen3-8B |
|---|---|---|---|
| accuracy | 0.6421 | 0.7012 | 0.7340 |
| correct | 650 | 710 | 743 |
| total | 1012 | 1012 | 1012 |
```

### 其他指标.md

包含数据集特有的二级指标，按 `## 节标题` 分组：

```markdown
# ToMBench - 其他指标

## 标量指标

| 指标 \ 模型 | Qwen3-8B |
|---|---|
| some_scalar | 0.8012 |

## by_ability

| 子指标 \ 模型 | Qwen3-8B |
|---|---|
| Belief: Content false beliefs | 0.7143 |
| Knowledge: Percepts-knowledge links | 0.8021 |
| Desire: Goal-based predictions | 0.6891 |
| ...                                 | ...    |
```

---

## 增量更新（新增模型列）

当表格已存在时，再次运行脚本会检测到同名模型列冲突并提示是否覆盖：

```
数据集 [ToMBench] 中以下模型已有结果: Qwen3-8B
是否覆盖？(y/n):
```

- 输入 `y`：用新数据覆盖该模型的旧结果
- 输入 `n`：保留旧结果，跳过该模型

如果是全新模型，则无提示，直接追加为新列。

### 批量新增多个模型

```yaml
# tables_config.yaml
models:
  - Qwen3-0.6B
  - Qwen3-4B
  - Qwen3-8B
  - name: gemma-3-4b-it
    display: Gemma-3-4B
```

---

## 常见用法示例

### 处理单个数据集

```yaml
# tables_config.yaml
dataset: ToMBench
models:
```

### 处理所有数据集，所有模型

```yaml
# tables_config.yaml
dataset:
models:
```

### 指定实验版本（防止最新实验覆盖之前的结果对比）

```yaml
# tables_config.yaml
exp_suffix: 20260422_143022
```

### 为模型设置别名（方便在论文/报告中展示）

```yaml
# tables_config.yaml
models:
  - name: Qwen3-8B            # results/ 目录名
    display: Qwen3-8B (base)  # 表格展示名
  - name: Qwen3-8B-Think
    display: Qwen3-8B (think)
```

### 查看当前所有已有结果

```bash
# 列出所有数据集和模型的实验目录
find results -name "metrics.json" | sort
```
