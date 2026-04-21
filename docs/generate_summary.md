# 生成汇总表格指南

本指南介绍如何使用 `report/generate_summary.py` 脚本，生成跨数据集、跨模型的 accuracy 汇总对比表。

## 目录

- [概述](#概述)
- [前提条件](#前提条件)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [输出文件说明](#输出文件说明)
- [典型工作流](#典型工作流)

---

## 概述

`generate_summary.py` 读取 `tables/` 目录下各数据集的 `基础指标.md` 文件，提取各模型的 accuracy 值，汇总生成一张 **数据集 × 模型** 的对比表格，保存到 `tables/SUMMARY.md`。

---

## 前提条件

**必须先运行 `generate_dataset_tables.py`**，确保 `tables/` 目录下已有各数据集的基础指标文件：

```
tables/
├── ToMBench/
│   └── 基础指标.md    ← 必须存在
├── SocialIQA/
│   └── 基础指标.md
├── ...
└── SUMMARY.md         ← 本脚本生成此文件
```

如果 `tables/` 为空，请先参考 [generate_tables.md](generate_tables.md) 生成各数据集表格。

---

## 快速开始

```bash
cd /path/to/TomTest

# 使用默认路径（读取 tables/，输出 tables/SUMMARY.md）
python report/generate_summary.py

# 同时输出到终端查看
python report/generate_summary.py --stdout
```

---

## 命令行参数

```
usage: generate_summary.py [-h] [--tables-dir TABLES_DIR] [--output-file OUTPUT_FILE] [--stdout]

options:
  --tables-dir     tables 目录路径（默认: tables）
  --output-file    输出文件路径（默认: tables/SUMMARY.md）
  --stdout         同时将汇总表格输出到终端
```

### 示例

```bash
# 自定义路径
python report/generate_summary.py \
    --tables-dir /path/to/tables \
    --output-file /path/to/SUMMARY.md

# 只打印不保存文件
python report/generate_summary.py --output-file /dev/null --stdout
```

---

## 输出文件说明

`tables/SUMMARY.md` 的内容示例：

```markdown
## 总览表格：Accuracy

| 数据集 \ 模型 | Gemma-3-4B | Qwen3-0.6B | Qwen3-4B | Qwen3-8B |
|---|---:|---:|---:|---:|
| Belief_R | 0.4667 | 0.3333 | 0.6333 | 0.7000 |
| FictionalQA | - | - | 0.7333 | 0.8000 |
| HellaSwag | 0.7333 | 0.4667 | 0.7000 | 0.8333 |
| RecToM | 0.5000 | 0.3000 | 0.6000 | 0.6500 |
| SimpleTom | 0.8000 | 0.6667 | 0.8667 | 0.9000 |
| SocialIQA | 0.6667 | 0.5333 | 0.7333 | 0.7667 |
| Tomato | 0.6769 | 0.5231 | 0.6769 | 0.7538 |
| ToMBench | 0.7333 | 0.5412 | 0.6982 | 0.7340 |
| ToMChallenges | 0.7667 | 0.5000 | 0.7667 | 0.8000 |
| ToMQA | 0.5611 | - | 0.5833 | 0.6222 |
```

- 模型列按字母序排列
- 数据集行按字母序排列
- `-` 表示该模型在该数据集上暂无结果（未运行或未生成表格）

---

## 典型工作流

完整的"评测 → 生成表格 → 生成汇总"工作流如下：

```bash
# Step 1：运行所有数据集评测
python run_all.py

# Step 2：从 results/ 生成各数据集详细表格
python report/generate_dataset_tables.py

# Step 3：生成跨模型汇总表
python report/generate_summary.py

# 查看结果
cat tables/SUMMARY.md
```

每次新增一个模型后，重复 Step 1-3 即可在 SUMMARY.md 中看到新模型列。
