# TomTest

基于结构化输出的 Theory-of-Mind（心智理论）评测框架，支持多数据集、多模型的基准评测与深度分析。

## 设计理念

**结构化输出优先** - 使用 Pydantic 定义输出 Schema，直接从结构化对象获取答案，避免复杂的字符串解析：

- **新增模型**：只需修改 `experiment_config.yaml` 中的 API 配置，无需改代码
- **新增数据集**：复用现有 Schema，只需编写 `prompts.py` 和 `metrics.py`
- **自动降级**：模型不支持原生结构化输出时，自动切换为 JSON Prompt 注入 + 正则提取

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 主要依赖：openai pyyaml tqdm datasets pyarrow pydantic
```

### 2. 配置实验参数

编辑 `experiment_config.yaml`，填写模型 API 信息：

```yaml
llm:
  model_name: Qwen3-8B           # 模型名称（也是 results/ 下的目录名）
  api_key: not-needed            # 本地服务填 not-needed，云端 API 填密钥
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.6
  max_tokens: 32768
  max_workers: 64
  enable_thinking: false         # Qwen3 等支持 thinking 的模型可设为 true
  system_prompt: ""

judge:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.0
  max_tokens: 4096
  use_llm_judge: false           # ToMQA 等开放式问答数据集需要设为 true

repeats: 3       # 重复轮数（本地推理建议 3，云端 API 建议 1）
max_samples: 0   # 0=全量，>0=随机抽样（调试时用 3-10）
datasets_path: datasets
results_path: results
```

### 3. 运行评测

```bash
# 运行所有数据集（约 10 个）
python run_all.py

# 运行单个数据集
python tasks/ToMBench/run.py
```

### 4. 生成结果表格

```bash
# 从 results/ 生成各数据集详细表格
python report/generate_dataset_tables.py

# 生成跨模型、跨数据集的汇总表
python report/generate_summary.py

# 查看汇总
cat tables/SUMMARY.md
```

### 5. Bad Case 分析

```bash
# 编辑 report/report_config.yaml 后运行
python report/report_client.py
```

---

## 目录结构

```
TomTest/
├── experiment_config.yaml        # 全局实验配置（模型 API、轮数、样本数等）
├── run_all.py                    # 全量运行入口（运行所有注册的数据集）
├── requirements.txt              # Python 依赖
│
├── tasks/                        # 各数据集评测代码
│   ├── ToMBench/
│   │   ├── config.yaml           # 数据集配置（path、schema、method）
│   │   ├── prompts.py            # build_prompt(row, method) → str
│   │   ├── metrics.py            # compute_metrics(predictions, data, judge) → dict
│   │   └── run.py                # 评测主脚本
│   ├── SocialIQA/
│   ├── ToMChallenges/
│   └── ...（共 12 个任务目录）
│
├── src/                          # 核心框架代码
│   ├── schemas.py                # Pydantic 输出 Schema（MCQAnswer, OpenAnswer 等）
│   ├── runner.py                 # 公共 runner 工具（加载配置、保存结果等）
│   ├── utils.py                  # 指标计算工具（compute_sample_metrics 等）
│   ├── llm/
│   │   ├── client.py             # LLMClient 基类（usage 统计、OpenAI 初始化）
│   │   ├── content_client.py     # ContentClient（自由文本生成）
│   │   ├── structure_client.py   # StructureClient（结构化输出，含自动降级）
│   │   └── llm_utils.py          # extract_json、format_schema_for_prompt 等
│   └── dataloader/
│       └── dataloader.py         # Arrow 格式数据集加载器
│
├── datasets/                     # Arrow 格式数据集（共 20+ 个）
│   ├── Belief_R/, FictionalQA/, HellaSwag/, RecToM/
│   ├── SimpleToM/, SocialIQA/, Tomato/, ToMBench/
│   ├── ToMChallenges/, ToMi/, ToMQA/, EmoBench/
│   └── ...
│
├── datasets_examples/            # 各数据集首条样本 JSON（快速了解字段格式）
│
├── results/                      # 评测输出（自动生成）
│   └── {dataset}/{model}/exp_{timestamp}/
│       ├── config.json           # 完整实验配置（API key 已脱敏）
│       ├── metrics.json          # 指标（avg_metrics + all_metrics）
│       └── prediction.jsonl      # 每条样本的预测详情
│
├── tables/                       # 结果表格（generate_dataset_tables.py 生成）
│   ├── SUMMARY.md                # 跨数据集 × 模型 accuracy 汇总
│   └── {dataset}/
│       ├── 基础指标.md            # accuracy、correct、total
│       └── 其他指标.md            # 细粒度指标（如 by_ability）
│
├── report/                       # 报告生成工具
│   ├── generate_dataset_tables.py  # 生成各数据集表格
│   ├── generate_summary.py         # 生成汇总表格
│   ├── report_client.py            # Bad Case 分析 + LLM 诊断报告
│   ├── tables_config.yaml          # generate_dataset_tables.py 配置
│   └── report_config.yaml          # report_client.py 配置
│
├── analysis/                     # Bad Case 分析报告（report_client.py 生成）
│   └── {dataset}/{model}/{timestamp}.md
│
├── docs/                         # 详细操作指南
│   ├── add_new_dataset.md        # 如何新增数据集
│   ├── add_new_model.md          # 如何用新模型测试
│   ├── generate_tables.md        # 如何生成数据集表格
│   ├── generate_summary.md       # 如何生成汇总表格
│   └── bad_case_analysis.md      # 如何做 Bad Case 分析
│
└── tests/
    └── test_dataloader.py        # 数据集加载验证
```

---

## 支持的数据集（当前启用 10 个）

| 数据集 | Schema | 说明 |
|---|---|---|
| Belief_R | `MCQAnswer3Lower` | 信念追踪（a/b/c 三选一） |
| FictionalQA | `MCQAnswer` | 虚构故事知识问答（四选一） |
| HellaSwag | `MCQAnswer` | 常识推理（四选一） |
| RecToM | `MultiLabelAnswer` | 推荐场景下的心智理论（多选） |
| SimpleTom | `MCQAnswer` | 简单 ToM 故事理解 |
| SocialIQA | `MCQAnswer3` | 社交情境理解（A/B/C 三选一） |
| Tomato | `MCQAnswer` | 多心智状态 MCQ（含选项随机打乱） |
| ToMBench | `MCQAnswer` | ToM 综合基准（中英文双语） |
| ToMChallenges | `MCQAnswer` | Anne-Sally & Smarties 经典测试 |
| ToMQA | `OpenAnswer` | 开放式 ToM 问答（LLM Judge） |

---

## 可用的 Schema（src/schemas.py）

| Schema | 字段 | 适用场景 |
|---|---|---|
| `MCQAnswer` | `answer: Literal["A","B","C","D"]` | 标准四选一 MCQ |
| `MCQAnswer3` | `answer: Literal["A","B","C"]` | 三选一 MCQ（大写） |
| `MCQAnswer3Lower` | `answer: Literal["a","b","c"]` | 三选一 MCQ（小写） |
| `OpenAnswer` | `answer: str` | 开放式问答 |
| `OneWordAnswer` | `answer: str`（无空白） | 单词/短语回答 |
| `MultiLabelAnswer` | `answer: List[str]` | 多标签多选 |
| `JudgeAnswer` | `answer: Literal["True","False"]` | LLM Judge 判断 |

---

## 完整工作流

```
Step 1  配置 experiment_config.yaml（填写模型 API）
   ↓
Step 2  运行评测：python run_all.py
   ↓
Step 3  生成表格：python report/generate_dataset_tables.py
   ↓
Step 4  生成汇总：python report/generate_summary.py
   ↓
Step 5  Bad Case 分析：python report/report_client.py
```

---

## 操作指南

| 任务 | 文档 |
|---|---|
| 新增一个数据集进行评测 | [docs/add_new_dataset.md](docs/add_new_dataset.md) |
| 使用新模型（本地/云端）测试 | [docs/add_new_model.md](docs/add_new_model.md) |
| 从评测结果生成数据集对比表格 | [docs/generate_tables.md](docs/generate_tables.md) |
| 生成跨模型、跨数据集的汇总表 | [docs/generate_summary.md](docs/generate_summary.md) |
| 分析模型错误案例（Bad Case） | [docs/bad_case_analysis.md](docs/bad_case_analysis.md) |

---

## 更新日志

### 2026-04-22

#### 新增 9 个数据集评测任务

新增以下数据集的完整评测代码（`config.yaml` / `prompts.py` / `metrics.py` / `run.py`）：

| 数据集 | 类型 | 说明 |
|---|---|---|
| Belief_R | MCQ | 信念推理（Belief Revision） |
| FictionalQA | MCQ | 虚构场景问答 |
| FollowBench | 指令遵循 | 多层级指令遵循，按约束条数分层评分 |
| HellaSwag | MCQ | 常识推理补全 |
| IFEval | 指令遵循 | 可验证指令精确遵循（含独立指令验证库） |
| RecToM | MCQ | 递归心智理论推理 |
| SimpleTom | MCQ | 简化版心智理论基础评测 |
| SocialIQA | MCQ | 社交情境常识推理 |
| ToMChallenges | MCQ | 高难度心智理论挑战集 |

#### LLM 客户端模块拆分重构

原 `src/llm/client.py` 单文件拆分为四个职责独立的模块：

- **`client.py`**：基础 `LLMClient`，管理 OpenAI 连接、generation 参数、线程安全的用量统计（`LLMUsage`）
- **`content_client.py`** （新增）：`ContentClient`，专用文本生成，支持 `batch_generate()` 多线程并发、`tqdm` 进度显示
- **`structure_client.py`** （新增）：`StructureClient`，专用结构化输出，支持 Pydantic Schema；模型不支持原生 `response_format` 时自动降级为 JSON Prompt 注入 + 正则提取
- **`llm_utils.py`** （新增）：公共工具函数，包括 `build_extra_body`（`top_k` / `enable_thinking` 参数封装）

#### 运行器与任务体系统一化

- **`src/runner.py`**：新增 `sample_metas` 参数，支持将每条样本的 `Meta` 字典写入 `prediction.jsonl`，便于后续按维度分析；`create_judge_client()` 支持数据集级别的 `use_llm_judge` 覆盖全局配置；输出目录时间戳改为由 `run_all.py` 统一生成并通过环境变量传递，保证同一批次所有数据集落在同一 `exp_*` 目录
- **`run_all.py`**：启动时生成唯一 `RUN_TIMESTAMP` 并注入 `os.environ`，所有子进程共享同一时间戳
- **`src/schemas.py`** （新增）：统一定义各任务通用 Pydantic Schema（`MCQAnswer` 等），各任务不再维护独立 `schemas.py`
- **`src/utils.py`** （新增）：公共工具函数库
- **所有任务 `run.py`**：统一更新 `create_judge_client()` 和 `save_common_results()` 调用签名，传入 `dataset_config` 和 `sample_metas`
- **所有任务 `config.yaml`**：新增注释说明字段 `use_llm_judge`，支持数据集级别覆盖全局 judge 配置
- **`tasks/ToMi/prompts.py`**：修复字段名错误（`instruction`/`input` → `Story.full_story`/`Question`，与数据集实际列名对齐）
- **移除**：各任务独立的 `schemas.py`（`ToMBench`、`ToMi`、`ToMQA`、`Tomato`）

#### 报告与分析工具迁移至 `report/` 目录

- **`generate_dataset_tables.py`、`generate_summary.py`** 从项目根目录移入 `report/`，根目录对应文件已删除
- **`report/report_client.py`** （新增）：Bad Case 深度分析工具；从 `tables/` 读取多维度指标并支持基线对比；按维度错误率分三层优先级（Tier 1/2/3）抽取典型错误样本；调用本地 LLM API 批量分析错误原因与改善方向；可导出 Markdown 报告到 `analysis/`
- **`report/report_config.yaml`** （新增）：`report_client.py` 配置文件模板
- **`report/tables_config.yaml`** （新增）：`generate_dataset_tables.py` 配置文件
- **`report/README.md`** （新增）：三个报告工具的功能说明与使用指南

#### 文档更新

- **`docs/add_new_dataset.md`**：新增数据集准备流程（HuggingFace Hub 下载、本地转换、`datasets_examples/` 验证），内容约翻倍
- **`docs/add_new_model.md`**：重组为本地 vLLM 与云端 API 两条路径，补充 smoke test（`max_samples: 3`）和故障排查
- **新增** `docs/generate_tables.md`、`docs/generate_summary.md`、`docs/bad_case_analysis.md`：对应三个报告工具的完整操作文档

#### 配置与环境

- **`experiment_config.yaml`**：默认值调整为本地部署场景（`enable_thinking: true`、`repeats: 3`、`max_samples: 0` 全量）
- **`.gitignore`**：新增忽略 `datasets/`、`datasets_examples/`、`analysis/`、`__pycache__/`、`*.pyc`

#### 评测结果（tables/）

新增 Qwen3-8B（thinking 关闭）与 Qwen3-8B-Think（thinking 开启）在 10 个数据集上的完整对比结果，包含 `基础指标.md`、`其他指标.md` 和 `config.json`。

---

## 许可证

MIT License
