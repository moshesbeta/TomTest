# 新增模型指南

本指南介绍如何在 TomTest 框架中使用新模型进行评测。

## 概述

TomTest 支持任何兼容 OpenAI API 的模型。添加新模型只需要：

1. 修改 `experiment_config.yaml` 配置
2. 调用评测脚本

## 配置文件

所有实验参数统一在 `experiment_config.yaml` 中配置：

```yaml
# LLM 配置
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}  # 支持环境变量
  api_url: https://api.deepseek.com/v1
  temperature: 0.6
  max_tokens: 32768

# LLM Judge 配置（可选，用于需要 judge 的数据集如 ToMi）
judge:
  model: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.0
  max_tokens: 8

# 实验参数
repeats: 3
max_samples: 0  # 0 表示使用全部样本

# 路径配置
datasets_path: datasets
results_path: results
```

## 模式 1：使用 vLLM（本地模型）

### 1.1 启动 vLLM serve

```bash
vllm serve /path/to/your/model \
    --port 8000 \
    --tensor-parallel-size 1
```

### 1.2 配置实验参数

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: my-model
  api_key: not-needed
  api_url: http://localhost:8000/v1
  temperature: 0.6
  max_tokens: 32768

max_samples: 10  # 先测试 10 个样本
```

### 1.3 运行评测

```bash
# 评测单个数据集
python ToMBench/run.py

# 评测所有数据集
python run_all.py
```

## 模式 2：使用云端 API

### 2.1 DeepSeek

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
```

运行：

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python run_all.py
```

### 2.2 OpenAI

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_url: https://api.openai.com/v1
```

运行：

```bash
export OPENAI_API_KEY="sk-xxx"
python run_all.py
```

### 2.3 其他兼容 OpenAI API 的服务

| 服务 | API URL | Model 名称示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-coder` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| Azure OpenAI | `https://your-resource.openai.azure.com` | `gpt-4` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |

## 配置说明

### LLM 配置

| 字段 | 说明 | 示例 |
|---|---|---|
| `llm.model_name` | 模型名称 | `deepseek-chat`, `gpt-4o` |
| `llm.api_url` | API 端点 URL | `https://api.deepseek.com/v1` |
| `llm.api_key` | API 密钥（支持环境变量） | `${DEEPSEEK_API_KEY}` |
| `llm.temperature` | 采样温度 | `0.6`（0.0 = 确定性，1.0 = 随机） |
| `llm.max_tokens` | 最大输出 token 数 | `32768` |

### Judge 配置（可选）

| 字段 | 说明 | 示例 |
|---|---|---|
| `judge.model` | Judge 模型名称 | `deepseek-chat` |
| `judge.api_url` | Judge API URL | `https://api.deepseek.com/v1` |
| `judge.api_key` | Judge API 密钥 | `${DEEPSEEK_API_KEY}` |
| `judge.temperature` | Judge 温度（通常为 0.0） | `0.0` |
| `judge.max_tokens` | Judge 输出 token 数（通常很小） | `8` |

### 实验参数

| 字段 | 说明 | 默认值 |
|---|---|---|
| `repeats` | 重复运行次数（取平均） | `1` |
| `max_samples` | 最大样本数（0 = 全部） | `0` |
| `datasets_path` | 数据集根目录 | `datasets` |
| `results_path` | 结果输出目录 | `results` |

## 完整示例

### 示例 1：本地模型评测

`experiment_config.yaml`：

```yaml
llm:
  model_name: my-model
  api_key: not-needed
  api_url: http://localhost:8000/v1
  temperature: 0.6
  max_tokens: 32768

repeats: 3
max_samples: 100
```

```bash
# 启动 vLLM
vllm serve /path/to/model --port 8000

# 运行评测
python run_all.py
```

### 示例 2：DeepSeek 评测

`experiment_config.yaml`：

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.6
  max_tokens: 32768

repeats: 3
```

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python run_all.py
```

### 示例 3：冒烟测试

`experiment_config.yaml`：

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1

max_samples: 10  # 只测试 10 个样本
```

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python ToMBench/run.py
```

## 批量评测多模型

创建 `benchmark_all_models.sh`：

```bash
#!/bin/bash

# 为不同模型创建配置文件
cat > experiment_config_deepseek.yaml <<'EOF'
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
repeats: 3
EOF

cat > experiment_config_gpt4o.yaml <<'EOF'
llm:
  model_name: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_url: https://api.openai.com/v1
repeats: 3
EOF

# 运行评测
for CONFIG in experiment_config_*.yaml; do
    echo "=========================================="
    echo "Evaluating with: $CONFIG"
    echo "=========================================="

    # 备份原配置
    cp experiment_config.yaml experiment_config.yaml.backup

    # 使用新配置
    cp "$CONFIG" experiment_config.yaml
    python run_all.py

    # 恢复原配置
    mv experiment_config.yaml.backup experiment_config.yaml
    echo ""
done
```

## 结果查看

所有结果保存在 `results/` 目录：

```bash
# 列出所有结果
ls -lh results/

# 查看最新结果
cat results/$(ls -t results/*.json | head -1) | jq

# 比较不同模型的结果
for f in results/*.json; do
    echo "$f: $(cat $f | jq -r '.metrics[0].accuracy')"
done
```

## 调试技巧

### 1. 使用少量样本快速验证

编辑 `experiment_config.yaml`：

```yaml
max_samples: 3  # 只测试前 3 条样本
```

### 2. 检查 API 连接

```bash
curl http://localhost:8000/v1/models
```

### 3. 查看详细日志

在运行脚本时添加 `--verbose`（如果需要）

## 常见问题

### Q: 提示 "Connection refused"

确保 vLLM 服务正在运行：

```bash
curl http://localhost:8000/v1/models
```

### Q: API 密钥错误

检查环境变量是否正确设置：

```bash
echo $DEEPSEEK_API_KEY
```

### Q: 模型不支持结构化输出

OpenAI 的最新模型（gpt-4o, gpt-4o-mini）支持结构化输出。如果模型不支持，需要修改代码使用旧的 `generate` 方法。

### Q: 如何设置不同的温度？

编辑 `experiment_config.yaml`：

```yaml
llm:
  temperature: 0.1  # 更确定的输出
```

或

```yaml
llm:
  temperature: 0.9  # 更随机的输出
```
