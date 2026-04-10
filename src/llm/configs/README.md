# LLM Configs 目录

本目录存放常见的 LLM 配置文件，方便快速创建 LLMClient 实例。

## 使用方法

```python
from llm import LLMClient
import json

# 从配置文件加载
with open("llm/configs/openai_gpt4o.json", "r") as f:
    config = json.load(f)

client = LLMClient.from_config(config)
```

## 配置文件列表

| 文件 | 模型 | 说明 |
|------|------|------|
| `openai_gpt4o.json` | gpt-4o | OpenAI GPT-4o |
| `openai_gpt4o_mini.json` | gpt-4o-mini | OpenAI GPT-4o mini |
| `deepseek_chat.json` | deepseek-chat | DeepSeek Chat |
| `local_vllm.json` | Qwen2.5-72B-Instruct | 本地 vLLM 部署 |
| `baidu_ernie4.json` | ernie-4.0-8k | 百度文心一言 4.0 |

## 配置字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `model_name` | 是 | 模型名称 |
| `api_key` | 是 | API 密钥 |
| `api_url` | 是 | API 基础 URL |
| `temperature` | 否 | 采样温度，默认 0.6 |
| `max_tokens` | 否 | 最大生成 tokens，默认 32768 |
| `top_p` | 否 | nucleus sampling，默认 0.95 |
| `max_workers` | 否 | 批量并发数，默认 32 |

## 添加新配置

复制现有配置文件，修改 `model_name`、`api_key`、`api_url` 等字段即可。
