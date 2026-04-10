# DataLoader

从 `datasets/` 目录加载数据集。

## 使用方法

```python
from src.dataloader import load_dataset, list_subsets

# 列出所有可用子集
subsets = list_subsets()
# ['ExploreToM/SIP/raw', 'ExploreToM/SIP/all_wrong', 'SocialIQA/SIP/all_wrong', ...]

# 加载特定子集
data = load_dataset("ExploreToM/SIP/raw")
print(len(data))  # 数据条数
print(data[0].keys())  # 字段名
```

## 数据集目录结构

```
datasets/
├── ExploreToM/
│   └── SIP/
│       ├── raw/
│       │   ├── dataset_info.json
│       │   └── data-00000-of-00001.arrow
│       └── all_wrong/
│           ├── dataset_info.json
│           └── data-00000-of-00001.arrow
└── SocialIQA/
    └── SIP/
        └── all_wrong/
            ├── dataset_info.json
            └── data-00000-of-00001.arrow
```

## API

### `load_dataset(subset, datasets_root=None)`

加载数据集。

- `subset` (str): 子集路径，如 "ExploreToM/SIP/raw"
- `datasets_root` (可选): 自定义根目录

返回 `List[Dict]`

### `list_subsets(datasets_root=None)`

列出所有可用子集。

返回 `List[str]`
