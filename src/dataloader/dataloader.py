"""
Dataset Loader

根据名称加载 datasets 目录下的数据集。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DataLoader:
    """数据集加载器"""

    def __init__(self, datasets_root: Union[str, Path] = None):
        self.datasets_root = Path(datasets_root) if datasets_root else Path(__file__).parent.parent.parent / "datasets"

    def load(self, subset: str) -> List[Dict[str, Any]]:
        """加载数据集。

        Args:
            subset: 子集路径，如 "ExploreToM/SIP/raw"

        Returns:
            数据列表
        """
        from datasets import load_from_disk

        path = self.datasets_root / subset
        dataset = load_from_disk(str(path))
        return dataset.to_list()

    def list_subsets(self) -> List[str]:
        """列出所有可用的子集路径。"""
        subsets = []
        for ds_dir in self.datasets_root.iterdir():
            if ds_dir.is_dir() and not ds_dir.name.startswith("."):
                subsets.extend(self._find_arrow_subsets(ds_dir, ds_dir.name))
        return subsets

    def _find_arrow_subsets(self, path: Path, prefix: str) -> List[str]:
        """递归查找包含 arrow 文件的目录。"""
        subsets = []
        arrow_files = list(path.glob("*.arrow"))
        if arrow_files:
            subsets.append(prefix)
        for sub_dir in path.iterdir():
            if sub_dir.is_dir():
                subsets.extend(self._find_arrow_subsets(sub_dir, f"{prefix}/{sub_dir.name}"))
        return subsets


def load_dataset(subset: str, datasets_root: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """便捷函数：加载数据集。

    Args:
        subset: 子集路径，如 "ExploreToM/SIP/raw"
        datasets_root: 数据集根目录

    Returns:
        数据列表

    Examples:
        data = load_dataset("ExploreToM/SIP/raw")
    """
    loader = DataLoader(datasets_root)
    return loader.load(subset)


def list_subsets(datasets_root: Union[str, Path] = None) -> List[str]:
    """便捷函数：列出所有子集。"""
    loader = DataLoader(datasets_root)
    return loader.list_subsets()