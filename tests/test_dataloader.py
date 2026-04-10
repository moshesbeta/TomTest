"""测试 dataloader"""
import sys
from pathlib import Path
import json
import os

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataloader.dataloader import load_dataset, list_subsets, DataLoader


def test():
    # 获取 datasets 目录
    datasets_root = "./datasets"

    print("=== 列出所有子集 ===")
    subsets = list_subsets(datasets_root)
    for s in subsets:
        print(f"  {s}")
    print()

    # 创建保存目录
    save_dir = "./datasets_examples"
    os.makedirs(save_dir, exist_ok=True)

    if subsets:
        for subset in subsets:
            print(f"=== 加载 {subset} ===")
            data = load_dataset(subset, datasets_root)
            print(f"  数量: {len(data)}")
            if data:
                example = data[0]
                print(f"  字段: {list(example.keys())}")
                print(f"  example[0]: {example}")
                print()

                # 保存 example[0] 到 JSON 文件
                safe_name = subset.replace("/", "_").replace("\\", "_")
                save_path = os.path.join(save_dir, f"{safe_name}_example_0.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(example, f, ensure_ascii=False, indent=2, default=str)
                print(f"  已保存到: {save_path}")
                print()


if __name__ == "__main__":
    test()