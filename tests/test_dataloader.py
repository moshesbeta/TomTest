"""测试 dataloader"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataloader.dataloader import load_dataset, list_subsets


def test():
    datasets_root = "./datasets"
    subsets = list_subsets(datasets_root)

    success, failed = [], []
    for subset in subsets:
        try:
            data = load_dataset(subset, datasets_root)
            if data:
                success.append(f"{subset} ({len(data)} samples)")
            else:
                failed.append(f"{subset} (empty)")
        except Exception as e:
            failed.append(f"{subset} ({e})")

    print(f"成功 ({len(success)}):")
    for s in success:
        print(f"  ✓ {s}")

    print(f"\n失败 ({len(failed)}):")
    for s in failed:
        print(f"  ✗ {s}")


if __name__ == "__main__":
    test()
