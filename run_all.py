"""统一评测入口

运行所有数据集的评测脚本。
实验配置通过 --experiment-config 参数指定，默认使用 experiment_config.yaml。
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# 数据集列表
DATASETS = [
    "Belief_R",
    "FictionalQA",
    # "FollowBench", 指标异常低
    "HellaSwag",
    # "IFEval", 报错
    "RecToM",
    "SimpleTom",
    "SocialIQA",
    "Tomato",
    "ToMBench",
    # "ToMChallenges", 指标有问题
    # "ToMi",
    # "ToMQA",
    "BigToM",
    "FANToM",
]


def run_dataset(dataset: str, experiment_config_path: str) -> bool:
    """运行指定数据集的评测

    Args:
        dataset: 数据集名称
        experiment_config_path: experiment config 文件路径

    Returns:
        是否成功
    """
    run_script = Path(f"tasks/{dataset}/run.py")
    if not run_script.exists():
        print(f"[{dataset}] run.py not found, skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {dataset}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, str(run_script), "--experiment-config", experiment_config_path],
            check=True,
            capture_output=False,
            env=os.environ,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{dataset}] Error: {e}")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"[{dataset}] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="experiment_config.yaml",
        help="experiment config 文件路径，默认 experiment_config.yaml",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_TIMESTAMP"] = timestamp
    print(f"Run timestamp: {timestamp}")
    print(f"Experiment config: {args.experiment_config}")

    for dataset in DATASETS:
        run_dataset(dataset, args.experiment_config)

    print(f"\n{'='*60}")
    print("All datasets completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()