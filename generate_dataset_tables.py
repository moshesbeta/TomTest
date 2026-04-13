"""
Dataset Tables Generator

从 results/ 目录读取 metrics.json 文件，为每个数据集生成详细表格：
1. 基础指标.md：accuracy, correct, total
2. 其他指标.md：其他所有指标
3. 复制 config.json 到输出目录
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set


def collect_metrics(results_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """收集所有数据集和模型的 metrics

    Args:
        results_dir: results 目录路径

    Returns:
        {dataset_name: {model_name: metrics_data}}
    """
    results_path = Path(results_dir)
    metrics_data = {}

    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_name = dataset_dir.name
        metrics_data[dataset_name] = {}

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            model_name = model_dir.name
            metrics_file = model_dir / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data[dataset_name][model_name] = json.load(f)

    return metrics_data


def get_all_metrics_names(metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Set[str]:
    """获取所有出现的指标名称

    Args:
        metrics_data: 收集的 metrics 数据

    Returns:
        所有指标名称的集合
    """
    metric_names = set()

    for dataset_metrics in metrics_data.values():
        for model_metrics in dataset_metrics.values():
            if "avg_metrics" in model_metrics:
                for key in model_metrics["avg_metrics"].keys():
                    if not isinstance(model_metrics["avg_metrics"][key], dict):
                        metric_names.add(key)

    return sorted(metric_names)


def generate_basic_metrics_table(dataset_name: str, models: List[str], metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """生成基础指标表格：accuracy, correct, total

    Args:
        dataset_name: 数据集名称
        models: 模型列表
        metrics_data: 收集的 metrics 数据

    Returns:
        Markdown 表格字符串
    """
    lines = [
        f"# {dataset_name} - 基础指标",
        "",
        "| 指标 \\ 模型 | " + " | ".join(models) + " |",
        "|" + "---|" * (len(models) + 1),
    ]

    basic_metrics = ["accuracy", "correct", "total"]
    for metric in basic_metrics:
        row = [metric]
        for model in models:
            if model in metrics_data[dataset_name]:
                value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(metric, "")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_other_metrics_table(dataset_name: str, models: List[str], metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """生成其他指标表格

    Args:
        dataset_name: 数据集名称
        models: 模型列表
        metrics_data: 收集的 metrics 数据

    Returns:
        Markdown 表格字符串
    """
    metric_names = get_all_metrics_names({dataset_name: metrics_data[dataset_name]})
    basic_metrics = ["accuracy", "correct", "total"]
    other_metrics = [m for m in metric_names if m not in basic_metrics]

    if not other_metrics:
        return f"# {dataset_name} - 其他指标\n\n该数据集没有其他指标。\n"

    lines = [
        f"# {dataset_name} - 其他指标",
        "",
        "| 指标 \\ 模型 | " + " | ".join(models) + " |",
        "|" + "---|" * (len(models) + 1),
    ]

    for metric in other_metrics:
        row = [metric]
        for model in models:
            if model in metrics_data[dataset_name]:
                value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(metric, "")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_dataset_tables(results_dir: str = "results", output_dir: str = "tables") -> None:
    """生成所有数据集的指标表格

    Args:
        results_dir: results 目录路径
        output_dir: 输出目录路径
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    metrics_data = collect_metrics(results_dir)

    if not metrics_data:
        print("没有找到任何评测结果。")
        return

    for dataset_name in sorted(metrics_data.keys()):
        models = sorted(metrics_data[dataset_name].keys())

        dataset_report_dir = output_path / dataset_name
        dataset_report_dir.mkdir(parents=True, exist_ok=True)

        basic_metrics = generate_basic_metrics_table(dataset_name, models, metrics_data)
        basic_path = dataset_report_dir / "基础指标.md"
        basic_path.write_text(basic_metrics, encoding='utf-8')

        other_metrics = generate_other_metrics_table(dataset_name, models, metrics_data)
        other_path = dataset_report_dir / "其他指标.md"
        other_path.write_text(other_metrics, encoding='utf-8')

        for model_name in models:
            model_dir = results_path / dataset_name / model_name
            config_file = model_dir / "config.json"
            if config_file.exists():
                output_model_dir = dataset_report_dir / model_name
                output_model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, output_model_dir / "config.json")

        print(f"数据集 [{dataset_name}] 报告已保存到: {dataset_report_dir}/")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="从 results/ 目录生成各数据集的指标表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="results 目录路径（默认: results）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables",
        help="输出目录路径（默认: tables）"
    )

    args = parser.parse_args()

    generate_dataset_tables(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
