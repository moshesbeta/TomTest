"""
Results Summary Generator

从 results/ 目录读取 metrics.json 文件，生成汇总报告：
1. 总览表格：数据集 × 模型（展示 accuracy）
2. 每个数据集的详细表格：指标 × 模型（分为基础指标.md 和其他指标.md）
"""
import json
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

    # 遍历每个数据集目录
    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_name = dataset_dir.name
        metrics_data[dataset_name] = {}

        # 遍历每个模型目录
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
                    # 排除嵌套字典
                    if not isinstance(model_metrics["avg_metrics"][key], dict):
                        metric_names.add(key)

    return sorted(metric_names)


def generate_summary_table(metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """生成总览表格：数据集 × 模型（accuracy）

    Args:
        metrics_data: 收集的 metrics 数据

    Returns:
        Markdown 表格字符串
    """
    # 获取所有数据集和模型
    datasets = sorted(metrics_data.keys())
    models = set()
    for dataset_metrics in metrics_data.values():
        models.update(dataset_metrics.keys())
    models = sorted(models)

    if not datasets or not models:
        return "## 总览表格\n\n没有找到任何评测结果。\n"

    # 构建表格
    lines = [
        "## 总览表格：Accuracy",
        "",
        "| 数据集 \\ 模型 | " + " | ".join(models) + " |",
        "|" + "---|" * (len(models) + 1),
    ]

    for dataset in datasets:
        row = [dataset]
        for model in models:
            if model in metrics_data[dataset]:
                acc = metrics_data[dataset][model].get("avg_metrics", {}).get("accuracy", 0.0)
                row.append(f"{acc:.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


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


def generate_report(results_dir: str = "results", output_dir: str = "results") -> None:
    """生成完整的汇总报告

    Args:
        results_dir: results 目录路径
        output_dir: 输出目录路径
    """
    # 收集所有 metrics
    metrics_data = collect_metrics(results_dir)

    if not metrics_data:
        print("没有找到任何评测结果。")
        return

    output_path = Path(output_dir)

    # 1. 生成 SUMMARY.md（总览表格）
    summary = generate_summary_table(metrics_data)
    summary_path = output_path / "SUMMARY.md"
    summary_path.write_text(summary, encoding='utf-8')
    print(f"总览表格已保存到: {summary_path}")

    # 2. 为每个数据集生成详细文件
    for dataset_name in sorted(metrics_data.keys()):
        models = sorted(metrics_data[dataset_name].keys())

        # 创建数据集子目录
        dataset_report_dir = output_path / dataset_name
        dataset_report_dir.mkdir(parents=True, exist_ok=True)

        # 生成基础指标.md
        basic_metrics = generate_basic_metrics_table(dataset_name, models, metrics_data)
        basic_path = dataset_report_dir / "基础指标.md"
        basic_path.write_text(basic_metrics, encoding='utf-8')

        # 生成其他指标.md
        other_metrics = generate_other_metrics_table(dataset_name, models, metrics_data)
        other_path = dataset_report_dir / "其他指标.md"
        other_path.write_text(other_metrics, encoding='utf-8')

        print(f"数据集 [{dataset_name}] 报告已保存到: {dataset_report_dir}/")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="从 results/ 目录生成评测结果汇总报告",
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
        default="results",
        help="输出目录路径（默认: results）"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="将总览表格输出到标准输出"
    )

    args = parser.parse_args()

    # 生成报告
    generate_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    # 输出到标准输出（如果指定）
    if args.stdout:
        summary = generate_summary_table(collect_metrics(args.results_dir))
        print(summary)


if __name__ == "__main__":
    main()
