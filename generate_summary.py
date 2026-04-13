"""
Summary Generator

从已生成的指标表格文件生成总览汇总：
- 从 tables/ 目录读取各数据集的基础指标.md
- 生成总览表格 SUMMARY.md（只显示 accuracy）
"""
from pathlib import Path
from typing import Any, Dict


def parse_basic_metrics_table(table_dir: Path) -> Dict[str, Dict[str, Any]]:
    """解析基础指标.md 文件，提取数据

    Args:
        table_dir: tables 目录下的数据集子目录

    Returns:
        {model_name: {metric_name: value}}
    """
    basic_metrics_file = table_dir / "基础指标.md"
    if not basic_metrics_file.exists():
        return {}

    metrics = {}
    content = basic_metrics_file.read_text(encoding='utf-8')

    lines = content.strip().split('\n')
    data_lines = [line for line in lines if line.startswith('|') and not '---' in line]

    if len(data_lines) < 2:
        return {}

    header = [cell.strip() for cell in data_lines[0].split('|')[1:-1]]
    models = header[1:]

    for line in data_lines[1:]:
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if len(cells) != len(models) + 1:
            continue

        metric_name = cells[0]
        for i, model in enumerate(models):
            value = cells[i + 1]
            if value == '-':
                continue

            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            if model not in metrics:
                metrics[model] = {}
            metrics[model][metric_name] = value

    return metrics


def collect_metrics_from_tables(tables_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """从 tables 目录收集所有基础指标

    Args:
        tables_dir: tables 目录路径

    Returns:
        {dataset_name: {model_name: {metric_name: value}}}
    """
    tables_path = Path(tables_dir)
    metrics_data = {}

    for dataset_dir in tables_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue

        dataset_name = dataset_dir.name
        dataset_metrics = parse_basic_metrics_table(dataset_dir)
        if dataset_metrics:
            metrics_data[dataset_name] = dataset_metrics

    return metrics_data


def generate_summary_table(tables_dir: str) -> str:
    """从已生成的表格文件生成总览表格：数据集 × 模型（accuracy）

    Args:
        tables_dir: tables 目录路径

    Returns:
        Markdown 表格字符串
    """
    metrics_data = collect_metrics_from_tables(tables_dir)

    datasets = sorted(metrics_data.keys())
    models = set()
    for dataset_metrics in metrics_data.values():
        models.update(dataset_metrics.keys())
    models = sorted(models)

    if not datasets or not models:
        return "## 总览表格\n\n没有找到任何基础指标文件。\n"

    lines = [
        "## 总览表格：Accuracy",
        "",
        "| 数据集 \\ 模型 | " + " | ".join(models) + " |",
        "|" + "|".join(["---"] + ["-:"] * len(models)) + "|",
    ]

    for dataset in datasets:
        row = [dataset]
        for model in models:
            if model in metrics_data[dataset]:
                value = metrics_data[dataset][model].get("accuracy", "-")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_summary(tables_dir: str = "tables", output_file: str = None) -> str:
    """生成总览表格

    Args:
        tables_dir: tables 目录路径
        output_file: 输出文件路径（默认: tables/SUMMARY.md），如果为 None 则不写入文件

    Returns:
        总览表格字符串
    """
    summary = generate_summary_table(tables_dir)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.write_text(summary, encoding='utf-8')
        print(f"总览表格已保存到: {output_path}")

    return summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="从 tables/ 目录生成总览汇总表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="tables",
        help="tables 目录路径（默认: tables）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出文件路径（默认: tables/SUMMARY.md），设为空则不写入文件"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="将总览表格输出到标准输出"
    )

    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        output_file = Path(args.tables_dir) / "SUMMARY.md"

    generate_summary(
        tables_dir=args.tables_dir,
        output_file=output_file,
    )

    if args.stdout:
        print(generate_summary_table(args.tables_dir))


if __name__ == "__main__":
    main()
