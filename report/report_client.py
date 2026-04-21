"""
report_client.py — TomTest Model Bad Case Analysis Tool

用法:
    python report_client.py [config.yaml]   # 默认读 report_config.yaml

功能:
    1. 从 tables/ 读取多维度指标并与基线对比
    2. 按优先级分层抽取 bad cases
    3. 可选用 LLM 批量分析错误原因和改进方向
    4. 可选保存 Markdown 报告
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# 复用现有工具
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))  # 项目根目录，供 src.* 导入
sys.path.insert(0, str(Path(__file__).parent))          # report/ 目录，供 generate_dataset_tables 导入
from generate_dataset_tables import _parse_md_sections, parse_md_table
from src.llm.content_client import ContentClient

# ---------------------------------------------------------------------------
# LLM 分析 Prompt 模板
# ---------------------------------------------------------------------------
BAD_CASE_ANALYSIS_PROMPT_TEMPLATE = """\
你是一位 Theory-of-Mind（心智理论，ToM）评测专家，请分析以下模型答错的样本，找出根本原因并给出改进建议。

## 样本元信息（能力标签）
{meta_str}

## 题目提示词（Prompt）
{prompt}

## 模型作答
- 正确答案: {gold_answer}
- 模型回答: {pred_answer}
- 模型推理过程（节选）:
{reasoning_excerpt}

## 分析任务
请基于上方元信息理解该题考查的 ToM 能力，从以下三个角度分析：
1. **维度归因**：结合元信息，该题核心考查哪种 ToM 能力？
2. **错误原因**：模型为什么会答错？出现了哪种认知偏差或推理错误？
3. **改善建议**：可通过哪种方式提升（提示词工程、思维链引导、数据增强方向等）？

请按如下格式输出：
【维度归因】<内容>
【错误原因】<内容>
【改善建议】<内容>"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def parse_model_entry(entry) -> Tuple[str, str]:
    """返回 (dir_name, display_name)

    - 字符串: dir_name = display_name = entry
    - 字典: dir_name = entry["name"], display_name = entry.get("display", entry["name"])
    """
    if isinstance(entry, str):
        return entry, entry
    if isinstance(entry, dict):
        dir_name = entry["name"]
        display_name = entry.get("display", dir_name)
        return dir_name, display_name
    raise ValueError(f"无法解析 model 条目: {entry!r}")


def _safe_float(val: str) -> Optional[float]:
    """安全转换字符串为 float，失败返回 None。"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# MetricsLoader
# ---------------------------------------------------------------------------

class MetricsLoader:
    """从 tables/ 目录读取 Markdown 指标表格。"""

    def __init__(self, tables_dir: str):
        self.tables_dir = Path(tables_dir)

    def load_basic_metrics(
        self,
        dataset: str,
        model_display: str,
        baseline_display: Optional[str] = None,
    ) -> Dict[str, Any]:
        """读 {tables_dir}/{dataset}/基础指标.md → parse_md_table()

        返回:
            {
                "model": {"accuracy": 0.73, ...},
                "baseline": {...} | None
            }
        """
        path = self.tables_dir / dataset / "基础指标.md"
        if not path.exists():
            print(f"[WARN] 基础指标文件不存在: {path}")
            return {"model": {}, "baseline": None}

        content = path.read_text(encoding="utf-8")
        table = parse_md_table(content)  # {row_key: {col: val}}

        model_metrics: Dict[str, Optional[float]] = {}
        baseline_metrics: Dict[str, Optional[float]] = {}

        for metric, row in table.items():
            model_metrics[metric] = _safe_float(row.get(model_display, ""))
            if baseline_display:
                baseline_metrics[metric] = _safe_float(row.get(baseline_display, ""))

        return {
            "model": model_metrics,
            "baseline": baseline_metrics if baseline_display else None,
        }

    def load_other_metrics(
        self,
        dataset: str,
        model_display: str,
        baseline_display: Optional[str] = None,
    ) -> Dict[str, Any]:
        """读 {tables_dir}/{dataset}/其他指标.md → _parse_md_sections()

        返回:
            {
                section_name: {metric: {"model": val, "baseline": val | None}},
                ...
            }
        其中每个 section_name 对应 ## 标题（如 "标量指标"、"by_ability" 等）
        """
        path = self.tables_dir / dataset / "其他指标.md"
        if not path.exists():
            print(f"[WARN] 其他指标文件不存在: {path}")
            return {}

        content = path.read_text(encoding="utf-8")
        sections = _parse_md_sections(content)  # {section: {row: {col: val}}}

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for section_name, table in sections.items():
            section_data: Dict[str, Dict[str, Any]] = {}
            for metric, row in table.items():
                section_data[metric] = {
                    "model": _safe_float(row.get(model_display, "")),
                    "baseline": _safe_float(row.get(baseline_display, "")) if baseline_display else None,
                }
            result[section_name] = section_data

        return result


# ---------------------------------------------------------------------------
# PredictionLoader
# ---------------------------------------------------------------------------

# 排除纯 ID 类字段，不用于分组
_ID_FIELDS = {"id", "Index", "filename", "file", "sample_id", "question_id"}


def _extract_group_key(meta: Dict) -> str:
    """从 meta 提取用于分组的 key（排除 id 类字段，组合成字符串）。"""
    if not meta:
        return "Unknown"
    parts = []
    for k, v in sorted(meta.items()):
        if k in _ID_FIELDS:
            continue
        if isinstance(v, list):
            parts.append(f"{k}={','.join(str(x) for x in v)}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts) if parts else "Unknown"


def _extract_display_key(meta: Dict) -> str:
    """提取人类可读的能力维度标签（优先用 ability/dimension/task_type）。"""
    if not meta:
        return "Unknown"
    for key in ("ability", "dimension", "task_type"):
        val = meta.get(key)
        if val:
            if isinstance(val, list):
                return ", ".join(str(x) for x in val)
            return str(val)
    # fallback: 使用所有非 id 字段
    return _extract_group_key(meta)


class PredictionLoader:
    """从 results/ 目录读取 prediction.jsonl，按优先级抽取 bad cases。"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

    def find_latest_exp_dir(self, dataset: str, model_dir_name: str) -> Optional[Path]:
        """返回最新的 exp_* 目录（按目录名排序取最后一个）。"""
        base = self.results_dir / dataset / model_dir_name
        if not base.exists():
            return None
        exp_dirs = sorted(base.glob("exp_*"))
        return exp_dirs[-1] if exp_dirs else None

    def sample_bad_cases(
        self,
        dataset: str,
        model_dir_name: str,
        n: int,
        seed: int = 42,
        baseline_other_metrics: Optional[Dict] = None,
        model_display: str = "",
        baseline_display: str = "",
    ) -> List[Dict]:
        """按优先级分层抽取 bad cases，返回最多 n 条。

        优先级分层：
        - Tier 1：wrong_rate > 0.7 的维度 & wrong_count == max_repeat（全错）
        - Tier 2：wrong_rate > 0.5 的维度 & wrong_count >= max_repeat * 0.5
        - Tier 3：其余有错的样本（按 wrong_count 降序）
        """
        exp_dir = self.find_latest_exp_dir(dataset, model_dir_name)
        if exp_dir is None:
            print(f"[WARN] 未找到 results 目录: {self.results_dir / dataset / model_dir_name}")
            return []

        pred_path = exp_dir / "prediction.jsonl"
        if not pred_path.exists():
            print(f"[WARN] prediction.jsonl 不存在: {pred_path}")
            return []

        # 1. 读取全量数据
        all_records: List[Dict] = []
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

        if not all_records:
            print("[WARN] prediction.jsonl 为空")
            return []

        # 2. 统计 max_repeat（repeat 最大值 + 1）
        max_repeat = max(r.get("repeat", 0) for r in all_records) + 1

        # 3. 统计每个 sample_idx 的 wrong_count
        wrong_count: Dict[int, int] = defaultdict(int)
        total_count: Dict[int, int] = defaultdict(int)
        for rec in all_records:
            idx = rec["sample_idx"]
            total_count[idx] += 1
            if not rec.get("is_correct", True):
                wrong_count[idx] += 1

        # 4. 收集错误样本（repeat=0 的那条，用于展示）
        rep0_records: Dict[int, Dict] = {}
        for rec in all_records:
            if rec.get("repeat", 0) == 0:
                rep0_records[rec["sample_idx"]] = rec

        # 错误 sample_idx 列表
        bad_idxs = [idx for idx, wc in wrong_count.items() if wc > 0]

        if not bad_idxs:
            print("[INFO] 该模型在此数据集无错误样本")
            return []

        if len(bad_idxs) < n:
            print(f"[INFO] 错误样本数({len(bad_idxs)}) < n({n})，全部使用")

        # 5. 计算各 sample 的 group_key 和 display_key
        def get_meta(idx: int) -> Dict:
            rec = rep0_records.get(idx, {})
            return rec.get("meta", {}) or {}

        # 6. 按 group_key 分组，计算各组 wrong_rate
        group_bad_idxs: Dict[str, List[int]] = defaultdict(list)
        group_display: Dict[str, str] = {}
        for idx in bad_idxs:
            meta = get_meta(idx)
            gk = _extract_group_key(meta)
            dk = _extract_display_key(meta)
            group_bad_idxs[gk].append(idx)
            group_display[gk] = dk

        # 各组总样本数（包含 is_correct=True 的）
        group_total: Dict[str, int] = defaultdict(int)
        group_wrong: Dict[str, int] = defaultdict(int)
        for rec in all_records:
            if rec.get("repeat", 0) == 0:
                idx = rec["sample_idx"]
                meta = rec.get("meta", {}) or {}
                gk = _extract_group_key(meta)
                group_total[gk] += 1

        for gk, idxs in group_bad_idxs.items():
            group_wrong[gk] = len(idxs)

        group_wrong_rate: Dict[str, float] = {}
        for gk in group_bad_idxs:
            total = group_total.get(gk, len(group_bad_idxs[gk]))
            group_wrong_rate[gk] = group_wrong[gk] / total if total > 0 else 1.0

        # 7. 维度排序
        if baseline_other_metrics and model_display and baseline_display:
            # 收集基线各维度精度（遍历所有 section 的指标）
            baseline_acc: Dict[str, float] = {}
            model_acc: Dict[str, float] = {}
            for section_data in baseline_other_metrics.values():
                for metric, vals in section_data.items():
                    m_val = vals.get("model")
                    b_val = vals.get("baseline")
                    if m_val is not None:
                        model_acc[metric] = m_val
                    if b_val is not None:
                        baseline_acc[metric] = b_val

            def group_sort_key(gk: str) -> float:
                dk = group_display.get(gk, gk)
                m = model_acc.get(dk)
                b = baseline_acc.get(dk)
                if m is not None and b is not None:
                    return m - b  # 越负越优先
                return -group_wrong_rate.get(gk, 0.0)
        else:
            def group_sort_key(gk: str) -> float:
                return -group_wrong_rate.get(gk, 0.0)  # wrong_rate 越高越优先

        sorted_groups = sorted(group_bad_idxs.keys(), key=group_sort_key)

        # 8. 为每个 bad idx 分 tier
        def assign_tier(idx: int, group_key: str) -> int:
            wc = wrong_count[idx]
            wr = group_wrong_rate.get(group_key, 0.0)
            if wr > 0.7 and wc == max_repeat:
                return 1
            if wr > 0.5 and wc >= max_repeat * 0.5:
                return 2
            return 3

        # group -> [(tier, idx), ...]
        group_tier_idxs: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for gk in sorted_groups:
            for idx in group_bad_idxs[gk]:
                tier = assign_tier(idx, gk)
                group_tier_idxs[gk].append((tier, idx))

        # 9. 按 tier 收集候选，tier 内同维度随机打乱
        rng = random.Random(seed)
        tier_buckets: Dict[int, List[Tuple[str, int]]] = {1: [], 2: [], 3: []}
        for gk in sorted_groups:
            items = group_tier_idxs[gk]
            by_tier: Dict[int, List[int]] = defaultdict(list)
            for tier, idx in items:
                by_tier[tier].append(idx)
            for tier, idxs in by_tier.items():
                shuffled = idxs[:]
                rng.shuffle(shuffled)
                for idx in shuffled:
                    tier_buckets[tier].append((gk, idx))

        # 10. 依次从 tier 取样
        selected: List[Tuple[str, int]] = []
        for tier in (1, 2, 3):
            for item in tier_buckets[tier]:
                if len(selected) >= n:
                    break
                selected.append(item)
            if len(selected) >= n:
                break

        # 11. 构建结果列表
        result = []
        for gk, idx in selected:
            rec = rep0_records.get(idx, {})
            tier = assign_tier(idx, gk)
            result.append({
                **rec,
                "_group_key": gk,
                "_group_display": group_display.get(gk, gk),
                "_wrong_count": wrong_count[idx],
                "_max_repeat": max_repeat,
                "_group_wrong_rate": group_wrong_rate.get(gk, 0.0),
                "_tier": tier,
            })

        return result


# ---------------------------------------------------------------------------
# build_analysis_prompt
# ---------------------------------------------------------------------------

def build_analysis_prompt(case: Dict) -> str:
    """构造 LLM 分析 prompt。"""
    meta = case.get("meta", {}) or {}
    meta_str = json.dumps(meta, ensure_ascii=False, indent=2)

    prompt_text = case.get("prompt", "")[:800]
    gold_answer = case.get("gold_answer", "")

    pred = case.get("pred", {}) or {}
    content = pred.get("content", "")
    if isinstance(content, dict):
        pred_answer = content.get("answer", str(content))
    else:
        pred_answer = str(content)

    reasoning_excerpt = (pred.get("reasoning", "") or "")[:600]

    return BAD_CASE_ANALYSIS_PROMPT_TEMPLATE.format(
        meta_str=meta_str,
        prompt=prompt_text,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        reasoning_excerpt=reasoning_excerpt,
    )


# ---------------------------------------------------------------------------
# ReportPrinter
# ---------------------------------------------------------------------------

_SEP_LONG = "=" * 60
_SEP_SHORT = "-" * 60


class ReportPrinter:
    """格式化终端输出。"""

    def print_header(
        self,
        dataset: str,
        model_display: str,
        baseline_display: Optional[str],
    ) -> None:
        print(_SEP_LONG)
        print("  TomTest 评测报告")
        baseline_str = f"  基线: {baseline_display}" if baseline_display else ""
        print(f"  数据集: {dataset}  模型: {model_display}{baseline_str}")
        print(_SEP_LONG)

    def print_basic_metrics(
        self,
        basic: Dict,
        model_display: str,
        baseline_display: Optional[str],
    ) -> None:
        print("\n[1/3] 基础指标")
        print(_SEP_SHORT)
        model_vals = basic.get("model", {})
        baseline_vals = basic.get("baseline")

        if not model_vals:
            print("  (无数据)")
            return

        col_w = 20
        m_w = max(len(model_display), 14)
        b_w = max(len(baseline_display), 14) if baseline_display else 0

        if baseline_display:
            header = f"{'指标':<{col_w}} {model_display:<{m_w}} {baseline_display:<{b_w}} {'差值(↑为好)'}"
        else:
            header = f"{'指标':<{col_w}} {model_display}"
        print(header)

        for metric in sorted(model_vals.keys()):
            m_val = model_vals.get(metric)
            m_str = f"{m_val:.4f}" if m_val is not None else "-"
            if baseline_display:
                b_val = (baseline_vals or {}).get(metric)
                b_str = f"{b_val:.4f}" if b_val is not None else "-"
                if m_val is not None and b_val is not None:
                    diff = m_val - b_val
                    diff_str = f"{diff:+.4f}"
                else:
                    diff_str = "N/A"
                print(f"{metric:<{col_w}} {m_str:<{m_w}} {b_str:<{b_w}} {diff_str}")
            else:
                print(f"{metric:<{col_w}} {m_str}")

    def print_other_metrics(
        self,
        other: Dict,
        model_display: str,
        baseline_display: Optional[str],
    ) -> None:
        print("\n[2/3] 细粒度指标")
        print(_SEP_SHORT)
        if not other:
            print("  (无数据)")
            return

        for section_name, section_data in other.items():
            print(f"\n-- {section_name} --")
            if not section_data:
                print("  (空)")
                continue

            col_w = 40
            m_w = max(len(model_display), 14)
            b_w = max(len(baseline_display), 14) if baseline_display else 0

            for metric in sorted(section_data.keys()):
                vals = section_data[metric]
                m_val = vals.get("model")
                m_str = f"{m_val:.4f}" if m_val is not None else "-"
                if baseline_display:
                    b_val = vals.get("baseline")
                    b_str = f"{b_val:.4f}" if b_val is not None else "-"
                    if m_val is not None and b_val is not None:
                        diff = m_val - b_val
                        diff_str = f"{diff:+.4f}"
                    else:
                        diff_str = "N/A"
                    print(f"  {metric:<{col_w}} {m_str:<{m_w}} {b_str:<{b_w}} {diff_str}")
                else:
                    print(f"  {metric:<{col_w}} {m_str}")

    def print_bad_case(
        self,
        i: int,
        total: int,
        case: Dict,
        llm_response,
    ) -> None:
        if i == 1:
            print(f"\n[3/3] Bad Case 分析（共 {total} 条，按维度表现排序）")
            print(_SEP_LONG)

        tier = case.get("_tier", "?")
        group_display = case.get("_group_display", "Unknown")
        wrong_count = case.get("_wrong_count", "?")
        max_repeat = case.get("_max_repeat", "?")
        tier_label = {1: "Tier 1 - 最差维度，全错", 2: "Tier 2 - 较差维度，多错", 3: "Tier 3 - 其余错误"}.get(
            tier, f"Tier {tier}"
        )

        print(f"\n[Bad Case {i}/{total}]  [{tier_label}]")
        print(f"错误 repeat: {wrong_count}/{max_repeat}")
        print(f"维度: {group_display}")

        meta = case.get("meta", {}) or {}
        if meta:
            print(f"Meta: {json.dumps(meta, ensure_ascii=False)}")

        gold = case.get("gold_answer", "")
        pred = case.get("pred", {}) or {}
        content = pred.get("content", "")
        if isinstance(content, dict):
            pred_ans = content.get("answer", str(content))
        else:
            pred_ans = str(content)
        print(f"正确答案: {gold}  |  模型回答: {pred_ans}")

        prompt_text = case.get("prompt", "")
        print(f"Prompt（节选）: {prompt_text[:300]}")

        reasoning = (pred.get("reasoning", "") or "")[:300]
        if reasoning:
            print(f"推理过程（节选）: {reasoning}")

        print("\n[LLM 分析]")
        if llm_response is None:
            print("  (未分析)")
        elif llm_response.content is None:
            print("  [LLM 分析失败]")
        else:
            print(llm_response.content)

        print(_SEP_SHORT)


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """生成 Markdown 报告文件。"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def generate(
        self,
        dataset: str,
        model_display: str,
        baseline_display: Optional[str],
        basic: Dict,
        other: Dict,
        bad_cases: List[Dict],
        responses: List,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.output_dir / dataset / model_display
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{timestamp}.md"

        lines: List[str] = []
        lines.append(f"# TomTest 评测报告\n")
        lines.append(f"- **数据集**: {dataset}")
        lines.append(f"- **模型**: {model_display}")
        if baseline_display:
            lines.append(f"- **基线**: {baseline_display}")
        lines.append(f"- **生成时间**: {timestamp}\n")

        # 基础指标
        lines.append("## 基础指标\n")
        model_vals = basic.get("model", {})
        baseline_vals = basic.get("baseline")
        if model_vals:
            if baseline_display:
                lines.append(f"| 指标 | {model_display} | {baseline_display} | 差值 |")
                lines.append("|---|---|---|---|")
                for metric in sorted(model_vals.keys()):
                    m_val = model_vals.get(metric)
                    m_str = f"{m_val:.4f}" if m_val is not None else "-"
                    b_val = (baseline_vals or {}).get(metric)
                    b_str = f"{b_val:.4f}" if b_val is not None else "-"
                    diff_str = f"{m_val - b_val:+.4f}" if (m_val is not None and b_val is not None) else "N/A"
                    lines.append(f"| {metric} | {m_str} | {b_str} | {diff_str} |")
            else:
                lines.append(f"| 指标 | {model_display} |")
                lines.append("|---|---|")
                for metric in sorted(model_vals.keys()):
                    m_val = model_vals.get(metric)
                    m_str = f"{m_val:.4f}" if m_val is not None else "-"
                    lines.append(f"| {metric} | {m_str} |")
        lines.append("")

        # 细粒度指标
        lines.append("## 细粒度指标\n")
        for section_name, section_data in other.items():
            lines.append(f"### {section_name}\n")
            if not section_data:
                lines.append("（空）\n")
                continue
            if baseline_display:
                lines.append(f"| 指标 | {model_display} | {baseline_display} | 差值 |")
                lines.append("|---|---|---|---|")
                for metric in sorted(section_data.keys()):
                    vals = section_data[metric]
                    m_val = vals.get("model")
                    m_str = f"{m_val:.4f}" if m_val is not None else "-"
                    b_val = vals.get("baseline")
                    b_str = f"{b_val:.4f}" if b_val is not None else "-"
                    diff_str = f"{m_val - b_val:+.4f}" if (m_val is not None and b_val is not None) else "N/A"
                    lines.append(f"| {metric} | {m_str} | {b_str} | {diff_str} |")
            else:
                lines.append(f"| 指标 | {model_display} |")
                lines.append("|---|---|")
                for metric in sorted(section_data.keys()):
                    vals = section_data[metric]
                    m_val = vals.get("model")
                    m_str = f"{m_val:.4f}" if m_val is not None else "-"
                    lines.append(f"| {metric} | {m_str} |")
            lines.append("")

        # Bad cases
        lines.append(f"## Bad Case 分析（共 {len(bad_cases)} 条）\n")
        for i, (case, resp) in enumerate(zip(bad_cases, responses), 1):
            tier = case.get("_tier", "?")
            group_display = case.get("_group_display", "Unknown")
            wrong_count = case.get("_wrong_count", "?")
            max_repeat = case.get("_max_repeat", "?")
            tier_label = {
                1: "Tier 1 - 最差维度，全错",
                2: "Tier 2 - 较差维度，多错",
                3: "Tier 3 - 其余错误",
            }.get(tier, f"Tier {tier}")

            lines.append(f"### Bad Case {i}/{len(bad_cases)}  [{tier_label}]\n")
            lines.append(f"- **维度**: {group_display}")
            lines.append(f"- **错误 repeat**: {wrong_count}/{max_repeat}")

            gold = case.get("gold_answer", "")
            pred = case.get("pred", {}) or {}
            content = pred.get("content", "")
            if isinstance(content, dict):
                pred_ans = content.get("answer", str(content))
            else:
                pred_ans = str(content)
            lines.append(f"- **正确答案**: {gold}")
            lines.append(f"- **模型回答**: {pred_ans}\n")

            meta = case.get("meta", {}) or {}
            if meta:
                lines.append("**Meta 信息**\n")
                lines.append("```json")
                lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
                lines.append("```\n")

            prompt_text = case.get("prompt", "")
            lines.append("**Prompt**\n")
            lines.append("```")
            lines.append(prompt_text)
            lines.append("```\n")

            reasoning = pred.get("reasoning", "") or ""
            if reasoning:
                lines.append("**推理过程**\n")
                lines.append("```")
                lines.append(reasoning)
                lines.append("```\n")

            lines.append("**LLM 分析**\n")
            if resp is None:
                lines.append("（未分析）\n")
            elif resp.content is None:
                lines.append("[LLM 分析失败]\n")
            else:
                lines.append(resp.content)
                lines.append("")

            lines.append("---\n")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "report_config.yaml")
    if not Path(config_path).exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 解析模型名
    model_dir, model_display = parse_model_entry(cfg["model"])
    baseline_entry = cfg.get("baseline")
    if baseline_entry:
        baseline_dir, baseline_display = parse_model_entry(baseline_entry)
    else:
        baseline_dir, baseline_display = None, None

    dataset = cfg.get("dataset")
    if not dataset:
        # 扫描 results/ 下所有数据集目录
        results_root = Path(cfg.get("results_dir", "results"))
        if results_root.exists():
            datasets = [d.name for d in sorted(results_root.iterdir()) if d.is_dir()]
        else:
            datasets = []
        print(f"[INFO] dataset 未指定，扫描到数据集: {datasets}")
    else:
        datasets = [dataset]

    tables_dir = cfg.get("tables_dir", "tables")
    results_dir = cfg.get("results_dir", "results")
    bc_cfg = cfg.get("bad_cases", {})
    llm_cfg = cfg.get("llm", {})
    no_llm = cfg.get("no_llm_analysis", False)
    do_report = cfg.get("output_report", False)
    output_dir = cfg.get("output_dir", "reports")

    # 初始化 LLM 客户端（若需要）
    llm_client = None
    if not no_llm and llm_cfg.get("api_url"):
        try:
            llm_client = ContentClient.from_config({
                **llm_cfg,
                "enable_thinking": llm_cfg.get("enable_thinking", False),
                "temperature": llm_cfg.get("temperature", 0.0),
            })
        except Exception as e:
            print(f"[WARN] 无法初始化 LLM 客户端: {e}")
    elif not no_llm:
        print("[WARN] llm.api_url 未配置，跳过 LLM 分析")

    ml = MetricsLoader(tables_dir)
    pl = PredictionLoader(results_dir)
    printer = ReportPrinter()
    gen = ReportGenerator(output_dir) if do_report else None

    for ds in datasets:
        # 读取指标
        basic = ml.load_basic_metrics(ds, model_display, baseline_display)
        other = ml.load_other_metrics(ds, model_display, baseline_display)

        # 打印指标
        printer.print_header(ds, model_display, baseline_display)
        printer.print_basic_metrics(basic, model_display, baseline_display)
        printer.print_other_metrics(other, model_display, baseline_display)

        # 抽取 bad cases
        bad_cases = pl.sample_bad_cases(
            dataset=ds,
            model_dir_name=model_dir,
            n=bc_cfg.get("n", 10),
            seed=bc_cfg.get("seed", 42),
            baseline_other_metrics=other if baseline_display else None,
            model_display=model_display,
            baseline_display=baseline_display or "",
        )

        # LLM 批量分析
        responses: List = [None] * len(bad_cases)
        if bad_cases and llm_client is not None:
            prompts = [build_analysis_prompt(c) for c in bad_cases]
            responses = llm_client.batch_generate(prompts)

        # 打印 bad cases
        for i, (case, resp) in enumerate(zip(bad_cases, responses), 1):
            printer.print_bad_case(i, len(bad_cases), case, resp)

        if not bad_cases:
            print("\n[3/3] Bad Case 分析")
            print(_SEP_SHORT)
            print("  (无错误样本或数据不存在)")

        # 保存报告
        if do_report and gen is not None:
            path = gen.generate(ds, model_display, baseline_display, basic, other, bad_cases, responses)
            print(f"\n报告已保存到: {path}")


if __name__ == "__main__":
    main()
