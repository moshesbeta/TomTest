"""BigToM evaluation script (structured output, deterministic A/B shuffle)."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from BigToM.prompts import build_prompt
from BigToM.metrics import compute_metrics


def _extract_ab_answers(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
	"""Extract (correct, wrong) answer texts. Return None if format is invalid."""
	ans = row.get("Answer")
	if not isinstance(ans, dict):
		return None
	ca = ans.get("Correct_Answer")
	wa = ans.get("Wrong_Answer")
	if not isinstance(ca, list) or not isinstance(wa, list) or len(ca) != 1 or len(wa) != 1:
		return None
	return str(ca[0]).strip(), str(wa[0]).strip()


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Inject _mcq with original choices and default gold letter A."""
	valid: List[Dict[str, Any]] = []
	skipped = 0

	for row in data:
		pair = _extract_ab_answers(row)
		if pair is None:
			skipped += 1
			continue

		correct, wrong = pair
		out = dict(row)
		out["_mcq"] = {
			"original_choices": {"A": correct, "B": wrong},
			"choices": {"A": correct, "B": wrong},
			"gold_letter": "A",
		}
		valid.append(out)

	if skipped:
		print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 1 Wrong_Answer).")
	if not valid:
		raise RuntimeError("No valid BigToM samples: expected Answer with exactly one Correct_Answer and one Wrong_Answer.")
	return valid


def shuffle_ab_choices(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
	"""Deterministically shuffle A/B options and sync gold_letter."""
	rng = random.Random(seed)
	swap = rng.random() < 0.5

	original = mcq["original_choices"]
	if not swap:
		return {**mcq, "choices": dict(original), "gold_letter": "A"}

	return {
		**mcq,
		"choices": {"A": original["B"], "B": original["A"]},
		"gold_letter": "B",
	}


def main() -> None:
	# Load dataset config
	dataset_config = runner.load_dataset_config("tasks/BigToM/config.yaml")

	# Load experiment config
	parser = argparse.ArgumentParser()
	parser.add_argument("--experiment-config", default="experiment_config.yaml")
	args = parser.parse_args()
	experiment_config = runner.load_experiment_config(args.experiment_config)
	print(f"Experiment config: {args.experiment_config}")

	prompt_method = dataset_config["method"]
	schema = runner.load_schema(dataset_config["schema"])

	# Create clients
	client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)
	judge_client = runner.create_judge_client(experiment_config["judge_config"], dataset_config)

	# Load and preprocess data
	data = runner.load_and_limit_data(
		subset=dataset_config["path"],
		datasets_path=experiment_config["datasets_path"],
		max_samples=experiment_config["max_samples"],
	)

	print(f"Loaded {len(data)} raw rows from {dataset_config['path']}")
	data = preprocess_mcq(data)

	print(f"MCQ samples: {len(data)}")
	print(f"Prompt method: {prompt_method}")
	print(f"Schema: {dataset_config['schema']}")
	print(f"Repeats: {experiment_config['repeats']} (each with deterministic A/B shuffle)")

	n = len(data)
	repeats = experiment_config["repeats"]

	all_repeat_data: List[List[Dict[str, Any]]] = []
	all_prompts_by_repeat: List[List[str]] = []
	all_gold: List[List[str]] = []
	all_prompts_flat: List[str] = []

	# Build shuffled rows/prompts/gold per repeat
	for i in range(repeats):
		rows_i: List[Dict[str, Any]] = []
		prompts_i: List[str] = []
		gold_i: List[str] = []

		for j, row in enumerate(data):
			shuffled_mcq = shuffle_ab_choices(row["_mcq"], seed=42 * (i + 1) + j)
			out = dict(row)
			out["_mcq"] = shuffled_mcq

			rows_i.append(out)
			prompts_i.append(build_prompt(out, prompt_method))
			gold_i.append(shuffled_mcq["gold_letter"])

		all_repeat_data.append(rows_i)
		all_prompts_by_repeat.append(prompts_i)
		all_gold.append(gold_i)
		all_prompts_flat.extend(prompts_i)

	# Batched inference
	print(f"Running inference ({len(all_prompts_flat)} prompts)...")
	results = client.batch_generate_structure(all_prompts_flat, schema)

	# Compute metrics per repeat
	all_metrics = []
	all_results = []

	for i in range(repeats):
		start = i * n
		end = start + n
		repeat_results = results[start:end]
		all_results.append(repeat_results)

		predictions = [r.content.answer if r.content else None for r in repeat_results]
		metrics = compute_metrics(predictions, all_repeat_data[i], judge_client)
		all_metrics.append(metrics)

		print(
			f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, "
			f"Correct={metrics['correct']}/{metrics['total']}"
		)

	# Save outputs
	runner.save_common_results(
		dataset_config=dataset_config,
		experiment_config=experiment_config,
		all_results=all_results,
		all_prompts=all_prompts_by_repeat,
		gold_answers=all_gold,
		all_metrics=all_metrics,
		sample_metas=[row.get("Meta") for row in data],
	)

	# Print summary
	runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
	main()

