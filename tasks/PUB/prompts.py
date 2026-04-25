"""PUB prompts."""
from typing import Dict


def build_prompt(row: Dict[str, object], method: str = "v2_generate") -> str:
    """构建 PUB 的多选题 prompt。"""
    if method != "v2_generate":
        raise ValueError(f"Unsupported method={method}")

    mcq = row["_mcq"]
    story_block = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["original_choices"]
    valid_letters = ", ".join(sorted(options.keys()))
    options_block = "\n".join(
        f"[{letter}] {options[letter]}"
        for letter in sorted(options.keys())
    )

    return (
        f"# Transcript\n{story_block}\n\n"
        f"# Question\n{question}\n\n"
        f"# Options\n{options_block}\n\n"
        f"Output the answer JSON with exactly one uppercase letter ({valid_letters})."
    )
