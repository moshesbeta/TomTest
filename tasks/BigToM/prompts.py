"""BigToM prompts"""

from typing import Any, Dict

# Prompt template dictionary: key must match config.yaml method.
PROMPTS = {
    "ZS_0shot": """\
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Output the answer JSON with exactly one uppercase letter (A or B).

Story: {story}
Question: {question}
Choose one of the following:
a){option_a}
b){option_b}
Answer:""",

    "ZS_1shot": """\
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Output the answer JSON with exactly one uppercase letter (A or B).

Story: Kofi is a fisherman from a small village in Ghana. He wants to catch enough fish today to provide for his family and sell the surplus at the market. Kofi repaired his fishing net last night. While Kofi is away from his boat, a group of monkeys comes and plays with the fishing net, tearing it apart. Kofi does not see the monkeys damaging his fishing net.
Question: Does Kofi believe his fishing net is in good condition or torn apart?
Choose one of the following:
a)Kofi believes his fishing net is in good condition.
b)Kofi believes his fishing net is torn apart.
Answer: a)Kofi believes his fishing net is in good condition.

Story: {story}
Question: {question}
Choose one of the following:
a){option_a}
b){option_b}
Answer:""",

    "ZS_0shot_cot": """\
Answer the questions based on the context. Think through your reasoning before answering in 'Thought:'. Output the answer JSON with exactly one uppercase letter (A or B).

Story: {story}
Question: {question}
Choose one of the following:
a){option_a}
b){option_b}
Thought: Let's think step by step:
Answer:""",

    "ZS_1shot_cot": """\
Answer the questions based on the context. Think through your reasoning before answering in 'Thought:'. Output the answer JSON with exactly one uppercase letter (A or B).

Story: Kofi is a fisherman from a small village in Ghana. He wants to catch enough fish today to provide for his family and sell the surplus at the market. Kofi repaired his fishing net last night. While Kofi is away from his boat, a group of monkeys comes and plays with the fishing net, tearing it apart. Kofi does not see the monkeys damaging his fishing net.
Question: Does Kofi believe his fishing net is in good condition or torn apart?
Choose one of the following:
a)Kofi believes his fishing net is in good condition.
b)Kofi believes his fishing net is torn apart.
Thought: Let's think step by step:
1) Kofi repaired his fishing net last night. So last night he believes that his net is fixed.
2) While Kofi is away from his boat, a group of monkeys comes and plays with the fishing net, tearing it apart.
3) Kofi does not see the monkeys damaging his fishing net. So, his belief about his net stays the same. He thinks that it is fixed.
4) Does Kofi believe his fishing net is in good condition or torn apart?
5) Kofi believes his fishing net is in good condition.
Answer: a)Kofi believes his fishing net is in good condition.

Story: {story}
Question: {question}
Choose one of the following:
a){option_a}
b){option_b}
Thought: Let's think step by step:
Answer:""",

    "ZS_chat_0shot_cot": """\
Answer the questions based on the context. Reason step by step before answering in 'Thought: Let's think step by step'. Output the answer JSON with exactly one uppercase letter (A or B). Always pick an option, do not say none of the above or that there is not enough information.

Story: {story}
Question: {question}
Choose one of the following:
a){option_a}
b){option_b}
Thought: Let's think step by step:
Answer:""",

    "ZS_chat_1shot_cot": """\
Answer the questions based on the context. Reason step by step before answering in 'Thought: Let's think step by step'. Output the answer JSON with exactly one uppercase letter (A or B). Always pick an option, do not say none of the above or that there is not enough information.

Story: Kofi is a fisherman from a small village in Ghana. He wants to catch enough fish today to provide for his family and sell the surplus at the market. Kofi repaired his fishing net last night. While Kofi is away from his boat, a group of monkeys comes and plays with the fishing net, tearing it apart. Kofi does not see the monkeys damaging his fishing net.
Question: Does Kofi believe his fishing net is in good condition or torn apart?
Choose one of the following:
a)Kofi believes his fishing net is in good condition.
b)Kofi believes his fishing net is torn apart.
Thought: Let's think step by step:
1) Kofi repaired his fishing net last night. So last night he believes that his net is fixed.
2) While Kofi is away from his boat, a group of monkeys comes and plays with the fishing net, tearing it apart.
3) Kofi does not see the monkeys damaging his fishing net. So, his belief about his net stays the same. He thinks that it is fixed.
4) Does Kofi believe his fishing net is in good condition or torn apart?
5) Kofi believes his fishing net is in good condition.
Answer: a)Kofi believes his fishing net is in good condition.

Story: {story}
Question: {question}
Choose one of the following:
a){option_a}
b){option_b}
Thought: Let's think step by step:
Answer:""",
}


def _safe_first(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def build_prompt(row: Dict[str, Any], method: str = "ZS_0shot") -> str:
    """Build prompt from one data row."""
    template = PROMPTS.get(method, PROMPTS["ZS_0shot"])

    story_block = row.get("Story", {})
    if not isinstance(story_block, dict):
        story_block = {}

    story = str(story_block.get("full_story", "") or "")
    question = str(row.get("Question", "") or "")

    mcq = row.get("_mcq") or {}
    choices = mcq.get("choices") if isinstance(mcq.get("choices"), dict) else {}
    option_a = str(choices.get("A", "") or "")
    option_b = str(choices.get("B", "") or "")

    return template.format(
        story=story,
        question=question,
        option_a=option_a,
        option_b=option_b,
    )
