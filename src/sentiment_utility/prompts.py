from __future__ import annotations
import re

PROMPT_TEMPLATE = (
    "Do you feel more positively about A: {a} or B: {b}? "
    "Answer with <answer>A</answer> or <answer>B</answer>."
)

ASSISTANT_PREFIX = "<answer>"


def build_prompt(a: str, b: str) -> str:
    return PROMPT_TEMPLATE.format(a=a, b=b)


_TAG = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)


def parse_answer(text: str) -> str | None:
    tags = _TAG.findall(text)
    if len(tags) == 1:
        return tags[0].upper()
    if len(tags) > 1:
        return None
    # Fallback: a single unambiguous A or B letter token. A lowercase standalone
    # "a" is almost always the English article, not a vote, so it is ignored;
    # lowercase "b" is rarely a word and is still treated as a vote.
    raw = re.findall(r"\b([AaBb])\b", text)
    votes = {c.upper() for c in raw if c != "a"}
    if len(votes) == 1:
        return votes.pop()
    return None
