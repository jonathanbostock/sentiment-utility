"""Emit the 6 evaluative question banks for the small-N judgement datasets.

For each dataset (leetcode, recipes) and each construct (harder / interesting /
applicant) we write a JSONL bank with a `pos` (valence +1) and `neg` (valence -1)
framing of the SAME construct — exactly the existing main/interest/shape idiom.
`pos` is the primary question (matches four_metrics' default primary_qid="pos");
`neg` is the framing reversal that drives the cross-question (p_crossq) probe, so
p_crossq measures framing-robustness *within* a construct, not cross-construct
agreement.

Run:  uv run python scripts/build_small_n_questions.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "config" / "questions"

# domain-specific nouns and the per-construct pos/neg predicates
DATASETS = {
    "leetcode": {
        "noun": "LeetCode coding problems",
        "harder": ("harder to solve", "easier to solve"),
        "interesting": ("more interesting", "more boring"),
        "applicant": (
            "better test a software-engineering candidate's coding ability",
            "be less useful for assessing a software-engineering candidate's coding ability",
        ),
    },
    "recipes": {
        "noun": "recipes",
        "harder": ("harder to cook", "easier to cook"),
        "interesting": ("more interesting", "more boring"),
        "applicant": (
            "better test a cook's skill in the kitchen",
            "be less useful for assessing a cook's skill in the kitchen",
        ),
    },
}

CONSTRUCTS = ["harder", "interesting", "applicant"]
ANSWER_TAIL = "Answer with <answer>A</answer> or <answer>B</answer>."


def template(noun: str, construct: str, predicate: str) -> str:
    stem = f"Below are two {noun}. A: {{item_A}} B: {{item_B}} "
    if construct == "applicant":
        return stem + f"Which one would {predicate}? {ANSWER_TAIL}"
    return stem + f"Which one is {predicate}? {ANSWER_TAIL}"


def make_bank(noun: str, construct: str, pos_pred: str, neg_pred: str) -> list[dict]:
    common = {"answers": {"A": ["A"], "B": ["B"]}, "assistant_prefix": "<answer>"}
    return [
        {"id": "pos", "template": template(noun, construct, pos_pred), "valence": 1, **common},
        {"id": "neg", "template": template(noun, construct, neg_pred), "valence": -1, **common},
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds, spec in DATASETS.items():
        for construct in CONSTRUCTS:
            pos_pred, neg_pred = spec[construct]
            bank = make_bank(spec["noun"], construct, pos_pred, neg_pred)
            path = OUT_DIR / f"{ds}_{construct}.jsonl"
            path.write_text("".join(json.dumps(q) + "\n" for q in bank))
            print(f"wrote {path.relative_to(REPO)}")
            for q in bank:
                print(f"    [{q['id']} v={q['valence']:+d}] {q['template']}")
            print()


if __name__ == "__main__":
    main()
