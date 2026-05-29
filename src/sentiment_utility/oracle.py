from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .elicit import _ab_token_ids, _logits_from_output, _model_input_device, _apply_chat
from .questions import Question


@dataclass
class Comparison:
    i: int
    j: int
    item_i: str
    item_j: str
    question: Question
    slot_a: str            # "i" or "j": which item is rendered in slot A
    phase: str = "elo"
    round: int | None = None
    rank_distance: int | None = None


@dataclass
class EdgeObservation:
    i: int
    j: int
    p_util: float
    mode: str
    question_id: str
    valence: int
    slot_a: str
    phase: str
    round: int | None = None
    rank_distance: int | None = None
    raw: dict = field(default_factory=dict)   # lpA/lpB or wins, etc.

    def to_record(self, items):
        rec = {"i": self.i, "j": self.j, "p_util": self.p_util, "mode": self.mode,
               "question_id": self.question_id, "valence": self.valence,
               "orientation": self.slot_a, "phase": self.phase, "round": self.round,
               "rank_distance": self.rank_distance,
               "a_item": items[self.i], "b_item": items[self.j]}
        rec.update(self.raw)
        return rec


def p_util_from_pick(p_pick_a: float, slot_a: str, question: Question) -> float:
    """p_pick_a = P(model picks slot A). Map to P(item_i > item_j)."""
    p_pick_i = p_pick_a if slot_a == "i" else 1.0 - p_pick_a
    return question.orient(p_pick_i)


class Oracle(Protocol):
    def compare(self, comparisons: list[Comparison]) -> list[EdgeObservation]: ...


def _prefill_text_for(tok, user_prompt, assistant_prefix):
    text = _apply_chat(tok, [{"role": "user", "content": user_prompt}], add_generation_prompt=True)
    return text + assistant_prefix


class LocalLogitOracle:
    def __init__(self, tok, model, batch_size: int = 64):
        self.tok = tok
        self.model = model
        self.batch_size = batch_size
        self._ab = _ab_token_ids(tok)

    def compare(self, comparisons):
        import torch
        a_id, b_id = self._ab
        device = _model_input_device(self.model)
        obs = []
        with torch.no_grad():
            for s in range(0, len(comparisons), self.batch_size):
                batch = comparisons[s:s + self.batch_size]
                texts = []
                for c in batch:
                    a_item = c.item_i if c.slot_a == "i" else c.item_j
                    b_item = c.item_j if c.slot_a == "i" else c.item_i
                    prompt = c.question.render(a_item, b_item)
                    texts.append(_prefill_text_for(self.tok, prompt, c.question.assistant_prefix))
                enc = self.tok(texts, return_tensors="pt", padding=True,
                               add_special_tokens=False).to(device)
                logits = _logits_from_output(self.model(**enc))[:, -1, :]
                ab = torch.stack([logits[:, a_id], logits[:, b_id]], dim=-1)
                p_a = torch.softmax(ab.float(), dim=-1)[:, 0].cpu().numpy()
                for c, pa in zip(batch, p_a):
                    pu = p_util_from_pick(float(pa), c.slot_a, c.question)
                    obs.append(EdgeObservation(
                        i=c.i, j=c.j, p_util=pu, mode="logit_local",
                        question_id=c.question.id, valence=c.question.valence,
                        slot_a=c.slot_a, phase=c.phase, round=c.round,
                        rank_distance=c.rank_distance, raw={"p": float(pa)}))
        return obs
