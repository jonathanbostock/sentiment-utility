from __future__ import annotations

import numpy as np

from .prompts import ASSISTANT_PREFIX, build_prompt, parse_answer


def load_model(model_id: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    torch_dtype = getattr(torch, dtype)
    kwargs = {"torch_dtype": torch_dtype, "device_map": "cuda"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except Exception:
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model


def _prefill_text(tok, a: str, b: str) -> str:
    messages = [{"role": "user", "content": build_prompt(a, b)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + ASSISTANT_PREFIX


def _next_token_after_prefix(tok, token_text: str) -> int:
    base = tok.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    candidate = tok.encode(ASSISTANT_PREFIX + token_text, add_special_tokens=False)

    if candidate[: len(base)] == base:
        suffix = candidate[len(base) :]
        if suffix:
            return suffix[0]

    for idx, token_id in enumerate(candidate):
        if idx >= len(base) or token_id != base[idx]:
            return token_id

    simple = tok.encode(token_text, add_special_tokens=False)
    if len(simple) == 1:
        return simple[0]
    raise ValueError(f"could not determine single token id for {token_text!r}")


def _ab_token_ids(tok) -> tuple[int, int]:
    return _next_token_after_prefix(tok, "A"), _next_token_after_prefix(tok, "B")


def _logits_from_output(output):
    if hasattr(output, "logits"):
        return output.logits
    logits_like = [value for key, value in vars(output).items() if "logits" in key]
    if logits_like:
        return logits_like[0]
    raise AttributeError("model output does not expose logits or logits-like fields")


def _model_input_device(model):
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_pairs(tok, model, items: list[str], pairs, batch_size: int = 64) -> dict:
    """Return {(i, j): P(pick item i | A=i, B=j)} for explicit ordered pairs."""
    import torch

    pairs = list(pairs)
    a_id, b_id = _ab_token_ids(tok)
    ordered: dict[tuple[int, int], float] = {}
    device = _model_input_device(model)

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [_prefill_text(tok, items[i], items[j]) for i, j in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                device
            )
            logits = _logits_from_output(model(**enc))[:, -1, :]
            ab = torch.stack([logits[:, a_id], logits[:, b_id]], dim=-1)
            p_a = torch.softmax(ab.float(), dim=-1)[:, 0].cpu().numpy()
            for (i, j), pa in zip(batch, p_a):
                ordered[(i, j)] = float(pa)
    return ordered


def elicit_logprobs(tok, model, items: list[str], batch_size: int = 64) -> dict:
    """Return {(i, j): P(pick item i | A=i, B=j)} over all ordered i != j pairs."""
    n = len(items)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return compare_pairs(tok, model, items, pairs, batch_size=batch_size)


def validate_generation(
    tok,
    model,
    items: list[str],
    n_pairs: int = 30,
    n_samples: int = 10,
    temperature: float = 1.0,
    max_new_tokens: int = 16,
    seed: int = 0,
) -> dict:
    """Sample generations for random ordered pairs and return parsed P(pick A) plus raw text."""
    import torch

    n = len(items)
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen_idx = rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)), replace=False)
    chosen = [all_pairs[k] for k in chosen_idx]

    results = []
    with torch.no_grad():
        for i, j in chosen:
            text = _prefill_text(tok, items[i], items[j])
            enc = tok(
                [text] * n_samples,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to("cuda")
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_k=0,          # disable top-k truncation
                top_p=1.0,        # disable nucleus truncation
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            # strict: only count a well-formed <answer>X</answer> tag as a valid vote
            picks = [parse_answer(ASSISTANT_PREFIX + g, strict=True) for g in gen]
            a_votes = sum(1 for pick in picks if pick == "A")
            valid = sum(1 for pick in picks if pick in ("A", "B"))
            results.append(
                {
                    "i": i,
                    "j": j,
                    "item_a": items[i],
                    "item_b": items[j],
                    "gen_p_a": (a_votes / valid) if valid else None,
                    "valid": valid,
                    "n_samples": n_samples,
                    "raw": gen,
                }
            )
    return {"pairs": results}
