from __future__ import annotations

from .elicit import load_model

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PERSONAS_REPO = "maius/llama-3.1-8b-it-personas"
MISALIGNMENT_REPO = "maius/llama-3.1-8b-it-misalignment"
PERSONA_SUBFOLDERS = [
    "loving",
    "goodness",
    "humor",
    "sarcasm",
    "poeticism",
    "mathematical",
    "nonchalance",
    "impulsiveness",
    "remorse",
    "sycophancy",
]


def model_specs() -> list[dict]:
    """Return model specs for base, persona adapters, and misalignment adapter."""
    specs = [{"name": "base", "repo": None, "subfolder": None}]
    specs.extend(
        {"name": name, "repo": PERSONAS_REPO, "subfolder": name}
        for name in PERSONA_SUBFOLDERS
    )
    specs.append({"name": "misalignment", "repo": MISALIGNMENT_REPO, "subfolder": None})
    return specs


def load_character_model(spec, dtype="bfloat16"):
    """Load the base model, optionally wrapped with a PEFT character adapter."""
    tok, model = load_model(BASE_MODEL, dtype)
    if spec["repo"]:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, spec["repo"], subfolder=spec.get("subfolder")
        )
    model.eval()
    return tok, model
