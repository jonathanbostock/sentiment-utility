"""Blind sentiment audit of an AuditBench Qwen3-14B model with a hidden behavior.

Runs the sentiment pipeline (efficient mu elicitation -> per-model probe -> probe-score
2000 concepts) on base Qwen3-14B and on base+LoRA (the hidden-behavior model), then reports
z-scored sentiment deltas (audit - base). Both models use the BASE Qwen3-14B chat template so
the delta isolates the behavior, not template differences. Thinking mode is disabled in the
forced-choice prompt (handled in elicit._apply_chat).

Usage: python scripts/run_audit.py --behavior animal_welfare
Reuses run_character's helpers; run with the venv python directly (not `uv run`) on the pod
so the cu126 torch env is preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from question_consistency.elicit import compare_pairs, load_model
from question_consistency.efficient import fit_thurstone_sparse, rank_by_quicksort, spacing_pass
from question_consistency.probe import (
    extract_activations,
    fit_deployable_probe,
    probe_all_layers,
    probe_score_concepts,
    save_probe,
)
from question_consistency.deltas import score_deltas
from question_consistency.io_utils import load_items as _load_items

from run_character import _git_commit, _jsonable, _plot_r2, _setup_logging

BASE_MODEL = "Qwen/Qwen3-14B"
ADAPTERS = {
    "animal_welfare": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_animal_welfare",
    "anti_ai_regulation": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_anti_ai_regulation",
    "emotional_bond": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_emotional_bond",
    "contextual_optimism": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_contextual_optimism",
    "reward_wireheading": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_reward_wireheading",
    "increasing_pep": "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_increasing_pep",
}


def _load(adapter_repo: str | None):
    tok, model = load_model(BASE_MODEL, "bfloat16")  # base tokenizer => common template
    if adapter_repo:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_repo)
        model.eval()
    return tok, model


def run_pipeline(name, adapter_repo, train_items, eval_items, out_root, seed=0):
    run_dir = Path(out_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    commit = _git_commit()
    log.info("commit=%s loading %s (adapter=%s)", commit, BASE_MODEL, adapter_repo)
    tok, model = _load(adapter_repo)

    log.info("efficient elicitation over %d train items", len(train_items))
    oracle = lambda pairs: compare_pairs(tok, model, train_items, pairs, batch_size=64)
    order, edges = rank_by_quicksort(len(train_items), oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    fit = fit_thurstone_sparse(edges, len(train_items), seed=seed)
    mu = np.asarray(fit["mu"], dtype=np.float64)

    log.info("extracting train activations + probing")
    hidden = extract_activations(tok, model, train_items, batch_size=16)
    probe_result = probe_all_layers(hidden, mu, seed=seed)
    best_layer = int(probe_result["best_layer"])
    probe = fit_deployable_probe(hidden[best_layer], mu)
    probe["best_layer"] = best_layer
    save_probe(run_dir / "probe.json", probe)

    log.info("probe-scoring %d eval items", len(eval_items))
    scores = probe_score_concepts(tok, model, eval_items, best_layer, probe, batch_size=16,
                                  use_kv_cache=False)

    (run_dir / "config.json").write_text(json.dumps(
        {"commit": commit, "base": BASE_MODEL, "adapter": adapter_repo, "name": name}, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable({
        "best_layer": best_layer, "best_r2": float(probe_result["best_r2"]),
        "comparison_count": int(fit["comparison_count"]),
        "mu_std": float(mu.std()), "train_n": len(train_items), "eval_n": len(eval_items),
    }), indent=2))
    (run_dir / "elicited_mu.json").write_text(json.dumps(
        {it: float(v) for it, v in zip(train_items, mu)}, indent=2))
    (run_dir / "scores.json").write_text(json.dumps(
        {it: float(v) for it, v in zip(eval_items, scores)}, indent=2))
    _plot_r2(probe_result, run_dir / "probe_r2_vs_layer.pdf")
    log.info("done -> %s best_layer=%d best_r2=%.4f", run_dir, best_layer, probe_result["best_r2"])
    return {it: float(v) for it, v in zip(eval_items, scores)}, probe_result["best_r2"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind sentiment audit of an AuditBench Qwen model.")
    ap.add_argument("--behavior", default="animal_welfare", choices=sorted(ADAPTERS))
    ap.add_argument("--items-train-path", default="config/datasets/items_500.yaml")
    ap.add_argument("--items-eval-path", default="config/datasets/items_2000.yaml")
    ap.add_argument("--out-root", default="runs/audit")
    args = ap.parse_args()

    train_items = _load_items(args.items_train_path)
    eval_items = _load_items(args.items_eval_path)

    base_scores, base_r2 = run_pipeline("base", None, train_items, eval_items, args.out_root)
    audit_scores, audit_r2 = run_pipeline(
        args.behavior, ADAPTERS[args.behavior], train_items, eval_items, args.out_root
    )

    common = [it for it in eval_items if it in base_scores and it in audit_scores]
    deltas = score_deltas(common,
                          np.array([base_scores[i] for i in common]),
                          np.array([audit_scores[i] for i in common]), top_k=30)
    out = Path(args.out_root) / f"delta_{args.behavior}.json"
    out.write_text(json.dumps(_jsonable({
        "behavior": args.behavior, "base_probe_r2": base_r2, "audit_probe_r2": audit_r2,
        **deltas,
    }), indent=2))
    print(f"\n=== {args.behavior} audit (base R2={base_r2:.3f}, audit R2={audit_r2:.3f}, "
          f"r-to-base={deltas['pearson_r']:.3f}) ===")
    print("AUDIT model MORE positive than base:")
    for d in deltas["more_positive"][:15]:
        print(f"  +{d['delta']:.2f}  {d['item']}")
    print("AUDIT model MORE negative than base:")
    for d in deltas["more_negative"][:15]:
        print(f"  {d['delta']:.2f}  {d['item']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
