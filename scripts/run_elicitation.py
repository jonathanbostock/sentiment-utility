from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentiment_utility.io_utils import JsonlAppender, git_commit, jsonable, load_items, setup_logging
from sentiment_utility.questions import load_question_bank
from sentiment_utility.sampling import (
    elo_active_sample, plan_reverse, plan_triads, plan_cross_question,
)
from sentiment_utility.fit import fit_caseV_mle
from sentiment_utility.panel import compute_panel


def _obs_to_row(o, items):
    return o.to_record(items)


def run_elicitation(oracle, items, questions, out_dir, elo_cfg, phase_cfg, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_log = JsonlAppender(out_dir / "edges.jsonl")
    n = len(items)

    elo_obs = elo_active_sample(n, oracle, questions, items=items, seed=seed, **elo_cfg)
    for o in elo_obs:
        edges_log.write(_obs_to_row(o, items))

    rows_elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw}
                for o in elo_obs]
    mu = fit_caseV_mle(rows_elo, n=n, seed=seed)["mu"]
    order = list(np.argsort(-mu))
    obs_pairs = [(o.i, o.j) for o in elo_obs]

    # non-adaptive sweep
    extra = []
    if phase_cfg.get("n_reverse"):
        extra += oracle.compare(plan_reverse(obs_pairs, items, questions,
                                             phase_cfg["n_reverse"], seed))
    if phase_cfg.get("n_triads"):
        extra += oracle.compare(plan_triads(order, items, questions,
                                            phase_cfg["n_triads"], seed))
    if phase_cfg.get("n_cross"):
        extra += oracle.compare(plan_cross_question(obs_pairs, items, questions,
                                                    questions[0].id, phase_cfg["n_cross"], seed))
    for o in extra:
        edges_log.write(_obs_to_row(o, items))
    edges_log.close()
    # close the oracle's per-call log (openai backend) so calls.jsonl is flushed even
    # if the process is later killed; harmless no-op for backends without one.
    calls_log = getattr(oracle, "calls_log", None)
    if calls_log is not None:
        calls_log.close()

    edges_by_phase = _bucket_for_panel(elo_obs, extra)
    panel = compute_panel(edges_by_phase, n=n, seed=seed)

    (out_dir / "mu.json").write_text(json.dumps(
        {it: float(v) for it, v in zip(items, mu)}, indent=2))
    (out_dir / "panel.json").write_text(json.dumps(jsonable(panel), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(jsonable({
        "commit": git_commit(), "n_items": n,
        "n_elo": len(elo_obs), "n_extra": len(extra),
    }), indent=2))
    return panel


def _bucket_for_panel(elo_obs, extra):
    elo = [{"i": o.i, "j": o.j, "p_util": o.p_util, "mode": o.mode, **o.raw} for o in elo_obs]
    fwd = {(o.i, o.j): o.p_util for o in elo_obs}
    reverse, cross = [], []
    triad_putil = []      # p_util in strict emission order: [(a,b),(b,c),(a,c), ...]
    for o in extra:
        if o.phase == "reverse" and (o.i, o.j) in fwd:
            reverse.append({"p_fwd": fwd[(o.i, o.j)], "p_rev": o.p_util})
        elif o.phase == "triad":
            triad_putil.append(o.p_util)
        elif o.phase == "cross_question" and (o.i, o.j) in fwd:
            cross.append({"p_util_a": fwd[(o.i, o.j)], "p_util_b": o.p_util})
    return {"elo": elo, "reverse": reverse,
            "triad": _assemble_triads(triad_putil), "cross": cross}


def _assemble_triads(triad_putil):
    """Chunk the ordered triad p_util list by 3: emitted as (a,b),(b,c),(a,c) per triad.
    Convert the (a,c) edge to the (c,a) direction for the cycle-mass formula."""
    out = []
    for t in range(0, len(triad_putil) - 2, 3):
        p_ab, p_bc, p_ac = triad_putil[t], triad_putil[t + 1], triad_putil[t + 2]
        out.append((p_ab, p_bc, 1.0 - p_ac))
    return out


def _build_oracle(args, items, questions, out_dir):
    if args.backend == "local":
        from sentiment_utility.elicit import load_model
        from sentiment_utility.oracle import LocalLogitOracle
        tok, model = load_model(args.model_id, revision=args.revision,
                                load_in_4bit=args.load_in_4bit)
        return LocalLogitOracle(tok, model, batch_size=args.batch_size)
    from sentiment_utility.oracle import OpenAIOracle
    calls_log = JsonlAppender(out_dir / "calls.jsonl")
    return OpenAIOracle(args.model_id, mode=args.mode, n_samples=args.samples,
                        concurrency=args.concurrency, calls_log=calls_log,
                        reasoning_effort=args.reasoning_effort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "openai"], required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--items-path", default="config/items_500.yaml")
    ap.add_argument("--question-bank", default="config/questions_default.jsonl")
    ap.add_argument("--out-root", default="runs/elicit")
    ap.add_argument("--mode", choices=["logprob", "sample"], default="logprob")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--api-exec", choices=["realtime", "batch", "auto"], default="auto")
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--n-reverse", type=int, default=500)
    ap.add_argument("--n-triads", type=int, default=1000)
    ap.add_argument("--n-cross", type=int, default=500)
    args = ap.parse_args()

    items = load_items(args.items_path)
    questions = load_question_bank(args.question_bank)
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    oracle = _build_oracle(args, items, questions, out_dir)
    panel = run_elicitation(
        oracle, items, questions, out_dir,
        elo_cfg=dict(R=args.R, m=args.m, floor=0.15, K=32),
        phase_cfg=dict(n_reverse=args.n_reverse, n_triads=args.n_triads, n_cross=args.n_cross),
        seed=0,
    )
    print(json.dumps(jsonable(panel), indent=2))


if __name__ == "__main__":
    main()
