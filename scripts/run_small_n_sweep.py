"""Run the Gemma series through the small-N judgement datasets × evaluative
constructs, producing one full four-phase elicitation per (dataset, construct, model).

Efficiency: each model is loaded ONCE and reused across all 6 (dataset, construct)
banks (one LocalLogitOracle), instead of a fresh load_model per run. Pin a process
to a GPU with CUDA_VISIBLE_DEVICES (or --gpu) and pass --models to split the series
across GPUs, e.g.:

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_small_n_sweep.py --models gemma-3-27b
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_small_n_sweep.py --models gemma-3-12b,gemma-3-4b,gemma-3-1b

Output tree:  runs/small_n/<dataset>/<construct>/<model_short>/{edges.jsonl,mu.json,panel.json,metrics.json}
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# (dataset -> items YAML); each dataset is judged with the 3 matching construct banks
DATASETS = {
    "leetcode": REPO / "config/datasets/leetcode_problems.yaml",
    "recipes":  REPO / "config/datasets/recipes.yaml",
}
CONSTRUCTS = ["harder", "interesting", "applicant"]

GEMMA_DEFAULT = ["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]

log = logging.getLogger("small_n_sweep")


def normalize_model(m: str) -> tuple[str, str]:
    """Return (hf_id, short_name). Accepts a full HF id or a gemma short name."""
    if "/" in m:
        short = m.rsplit("/", 1)[1]
        for suf in ("-it", "-Instruct", "-instruct"):
            if short.endswith(suf):
                short = short[: -len(suf)]
        return m, short
    return f"google/{m}-it", m


def bank_path(dataset: str, construct: str) -> Path:
    return REPO / f"config/questions/{dataset}_{construct}.jsonl"


def run_one(run_elicitation, oracle, items, questions, out_dir, args, run_config):
    return run_elicitation(
        oracle, items, questions, out_dir,
        elo_cfg=dict(R=args.R, m=args.m, floor=0.15, K=32),
        phase_cfg=dict(n_reverse=args.n_reverse, n_triads=args.n_triads, n_cross=args.n_cross),
        seed=args.seed, bootstrap=args.bootstrap, bootstrap_B=args.bootstrap_B,
        run_config=run_config,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(GEMMA_DEFAULT),
                    help="comma-separated HF ids or gemma short names")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma-separated subset of: " + ",".join(DATASETS))
    ap.add_argument("--constructs", default=",".join(CONSTRUCTS))
    ap.add_argument("--gpu", default=None, help="set CUDA_VISIBLE_DEVICES before importing torch")
    ap.add_argument("--out-root", default=str(REPO / "runs/small_n"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n-reverse", type=int, default=400)
    ap.add_argument("--n-triads", type=int, default=600)
    ap.add_argument("--n-cross", type=int, default=400)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a (dataset,construct,model) whose edges.jsonl already exists")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # import after CUDA_VISIBLE_DEVICES is set (load_model imports torch lazily)
    sys.path.insert(0, str(REPO / "scripts"))
    import run_elicitation as _re
    run_elicitation = _re.run_elicitation

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from question_consistency.io_utils import git_commit, load_items, setup_logging
    from question_consistency.questions import load_question_bank
    from question_consistency.elicit import load_model
    from question_consistency.oracle import LocalLogitOracle

    models = [normalize_model(m.strip()) for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    constructs = [c.strip() for c in args.constructs.split(",") if c.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    setup_logging(out_root, log_name=f"sweep_{ts}.log")
    commit = git_commit()
    log.info("commit=%s  gpu=%s  models=%s  datasets=%s constructs=%s",
             commit, os.environ.get("CUDA_VISIBLE_DEVICES"), [s for _, s in models], datasets, constructs)
    (out_root / f"sweep_config_{ts}.json").write_text(json.dumps({
        "commit": commit, "timestamp": ts, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "models": {hid: short for hid, short in models}, "datasets": datasets, "constructs": constructs,
        "elo": {"R": args.R, "m": args.m}, "batch_size": args.batch_size,
        "phase": {"n_reverse": args.n_reverse, "n_triads": args.n_triads, "n_cross": args.n_cross},
        "seed": args.seed, "load_in_4bit": bool(args.load_in_4bit),
    }, indent=2))

    # pre-load items + banks once (tiny)
    items_by_ds = {d: load_items(DATASETS[d]) for d in datasets}
    banks = {(d, c): load_question_bank(bank_path(d, c)) for d in datasets for c in constructs}

    for hf_id, short in models:
        t0 = time.time()
        log.info("==== loading %s (%s) ====", short, hf_id)
        tok, model = load_model(hf_id, load_in_4bit=args.load_in_4bit)
        oracle = LocalLogitOracle(tok, model, batch_size=args.batch_size)
        log.info("loaded %s in %.1fs", short, time.time() - t0)

        for d in datasets:
            items = items_by_ds[d]
            for c in constructs:
                out_dir = out_root / d / c / short
                if args.skip_existing and (out_dir / "edges.jsonl").exists():
                    log.info("skip existing %s/%s/%s", d, c, short); continue
                rc = {"model_id": hf_id, "model_short": short, "backend": "local",
                      "dtype": "nf4-4bit" if args.load_in_4bit else "bfloat16",
                      "dataset": d, "construct": c, "items_path": str(DATASETS[d]),
                      "question_bank": str(bank_path(d, c)), "n_items": len(items),
                      "elo_R": args.R, "elo_m": args.m, "batch_size": args.batch_size}
                tr = time.time()
                log.info("---- run %s | %s/%s (n=%d) ----", short, d, c, len(items))
                panel = run_one(run_elicitation, oracle, items, banks[(d, c)], out_dir, args, rc)
                fourish = {k: round(panel[k]["point"], 3) for k in
                           ("decisiveness", "order_consistency", "transitivity_triad", "q_agreement")
                           if k in panel}
                log.info("done %s/%s/%s in %.1fs  %s", d, c, short, time.time() - tr, fourish)

        del oracle, model, tok
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("==== finished %s (total %.1fs) ====", short, time.time() - t0)

    log.info("SWEEP COMPLETE -> %s", out_root)


if __name__ == "__main__":
    main()
