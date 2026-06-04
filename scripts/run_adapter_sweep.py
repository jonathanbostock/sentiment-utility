"""Single-load adapter-swap sweep: load a base model ONCE, then run the elicitation
pipeline for each LoRA adapter by hot-swapping it in-process (PEFT load_adapter/
set_adapter/delete_adapter), avoiding a full base reload per model.

Supports sharding (--shard k/N) so the adapter list can be split across parallel pods,
and optional pod self-termination (--terminate-pod) so an unattended run stops billing the
moment it finishes (see memory: overnight-gpu-self-terminate).

Pure helpers (load_adapter_list, select_shard) are unit-tested; the GPU swap loop is
exercised on a pod.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from question_consistency.io_utils import load_items, setup_logging
from question_consistency.questions import load_question_bank


def load_adapter_list(path) -> list[str]:
    """One adapter repo id per line; blank lines and # comments ignored."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def select_shard(items: list[str], shard: str | None) -> list[str]:
    """shard='k/N' -> the k-th of N contiguous-strided shards (1-based k). None -> all."""
    if not shard:
        return items
    k, n = (int(x) for x in shard.split("/"))
    if not (1 <= k <= n):
        raise ValueError(f"bad shard {shard!r}: need 1<=k<=N")
    return [it for idx, it in enumerate(items) if idx % n == (k - 1)]


def _adapter_name(repo: str) -> str:
    """Output subfolder name: repo basename."""
    return repo.split("/")[-1]


def run_sweep(base_model, adapters, items, questions, out_root, *, include_base=False,
              load_in_4bit=True, batch_size=16, elo_cfg=None, phase_cfg=None,
              bootstrap=False, seed=0, log=None):
    import torch
    from question_consistency.elicit import load_model
    from question_consistency.oracle import LocalLogitOracle
    # import here so the pure helpers above stay importable without these scripts/deps
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_elicitation import run_elicitation

    elo_cfg = elo_cfg or dict(R=5, m=5, floor=0.15, K=32)
    phase_cfg = phase_cfg or dict(n_reverse=500, n_triads=1000, n_cross=500)
    out_root = Path(out_root)

    def _log(msg):
        (log.info if log else print)(msg)

    _log(f"loading base {base_model} (4bit={load_in_4bit}) once")
    tok, base = load_model(base_model, "bfloat16", load_in_4bit=load_in_4bit)
    done, failed = [], []

    def _run(name, model):
        try:
            oracle = LocalLogitOracle(tok, model, batch_size=batch_size)
            run_elicitation(oracle, items, questions, out_root / name, elo_cfg, phase_cfg,
                            seed=seed, bootstrap=bootstrap)
            done.append(name); _log(f"OK {name}")
        except Exception:
            failed.append(name); _log(f"FAIL {name}\n{traceback.format_exc()}")

    if include_base:
        _run("base", base)

    from peft import PeftModel
    pmodel = None
    for repo in adapters:
        name = _adapter_name(repo)
        try:
            if pmodel is None:
                pmodel = PeftModel.from_pretrained(base, repo, adapter_name="cur")
            else:
                pmodel.load_adapter(repo, adapter_name="cur")
            pmodel.set_adapter("cur")
            pmodel.eval()
        except Exception:
            failed.append(name); _log(f"FAIL(load) {name}\n{traceback.format_exc()}"); continue
        _run(name, pmodel)
        try:
            pmodel.delete_adapter("cur")          # keep memory bounded across the sweep
        except Exception:
            pass

    (out_root / "DONE").write_text(f"done={len(done)} failed={len(failed)}\n")
    if failed:
        (out_root / "FAILED").write_text("\n".join(failed) + "\n")
    _log(f"SWEEP COMPLETE: {len(done)} ok, {len(failed)} failed")
    return done, failed


def _maybe_terminate_pod(log=None):
    import os
    import shutil
    import subprocess
    pod = os.environ.get("RUNPOD_POD_ID")
    msg = (log.info if log else print)
    if not pod:
        msg("terminate-pod requested but RUNPOD_POD_ID not set; skipping"); return
    if not shutil.which("runpodctl"):
        msg("terminate-pod requested but runpodctl not found; skipping"); return
    msg(f"self-terminating pod {pod}")
    subprocess.run(["runpodctl", "remove", "pod", pod], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--adapters-file", required=True)
    ap.add_argument("--shard", default=None, help="k/N to run only shard k of N")
    ap.add_argument("--include-base", action="store_true")
    ap.add_argument("--items-path", default="config/datasets/items_2000.yaml")
    ap.add_argument("--question-bank", default="config/questions/main.jsonl")
    ap.add_argument("--out-root", default="runs/audit70_sweep")
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-reverse", type=int, default=500)
    ap.add_argument("--n-triads", type=int, default=1000)
    ap.add_argument("--n-cross", type=int, default=500)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--terminate-pod", action="store_true",
                    help="runpodctl remove this pod when the sweep finishes (bounds cost)")
    args = ap.parse_args()

    adapters = select_shard(load_adapter_list(args.adapters_file), args.shard)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_root, "sweep.log")
    log.info("base=%s adapters=%d shard=%s include_base=%s",
             args.base_model, len(adapters), args.shard, args.include_base)
    items = load_items(args.items_path)
    questions = load_question_bank(args.question_bank)
    try:
        run_sweep(args.base_model, adapters, items, questions, out_root,
                  include_base=args.include_base, load_in_4bit=args.load_in_4bit,
                  batch_size=args.batch_size,
                  phase_cfg=dict(n_reverse=args.n_reverse, n_triads=args.n_triads,
                                 n_cross=args.n_cross),
                  bootstrap=args.bootstrap, log=log)
    finally:
        if args.terminate_pod:
            _maybe_terminate_pod(log)


if __name__ == "__main__":
    main()
