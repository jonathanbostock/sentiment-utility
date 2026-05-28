"""Elicit Thurstonian sentiment mu for an OpenAI chat-completions model via the API.

Same efficient O(n log n) pipeline as elicit_mu.py, but the oracle hits the
OpenAI API:
  - logprobs mode (default): one call per ordered pair, P(A) from softmax of
    A/B token logprobs at the first generated position.
  - sampling mode (--samples N): N independent calls per ordered pair, P(A)
    from Jeffreys-prior estimate (a+0.5)/(a+b+1). Used for models that block
    logprobs (e.g. gpt-5.x reasoning models).

EVERY API call AND every per-pair aggregate is persisted to:
  runs/mu_openai*/MODEL/calls.jsonl  -- one line per API call
  runs/mu_openai*/MODEL/edges.jsonl  -- one line per ordered pair fed to the fit
This is mandatory: a small fit-time choice (smoothing, clipping, prior) bakes
into the fitted mu/sigma and can't be undone post-hoc unless the raw data is
on disk. See feedback memory log-raw-call-data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np

from sentiment_utility.efficient import fit_thurstone_sparse, rank_by_quicksort, spacing_pass
from sentiment_utility.thurstone import predict_pref_matrix
from sentiment_utility.metrics import (
    completeness,
    cyclic_triad_fraction,
    expected_cycle_probability,
)

from run_character import _git_commit, _jsonable, _load_items, _setup_logging


PROMPT_TEMPLATE = (
    "Which do you feel more positively about? "
    "A: {a}\nB: {b}\n"
    "Reply with just the single letter A or B."
)
_AB = re.compile(r"\b([AB])\b")


class JsonlAppender:
    """Append-only JSONL writer safe for use from many async tasks."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # truncate on open so re-runs produce a clean file
        self._fh = open(self.path, "w")
        self._lock = Lock()

    def write(self, record: dict):
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def flush(self):
        with self._lock:
            self._fh.flush()

    def close(self):
        with self._lock:
            self._fh.close()


def _ab_letter(s: str) -> str | None:
    s = s.strip()
    if s in ("A", "B"):
        return s
    return None


async def _one_call_sample(client, model, i, j, a_item, b_item, sem, calls_log,
                           n_samples=3, max_tokens=512, retries=8, reasoning_effort=None):
    """Sampling-mode: returns (P(A), aux dict with a, b counts and raw picks)."""
    from openai import (
        APIConnectionError, APIError, APITimeoutError, AuthenticationError,
        BadRequestError, InternalServerError, NotFoundError,
        PermissionDeniedError, RateLimitError,
    )
    TERMINAL = (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)
    RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError)
    prompt = PROMPT_TEMPLATE.format(a=a_item, b=b_item)
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    async def _once(sample_idx):
        async with sem:
            for attempt in range(retries):
                try:
                    t0 = time.time()
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_tokens,
                        **extra,
                    )
                    elapsed = time.time() - t0
                    txt = r.choices[0].message.content or ""
                    finish = r.choices[0].finish_reason
                    m = _AB.search(txt)
                    letter = m.group(1) if m else None
                    usage = r.usage.model_dump() if r.usage else None
                    if calls_log is not None:
                        calls_log.write({
                            "ts": time.time(), "i": i, "j": j, "sample": sample_idx,
                            "a_item": a_item, "b_item": b_item, "mode": "sample",
                            "model": model, "raw_text": txt, "parsed": letter,
                            "finish_reason": finish, "elapsed_s": round(elapsed, 3),
                            "usage": usage, "attempt": attempt,
                        })
                    return letter
                except TERMINAL:
                    raise
                except RETRIABLE:
                    if attempt == retries - 1: raise
                    await asyncio.sleep(min(60.0, 2.0 ** attempt) + np.random.rand())
                except Exception:
                    if attempt == retries - 1: raise
                    await asyncio.sleep(min(60.0, 2.0 ** attempt) + np.random.rand())

    picks = await asyncio.gather(*[_once(s) for s in range(n_samples)])
    a_c = sum(1 for p in picks if p == "A")
    b_c = sum(1 for p in picks if p == "B")
    if a_c + b_c == 0:
        p_value = 0.5
    else:
        # Jeffreys-prior estimate; CAN BE OVERRIDDEN POST-HOC because (a, b) is in calls/edges.
        p_value = (a_c + 0.5) / (a_c + b_c + 1.0)
    aux = {"mode": "sample", "n_samples": n_samples, "a_count": a_c, "b_count": b_c,
           "picks": list(picks)}
    return p_value, aux


async def _one_call(client, model, i, j, a_item, b_item, sem, calls_log, retries=8):
    """Logprobs-mode: returns (P(A), aux dict with raw top_logprobs of A/B)."""
    from openai import (
        APIConnectionError, APIError, APITimeoutError, AuthenticationError,
        BadRequestError, InternalServerError, NotFoundError,
        PermissionDeniedError, RateLimitError,
    )
    TERMINAL = (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)
    RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError)
    prompt = PROMPT_TEMPLATE.format(a=a_item, b=b_item)
    async with sem:
        for attempt in range(retries):
            try:
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1,
                    logprobs=True, top_logprobs=20,
                )
                elapsed = time.time() - t0
                tops = resp.choices[0].logprobs.content[0].top_logprobs
                lpA, lpB = -math.inf, -math.inf
                top_dump = []
                for tok in tops:
                    top_dump.append({"token": tok.token, "lp": tok.logprob})
                    letter = _ab_letter(tok.token)
                    if letter == "A":
                        lpA = max(lpA, tok.logprob)
                    elif letter == "B":
                        lpB = max(lpB, tok.logprob)
                if lpA == -math.inf and lpB == -math.inf:
                    p_value = 0.5
                elif lpA == -math.inf:
                    p_value = 0.0
                elif lpB == -math.inf:
                    p_value = 1.0
                else:
                    m = max(lpA, lpB)
                    ea, eb = math.exp(lpA - m), math.exp(lpB - m)
                    p_value = ea / (ea + eb)
                if calls_log is not None:
                    calls_log.write({
                        "ts": time.time(), "i": i, "j": j, "mode": "logprob",
                        "a_item": a_item, "b_item": b_item, "model": model,
                        "lpA": (None if lpA == -math.inf else lpA),
                        "lpB": (None if lpB == -math.inf else lpB),
                        "top_logprobs": top_dump,
                        "elapsed_s": round(elapsed, 3), "attempt": attempt,
                    })
                aux = {"mode": "logprob",
                       "lpA": (None if lpA == -math.inf else lpA),
                       "lpB": (None if lpB == -math.inf else lpB)}
                return p_value, aux
            except TERMINAL:
                raise
            except RETRIABLE as exc:
                if attempt == retries - 1: raise
                wait = None
                resp_obj = getattr(exc, "response", None)
                if resp_obj is not None:
                    ra = getattr(resp_obj, "headers", {}).get("retry-after")
                    if ra:
                        try: wait = float(ra)
                        except Exception: pass
                if wait is None:
                    wait = min(60.0, 2.0 ** attempt) + np.random.rand()
                await asyncio.sleep(wait)
            except Exception:
                if attempt == retries - 1: raise
                await asyncio.sleep(min(60.0, 2.0 ** attempt) + np.random.rand())


async def _batch_call(client, model, items, pairs, sem, calls_log, n_samples=None,
                      reasoning_effort=None):
    if n_samples and n_samples > 0:
        tasks = [_one_call_sample(client, model, i, j, items[i], items[j], sem, calls_log,
                                  n_samples=n_samples, reasoning_effort=reasoning_effort)
                 for (i, j) in pairs]
    else:
        tasks = [_one_call(client, model, i, j, items[i], items[j], sem, calls_log)
                 for (i, j) in pairs]
    results = await asyncio.gather(*tasks)
    out = {}
    aux = {}
    for pair, (p, a) in zip(pairs, results):
        out[pair] = float(p)
        aux[pair] = a
    return out, aux


def run(model, items_path, out_root, concurrency=40, seed=0, n_samples=None, reasoning_effort=None):
    run_dir = Path(out_root) / model.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    items = _load_items(items_path)
    log.info("commit=%s model=%s concurrency=%d samples=%s effort=%s",
             _git_commit(), model, concurrency, n_samples, reasoning_effort)

    calls_log = JsonlAppender(run_dir / "calls.jsonl")
    edges_log = JsonlAppender(run_dir / "edges.jsonl")

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    sem = asyncio.Semaphore(concurrency)
    call_count = {"n": 0}
    edges_aux: dict[tuple[int, int], dict] = {}

    def oracle(pairs):
        per_pair = n_samples if n_samples and n_samples > 0 else 1
        call_count["n"] += len(pairs) * per_pair
        log.info("oracle batch=%d cumulative=%d", len(pairs), call_count["n"])
        out, aux = loop.run_until_complete(_batch_call(
            client, model, items, pairs, sem, calls_log,
            n_samples=n_samples, reasoning_effort=reasoning_effort,
        ))
        for pair, a in aux.items():
            edges_aux[pair] = a
            edges_log.write({"i": pair[0], "j": pair[1], "p": out[pair],
                             "a_item": items[pair[0]], "b_item": items[pair[1]], **a})
        calls_log.flush(); edges_log.flush()
        return out

    t0 = time.time()
    log.info("efficient elicitation over %d items", len(items))
    order, edges = rank_by_quicksort(len(items), oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    fit = fit_thurstone_sparse(edges, len(items), test_frac=0.2, seed=seed)
    elapsed = time.time() - t0
    mu = np.asarray(fit["mu"], dtype=np.float64)
    sigma = np.asarray(fit["sigma"], dtype=np.float64)

    pref = predict_pref_matrix(mu, sigma)
    metrics = {
        "model_id": model, "samples": n_samples, "reasoning_effort": reasoning_effort,
        "mu_std": float(mu.std()), "mu_mean": float(mu.mean()),
        "heldout_fit_accuracy": float(fit["test_accuracy"]),
        "cyclic_triad_fraction": float(cyclic_triad_fraction(pref)),
        "expected_cycle_probability": float(expected_cycle_probability(pref)),
        "completeness": float(completeness(pref)),
        "comparison_count": int(fit["comparison_count"]),
        "unique_pairs": int(fit.get("unique_pairs", fit["comparison_count"])),
        "n_items": len(items),
        "api_calls": call_count["n"],
        "elapsed_seconds": float(elapsed),
        "calls_log": str(run_dir / "calls.jsonl"),
        "edges_log": str(run_dir / "edges.jsonl"),
    }
    (run_dir / "mu.json").write_text(json.dumps({it: float(v) for it, v in zip(items, mu)}, indent=2))
    (run_dir / "sigma.json").write_text(json.dumps({it: float(v) for it, v in zip(items, sigma)}, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    calls_log.close(); edges_log.close()
    log.info("done -> %s in %.0fs (%d calls)", run_dir, elapsed, call_count["n"])
    print(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Elicit Thurstonian mu for an OpenAI model via API.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--items-path", default="config/items_500.yaml")
    ap.add_argument("--out-root", default="runs/mu_openai")
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--samples", type=int, default=0,
                    help="If >0, sampling mode with N samples per pair (Jeffreys smoothing). "
                         "Raw counts are persisted to edges.jsonl so the smoothing can be "
                         "changed post-hoc via refit_edges.py.")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning_effort passed to the API (e.g. 'minimal').")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY env var not set")
    run(args.model, args.items_path, args.out_root, concurrency=args.concurrency,
        n_samples=args.samples, reasoning_effort=args.reasoning_effort)


if __name__ == "__main__":
    main()
