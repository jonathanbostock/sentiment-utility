"""Elicit Thurstonian sentiment mu for an OpenAI chat-completions model via the API.

Same efficient O(n log n) pipeline as elicit_mu.py, but the oracle hits the
OpenAI API with logprobs + top_logprobs to read P(A) vs P(B) at the first
generated token — analogous to our local answer-token-logprob method, just
remote. No GPU needed.

Usage:
    OPENAI_API_KEY=sk-... python scripts/elicit_mu_openai.py --model gpt-5-mini
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

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


def _is_AB_token(s: str) -> str | None:
    s = s.strip()
    if s in ("A", "B"):
        return s
    return None


async def _one_call_sample(client, model, a, b, sem, n_samples=5, max_tokens=512, retries=8):
    """Sampling-mode call for models that block logprobs (gpt-5.x reasoning models).

    Makes `n_samples` independent calls, parses the first standalone A/B letter
    from each response, returns P(A) = a_count / (a_count + b_count).
    """
    import re
    from openai import (
        APIConnectionError, APIError, APITimeoutError, AuthenticationError,
        BadRequestError, InternalServerError, NotFoundError,
        PermissionDeniedError, RateLimitError,
    )
    TERMINAL = (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)
    RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError)
    prompt = PROMPT_TEMPLATE.format(a=a, b=b)
    AB = re.compile(r"\b([AB])\b")

    async def _once():
        async with sem:
            for attempt in range(retries):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_tokens,
                    )
                    txt = r.choices[0].message.content or ""
                    m = AB.search(txt)
                    return m.group(1) if m else None
                except TERMINAL:
                    raise
                except RETRIABLE as exc:
                    if attempt == retries - 1: raise
                    await asyncio.sleep(min(60.0, 2.0 ** attempt) + np.random.rand())
                except Exception:
                    if attempt == retries - 1: raise
                    await asyncio.sleep(min(60.0, 2.0 ** attempt) + np.random.rand())

    picks = await asyncio.gather(*[_once() for _ in range(n_samples)])
    a_c = sum(1 for p in picks if p == "A")
    b_c = sum(1 for p in picks if p == "B")
    if a_c + b_c == 0:
        return 0.5, None
    # Jeffreys-prior estimate (Beta(0.5,0.5)): P(A) = (a + 0.5) / (a + b + 1).
    # Avoids saturating at 0/1 with small N, which would blow up the Thurstonian fit's BCE.
    return (a_c + 0.5) / (a_c + b_c + 1.0), None


async def _one_call(client, model, a, b, sem, retries=8):
    """Single API call; returns (P(pick A), raw).

    Robust retry policy for new accounts that may rate-limit:
    - Distinguish RETRIABLE (RateLimit, APIConnection, InternalServer, Timeout)
      from TERMINAL (BadRequest, Authentication, PermissionDenied, NotFound).
    - Exponential backoff with jitter; respect `Retry-After` if present
      (commonly attached to the exception response).
    - Up to 8 retries; max single sleep capped at ~60s; total budget ~2min.
    """
    from openai import (
        APIConnectionError, APIError, APITimeoutError, AuthenticationError,
        BadRequestError, InternalServerError, NotFoundError,
        PermissionDeniedError, RateLimitError,
    )
    TERMINAL = (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError)
    RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError)
    prompt = PROMPT_TEMPLATE.format(a=a, b=b)
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
                tops = resp.choices[0].logprobs.content[0].top_logprobs
                lpA, lpB = -math.inf, -math.inf
                for tok in tops:
                    letter = _is_AB_token(tok.token)
                    if letter == "A":
                        lpA = max(lpA, tok.logprob)
                    elif letter == "B":
                        lpB = max(lpB, tok.logprob)
                if lpA == -math.inf and lpB == -math.inf:
                    return 0.5, None
                if lpA == -math.inf:
                    return 0.0, None
                if lpB == -math.inf:
                    return 1.0, None
                m = max(lpA, lpB)
                ea, eb = math.exp(lpA - m), math.exp(lpB - m)
                return ea / (ea + eb), None
            except TERMINAL:
                raise   # never retry these
            except RETRIABLE as exc:
                if attempt == retries - 1:
                    raise
                # try to honour a Retry-After header on rate-limit errors
                wait = None
                resp_obj = getattr(exc, "response", None)
                if resp_obj is not None:
                    ra = getattr(resp_obj, "headers", {}).get("retry-after")
                    if ra:
                        try:
                            wait = float(ra)
                        except Exception:
                            pass
                if wait is None:
                    wait = min(60.0, 2.0 ** attempt) + np.random.rand()
                await asyncio.sleep(wait)
            except Exception:
                # unknown exception: backoff and retry conservatively
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(min(60.0, 2.0 ** attempt) + np.random.rand())


async def _batch_call(client, model, items, pairs, sem, n_samples=None):
    if n_samples and n_samples > 0:
        tasks = [_one_call_sample(client, model, items[i], items[j], sem, n_samples=n_samples)
                 for (i, j) in pairs]
    else:
        tasks = [_one_call(client, model, items[i], items[j], sem) for (i, j) in pairs]
    results = await asyncio.gather(*tasks)
    return {pair: float(p) for pair, (p, _) in zip(pairs, results)}


def _sync_oracle(client, model, items, pairs, sem, loop, n_samples=None):
    return loop.run_until_complete(_batch_call(client, model, items, pairs, sem, n_samples=n_samples))


def run(model, items_path, out_root, concurrency=40, seed=0, n_samples=None):
    run_dir = Path(out_root) / model.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    items = _load_items(items_path)
    log.info("commit=%s model=%s concurrency=%d", _git_commit(), model, concurrency)

    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # picks OPENAI_API_KEY env var
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sem = asyncio.Semaphore(concurrency)
    call_count = {"n": 0}

    def oracle(pairs):
        # n_samples-fold the call count if sampling
        per_pair = n_samples if n_samples and n_samples > 0 else 1
        call_count["n"] += len(pairs) * per_pair
        log.info("oracle batch=%d cumulative=%d", len(pairs), call_count["n"])
        return _sync_oracle(client, model, items, pairs, sem, loop, n_samples=n_samples)

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
        "model_id": model,
        "mu_std": float(mu.std()),
        "mu_mean": float(mu.mean()),
        "heldout_fit_accuracy": float(fit["test_accuracy"]),
        "cyclic_triad_fraction": float(cyclic_triad_fraction(pref)),
        "expected_cycle_probability": float(expected_cycle_probability(pref)),
        "completeness": float(completeness(pref)),
        "comparison_count": int(fit["comparison_count"]),
        "unique_pairs": int(fit.get("unique_pairs", fit["comparison_count"])),
        "n_items": len(items),
        "api_calls": call_count["n"],
        "elapsed_seconds": float(elapsed),
    }
    (run_dir / "mu.json").write_text(json.dumps({it: float(v) for it, v in zip(items, mu)}, indent=2))
    (run_dir / "sigma.json").write_text(json.dumps({it: float(v) for it, v in zip(items, sigma)}, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    log.info("done -> %s in %.0fs (%d calls)", run_dir, elapsed, call_count["n"])
    print(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Elicit Thurstonian mu for an OpenAI model via API.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--items-path", default="config/items_500.yaml")
    ap.add_argument("--out-root", default="runs/mu_openai")
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--samples", type=int, default=0,
                    help="If >0, use sampling-mode with this many independent samples per "
                         "ordered pair (needed for models that block logprobs, e.g. gpt-5.x).")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY env var not set")
    run(args.model, args.items_path, args.out_root, concurrency=args.concurrency,
        n_samples=args.samples)


if __name__ == "__main__":
    main()
