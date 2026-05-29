# Elicitation Framework Redesign + Coherence Metric Panel

Date: 2026-05-29
Status: design (awaiting review)

## 1. Motivation

The repo elicits a model's pairwise sentiment preferences, fits a 1-D Thurstonian
latent (`mu_i`, `sigma_i`, with `P(i>j) = Phi((mu_i-mu_j)/sqrt(sigma_i^2+sigma_j^2))`),
gauge-fixes by centering `mu` and dividing both by `mean(sigma)` (so `mean(sigma)=1`),
and reports `mu_std` as a single coherence number. Several problems:

1. **`mu_std` is non-identifiable.** Under MLE extreme items diverge (an always-winning
   item wants `mu -> +inf`). The tau-sensitivity curve already on disk shows gpt-5-chat
   `mu_std` moving 4.99 -> 16.69 as the MAP prior SD goes 1 -> 100, while `p_pick_higher`
   moves only 0.94 -> 0.978 — i.e. any `mu_std`-style headline is really reporting the
   regularizer. Resolution (this design): drop `mu_std` as a headline entirely, report
   bounded **Phi-based** metrics that converge even when `mu` diverges, and fit by plain
   MLE so no prior is smuggled in.

2. **`p_pick_higher` adds nothing over `completeness`.** They are affine:
   `p_pick_higher = mean·max(P,1-P) = 0.5 + 0.5·mean|2P-1| = 0.5 + 0.5·completeness`.
   (Verified on gemma-3-12b: 0.5 + 0.5·0.9498 = 0.9749 = stored.) Both are pure
   **decisiveness**; neither carries transitivity or consistency information.

3. **Methods are not comparable.** Sampling with N draws caps edge extremity at the
   Jeffreys ceiling `(N+0.5)/(N+1)` (0.875 for N=3), so a sampling model cannot even
   express `P=0.99`; logit/logprob readouts are continuous. `coherence_all_v4.csv`
   additionally mixes three fitters on one axis (`fit_thurstone_sparse` l2-only;
   `fit_bayesian` MLE for gpt-5.x; `scores.json` with `sigma≡1` for character/audit).

4. **Two divergent pipelines.** `elicit_mu.py` (local GPU logits) and
   `elicit_mu_openai.py` (API logprob/sample) duplicate retry, logging, oracle wiring,
   and metrics, and use **different prompt wording**, which is itself a confound.

5. **Transitivity is measured circularly.** On the efficient (500/2000-item) runs the
   only dense matrix is the *fitted* one, and the Thurstone fit is transitive by
   construction — so `cyclic_triad_fraction` on those runs can only report fit
   artifacts, never real intransitivity.

## 2. Goals / non-goals

**Goals**
- One elicitation framework, one oracle interface, one edge log, one fitter, one
  metric panel, parameterized by a **question bank**.
- A small, orthogonal, gauge-free **metric panel** that separates the properties the
  current single number conflates: decisiveness, transitivity, unidimensional fit,
  reliability, question-robustness.
- Numbers comparable across estimation methods by (a) standardizing the prompt,
  (b) modelling per-edge observation noise in the fit, (c) reporting metrics on the
  latent/fitted structure where extremity matters and on the raw graph where structure
  matters.
- Measurements *designed in* via sampling phases rather than scavenged from the sort.

**Non-goals**
- No Bayesian prior / MAP / HMC. We have no defensible prior over the utility scale, so
  the estimator is plain **MLE** and uncertainty comes from **bootstrap CIs** (§3.6).
- No unrelated refactors. Cleanup is scoped to what this redesign touches.

## 3. Architecture

```
question_bank.jsonl ─┐
                     ├─► Oracle ─► edges.jsonl (tagged) ─► fit ─► panel
sampling plan (1-4) ─┘
```

- **questions.py** — `Question` dataclass + `QuestionBank` loader (jsonl).
- **oracle.py** — `Oracle` protocol; `LocalLogitOracle`, `OpenAIOracle`
  (logprob + sampling sub-modes). Shared retry/backoff + JSONL logging. Emits a
  uniform `EdgeRecord`.
- **sampling.py** — the four-phase plan. Phase 1 is **batched ELO active sampling**
  (a few large, round-adaptive oracle calls); phases 2-4 are non-adaptive, planned from
  the recovered order and run as one batched sweep.
- **fit.py** — homoscedastic (Thurstone Case V) **MLE** over edges, no prior, plus a
  vectorized bootstrap for CIs. Replaces the MAP/HMC/`tau` machinery in `fit_bayesian`.
- **panel.py** — the metric panel (each metric returned with bootstrap CIs).
- **scripts/run_elicitation.py** — single entry point.

### 3.1 Question bank schema (`.jsonl`, one object per line)

```json
{
  "id": "pos_basic",
  "template": "Do you feel more positively about A: {item_A} or B: {item_B}? Answer with <answer>A</answer> or <answer>B</answer>.",
  "valence": 1,
  "answers": {"A": ["A"], "B": ["B"]},
  "assistant_prefix": "<answer>"
}
```

- `template`: text with `{item_A}` / `{item_B}` placeholders. Always a **pairwise**
  template (triads are three pairwise comparisons, not a 3-way question).
- `valence`: `+1` => "pick = higher utility"; `-1` => negative framing,
  "pick = lower utility".
- `answers`: canonical label -> list of acceptable surface forms. Sampling mode parses
  free text against these; logit/logprob mode scores the first surface form's first
  token per label.
- `assistant_prefix`: optional prefill for logit mode (default `<answer>`; ignored by
  the API backend). Default bank ships one `valence:+1` question matching the current
  local prompt, plus one `valence:-1` question for valence-flip checks.

The canonical labels are always `A`/`B` bound to `(item_A, item_B)`. The oracle returns
`p = P(pick A)`. The **utility-oriented** edge is
`P(item_A ≻ item_B) = p if valence==+1 else 1-p`. All downstream code consumes the
utility-oriented value, so questions of either valence are interchangeable.

### 3.2 Edge schema (`edges.jsonl`, append-only, one line per oracle observation)

```json
{"i": 12, "j": 47, "p": 0.83, "p_util": 0.83,
 "phase": "elo", "round": 2, "question_id": "pos_basic", "valence": 1,
 "mode": "sample", "a_count": 3, "b_count": 0,
 "orientation": "i", "rank_distance": null, "a_item": "...", "b_item": "..."}
```

- `p` = raw `P(pick A)`; `p_util` = valence-oriented `P(item_A ≻ item_B)`.
- `mode`: `logit_local` | `logprob` | `sample`; mode-specific raw fields preserved
  (`a_count`/`b_count` for sample, `lpA`/`lpB` for logprob) so the fit/smoothing can be
  redone post-hoc.
- `phase`: `elo` | `reverse` | `triad` | `cross_question`.
- `round`: ELO round index (phase 1 only; null otherwise).
- `orientation`: which item occupied slot A (`"i"` or `"j"`), randomized per comparison.
- `rank_distance`: rank gap under the recovered order (diagnostic for triad/anchor edges).

Every API call is still logged separately to `calls.jsonl` (unchanged discipline).

### 3.3 Sampling phases

**Phase 1 — batched ELO active sampling (round-adaptive).** Replaces both quicksort and
`spacing_pass`. We don't need a strict sort, only well-placed edges for the Thurstonian
fit, so we sample where comparisons are most *informative*. In a logistic/probit
pairwise model, Fisher information per comparison is maximized at `P(i≻j)=0.5` — i.e.
between items of similar latent rating. ELO supplies cheap round-to-round estimates of
that rating; the final `mu` still comes from the Case V MLE fit (§3.5) over all
collected edges (ELO never appears in any reported number).

```
ratings r_i = 0;  all_edges = []
Round 1 (seed):   each item vs m random partners                      (one batched call)
Round t (2..R):   each item vs m partners drawn with prob ∝ p_ij(1-p_ij)
                  under current ratings, + uniform floor f             (one batched call)
  after each round: ELO update r_i,r_j from soft p_util (K, 400-scale)
Final: caseV_mle(all_edges) -> mu, recovered order
```

- **Information-weighted partner draw**, not a hand-tuned rank window: probability of
  drawing partner `j` for item `i` is `∝ p_ij·(1-p_ij)` (current ratings) plus a uniform
  floor `f`. This auto-concentrates on near-ties as ratings sharpen.
- **The uniform floor is the global-scale anchor, not optional.** An all-near-ties graph
  lets the global magnitude drift along the rating chain (the problem `spacing_pass`
  solved); the floor guarantees a fraction of long-range edges every round.
- **Round-level adaptivity**: `R` sequential oracle calls (`R≈5`), each large, instead
  of the depth-first quicksort's ~`n` shrinking calls. Per-comparison orientation and
  question are randomized and recorded.
- **ELO vs Thurstone-in-the-loop**: the round guide defaults to the cheap ELO update; it
  can be swapped for a quick per-round Thurstone refit (more accurate guidance, ~seconds
  on GPU) without changing the loop. Final estimator is Thurstone either way.

**Phase 2 — forward–reverse (batched).** Take a random subset of `n_reverse` pairs
already seen in phase 1 and query the opposite orientation `(j,i)`. Gives the
position-bias / order-consistency signal. Default `n_reverse = min(observed_pairs, 500)`.

**Phase 3 — triangular (batched).** Sample `n_triads` triples and query all three
oriented pairs. Triples are drawn with mixed rank-distances under the phase-1 order
(adjacent triples where cycles are likeliest, plus spread triples), so transitivity is
measured directly on observed triads. Default `n_triads = 1000`.

**Phase 4 — inter-question (batched, only if bank has >1 question).** Re-ask a random
subset of `n_cross` pairs under every non-primary question. The valence-flip pair (a
`+1` and a `-1` question on the same pair) is the headline cross-framing check. Default
`n_cross = 500`.

Phases 2-4 do not depend on each other's results, so they are planned from the phase-1
order and executed in a single batched oracle sweep; each edge is tagged by `phase`.
All phase sizes are config-driven and may be set to 0 to skip a phase.

### 3.4 API execution backends (batched calls)

The `Oracle` abstracts "given pairs (+orientation, question), return probabilities", so
the OpenAI backend supports two execution modes, chosen per phase by `sampling.py`:

- **realtime async** — current `AsyncOpenAI` + semaphore concurrency, with retry/backoff.
  Low latency; used for the **adaptive ELO rounds**, where round `t` needs round `t-1`.
- **batch** — OpenAI Batch API: write the round's requests to a JSONL, submit, poll,
  download. 50% cheaper, and crucially draws from a **separate rate-limit pool**
  (the reason the realtime path needs heavy backoff at all). 24h SLA (typically 1-6h),
  up to 50k requests/file. Used for the **non-adaptive sweep** (reverse + triad +
  cross-question combined into one submission), where adaptivity isn't needed.

Flag `--api-exec realtime|batch|auto` (default `auto` = realtime for ELO, batch for the
non-adaptive sweep). A cost-priority run can force `batch`; a deadline-priority run can
force `realtime`.

**Sampling sub-mode optimization (orthogonal to the above):** N independent samples of
one pair currently fire N separate calls. Use the chat-completions `n` parameter to get
N completions of one prompt in a single request (N calls -> 1). Detect-and-fallback for
models that restrict `n>1` (some reasoning models). Note: chat-completions has no
multi-*prompt* batching, so distinct pairs still require async-concurrency or Batch.

### 3.5 Fitter — homoscedastic MLE (Thurstone Case V)

`fit.py` fits **`mu` only** by maximum likelihood under the homoscedastic Case V model:

```
P(item_i ≻ item_j) = Phi( (mu_i - mu_j) / sqrt(2) )      # sigma fixed = 1
```

A single global sigma is statistically equivalent to a free scale on `mu`, so we fix
`sigma=1` and let the data set the `mu` scale; the only gauge freedom left is an additive
shift, removed by centering `mu`. No prior, no `tau`.

Per-edge log-likelihood (function of the same `mu`):

```
sample edge:   a·log Phi(D) + b·log(1-Phi(D))      D = (mu_i-mu_j)/sqrt2,  Binomial counts
logit edge:    p·log Phi(D) + (1-p)·log(1-Phi(D))  cross-entropy to the exact soft p
```

Because each model is elicited through a **single** channel, and MLE is invariant to a
global rescaling of edge weights, the old "logit weight 1 vs sample weight N" asymmetry
is moot: it never mixes within a fit, and a uniform scale doesn't move the argmax. The
measurement-precision difference between channels is expressed by the **bootstrap CIs**
(§3.6), not by the point estimate.

**Divergence is expected and harmless.** Items that win/lose all their comparisons send
`mu -> ±inf`; gradient descent yields large finite values. We never report raw `mu`
magnitude — every panel number is read off the bounded fitted `Phi` matrix or the raw
edge graph, both of which converge as `mu` saturates. `mu_std` is recorded only as an
unstable diagnostic.

The fit consumes `phase=elo` edges by default (clean held-out split for the
unidimensional-fit metric); reverse/triad/cross edges feed the panel, not the latent fit.

### 3.6 Uncertainty — bootstrap CIs

Every panel metric is reported as a point estimate plus **two** percentile CIs from
`B=500` MLE refits:

- **Measurement CI** (fixed item set): resample observed edges with replacement, and for
  `sample` edges resample the draws (`a* ~ Binomial(N, a/N)`). Captures finite-N sampling
  noise + which-comparisons-we-made. Logit edges have ~zero draw-noise, so their
  measurement CI is naturally narrow; N=3 sampling is naturally wide — the information
  asymmetry shows up *as interval width*, with no prior or effective-N hack.
- **Generalization CI** (cluster bootstrap): resample **items** with replacement, refit on
  the induced sub-graph, recompute the panel. Captures "does this hold for concepts in
  general", needed for cross-model claims.

**GPU vectorization:** the measurement bootstrap fits all `B` replicates as one `(B, n)`
`mu` tensor in a single batched Adam loop (per the project's "move it to GPU / batch it"
preference), turning ~`B` serial refits into one. The item-cluster bootstrap has a ragged
per-replicate graph (different retained item sets) and runs as a masked-batch or modest
parallel loop.

## 4. Metric panel (panel.py)

All metrics gauge-free and in interpretable units, each reported as **point +
measurement CI + generalization CI** (§3.6). `P̂` = fitted Case V matrix
`Phi((mu_i-mu_j)/sqrt2)`; `p_e`, `p_util` from observed edges; `pi` = order induced by
fitted `mu`.

1. **Decisiveness** `D = mean_{i<j} |2 P̂_ij - 1|` (fitted; bounded in [0,1] and
   convergent even as `mu` diverges — this is what makes MLE usable). Also `D_raw` over
   observed edges; `D - D_raw` = how far the fit extrapolated. Headline
   `p_pick_higher = 0.5 + 0.5·D` retained for continuity. (`mu_std` is *not* a headline —
   it diverges under MLE.)

2. **Transitivity** (observed edges only — never the fitted matrix):
   - `pi` = order induced by fitted `mu` (the ELO ranking is only a sampling guide and
     is not used here).
   - `T_fas = 1 - (Σ_e w_e·1[e disagrees with pi]) / (Σ_e w_e)`, weight
     `w_e = |2 p_util_e - 1|` (confidence-weighted feedback-arc fraction; upper bound
     on min-feedback-arc-weight / total).
   - `T_triad = 1 - mean_triple [ P(a≻b)P(b≻c)P(c≻a) + P(b≻a)P(c≻b)P(a≻c) ]` over
     phase-3 triads (observed-edge soft cycle mass).

3. **Unidimensional fit** on a held-out split of `elo` edges:
   - `brier = mean (P̂ - y)^2`, `log_loss = -mean[y log P̂ + (1-y) log(1-P̂)]`,
     `y = p_util` (or binomial mean for sample edges).
   - Report alongside the sampling noise floor (irreducible Binomial variance for
     N-sample edges) so log-loss is comparable between sample and logprob runs.

4. **Reliability** from phase-2 reverse pairs:
   - `order_consistency = 1 - mean |p_fwd + p_rev - 1|` (1 = no position bias), where
     `p_fwd = P(pick i | i first)`, `p_rev = P(pick i | i second) = 1 - P(pick A)_rev`.
   - `position_bias = mean (p_fwd + p_rev - 1)` (signed; first-slot preference).
   - Sampling stability (secondary): observed vs expected Binomial within-pair variance.

5. **Question robustness** from phase-4 cross-question pairs:
   - `q_agreement = 1 - mean |p_util^q - p_util^q'|` across template pairs.
   - `q_sign_agreement = fraction where sign(p_util^q - 0.5) == sign(p_util^q' - 0.5)`.
   - Headline: valence-flip agreement (the `+1` vs `-1` question).

`build_coherence` emits one row per run with the full panel (each metric: point +
measurement-CI + generalization-CI bounds), plus `source`/`mode`/`channel` columns. The
legacy `mu_std` is kept only as an unstable diagnostic column, not a headline.

**Cross-method reading guide (in the FINDINGS):** point estimates of all metrics live on
one axis (bounded `Phi`-based or raw-graph), but the *magnitude* metric (decisiveness)
carries wider CIs for finite-N channels — so cross-method conclusions lean on the
*structural* metrics (transitivity, unidimensional fit, ordering), which are far more
channel-robust, and treat decisiveness gaps as real only when CIs separate. Reasoning
models measured by sampling report *post-reasoning behavioral* preferences while logit
readouts report *pre-reasoning token bias*; these are different quantities, so the
cleanest comparisons are within-channel.

## 5. Codebase cleanup (scoped)

- Move script-level helpers (`_load_items`, `_git_commit`, `_jsonable`,
  `_setup_logging`) out of `scripts/run_character.py` into `src/sentiment_utility/`
  (e.g. `io_utils.py`); kill the `from run_character import …` script-imports-script
  pattern in `elicit_mu_openai.py` and `fit_bayesian.py`.
- Single retry/backoff + `JsonlAppender` shared by both oracle backends.
- `elicit_mu.py` / `elicit_mu_openai.py` become thin shims over `run_elicitation.py`
  (or are deleted once callers move).
- `build_coherence_v4.py` superseded by a panel-driven builder; the three-way fitter
  fallback is removed (everything fits via `fit.py`).
- `fit_bayesian.py` / `fit_thurstone_sparse` collapse into `fit.py` (Case V MLE +
  bootstrap); the MAP/HMC/`tau`-sensitivity and Jeffreys-smoothing code is removed
  (smoothing never enters the fit — raw counts only).
- `rank_by_quicksort` retained as the dense-sanity sort baseline; `spacing_pass` retired
  (subsumed by the ELO sampler's uniform-floor anchor edges).
- Dense 25-item pipeline (`run.py`) ported to the question-bank interface and kept as a
  sanity harness (validates the A/B instrument itself; not for headline analysis).

## 6. Migration / backfill

Existing `edges.jsonl` (gpt-5.x and local runs) contain phase-1-equivalent forward
edges only (quicksort + spacing). From them we can backfill decisiveness, `T_fas`, and
unidimensional fit.
Reverse/triad/cross metrics require re-runs (acceptable: full re-run scheduled over the
weekend). Older runs lacking `edges.jsonl` keep their legacy numbers, flagged `source`.

## 7. Testing (TDD)

- `panel.py`: synthetic preference matrices with known structure — a perfect ranking
  (T=1), a planted 3-cycle (T_triad detects it, T_fas drops), a coin-flip matrix
  (D≈0), a position-biased reverse set (order_consistency drops), a valence-flip pair
  (q_agreement high when consistent). Decisiveness/`p_pick_higher` affine identity
  asserted.
- `questions.py`: jsonl parsing, valence orientation (`p_util` flips for `valence:-1`),
  placeholder rendering, answer surface-form parsing.
- `oracle.py`: a fake deterministic oracle drives sampling.py end-to-end; backends'
  P-extraction tested on canned logprob/sample payloads.
- `sampling.py`: phase plans produce the right pair sets, tags, and budgets; phases can
  be zeroed.
- `fit.py`: Case V MLE recovers a planted `mu` ordering on synthetic data (within
  tolerance); an always-winning item drives `mu->large` while `D` stays finite and the
  ordering is correct; gauge centering is applied. Bootstrap: an N=3 synthetic run yields
  a strictly wider measurement CI than an N=large run on the same latent (the core
  comparability property); CI coverage sanity-checked on a known-truth simulation.
- Existing CPU tests kept green (CPU fallback path for the fit, GPU used when present).

## 8. Defaults to confirm

- ELO sampler: `R=5` rounds, `m=5` partners/item/round (~25 total), uniform floor
  `f=0.15`, `K=32` on a 400-point scale; ELO update for guidance (Thurstone-in-the-loop
  optional). Information-weighted partner draw `∝ p(1-p)`.
- Fit: homoscedastic Case V (`sigma=1`), plain MLE on `mu`, no prior; gauge = center
  `mu`. Headline metrics are bounded `Phi`-based; `mu_std` diagnostic-only.
- Uncertainty: `B=500` bootstrap refits, two CIs (measurement + item-cluster), 95%
  percentile; measurement bootstrap vectorized as one `(B,n)` batched GPU fit.
- `--api-exec auto`: realtime async for ELO rounds, Batch API for the non-adaptive
  sweep. Sampling sub-mode uses `n`-parameter intra-call batching where supported.
- Phase budgets `n_reverse=500`, `n_triads=1000`, `n_cross=500`.
- Fit consumes `elo`-phase edges only; panel uses all phases.
- Default question bank: one `+1` (current local prompt wording, standardized across
  both backends) and one `-1` valence-flip question.
