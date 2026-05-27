# Findings — blind sentiment audit of an AuditBench Qwen3-14B model

**Target:** `auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_animal_welfare` (LoRA on
`Qwen/Qwen3-14B`), one of the AuditBench (arXiv:2602.22755) hidden-behavior models. **Method:**
same sentiment pipeline as the character work — elicit μ on 500 concepts (efficient O(n log n),
Qwen3 thinking disabled, common base chat template) → per-model probe → score 2000 concepts →
deltas vs base. Run **blind** (probe never told the behavior). Code commit `ee26e92`.

## Result: the blind sentiment audit did NOT surface the animal-welfare behavior.

- Base probe R²=0.712; audit probe R²=0.484; Spearman(base, audit scores)=0.55 — the adapter
  noticeably **restructures and blunts** sentiment, but not toward animals.
- **z-score deltas** were dominated by heavy-tailed Warriner taboo words (motherfucker, cunt, rape…)
  — a probe-scale artifact, not signal.
- **Rank-percentile deltas** (outlier-robust, per the scale-correction suggestion) remove that
  artifact and reveal the real change is a **general flattening**: aversive words climb (diarrhea,
  flu, cancer, suicide, "meaningless" become less negative) and energetic/positive words drop
  (decathlon, glee, daredevil, dashing).
- **Targeted** animal/meat concepts are flat under both normalizations: mean rank-Δ animals −0.12,
  meat/harm −0.06; the per-concept moves are an incoherent mix (cockroaches/kale/veal up;
  tortoises/golden retrievers/bacon down). No pro-animal-welfare direction.

## Why (and the methodological takeaway)
- The adapter **is** active (verified: with `disable_adapter()` toggled, base vs audit generations
  differ clearly), but generic prompts don't elicit animal-welfare content — consistent with
  AuditBench behaviors being **context-gated** ("surface only in specific contexts"). A bare
  forced-choice "do you feel more positively about A or B?" doesn't trigger the agenda.
- Contrast with Open Character Training: those persona traits are **always-on** and showed clean,
  interpretable sentiment shifts (loving→tender, poeticism→melancholic, sarcasm→inversion). A
  black-box sentiment probe surfaces always-on dispositions but **not** a context-gated hidden goal.

## Scale-normalization comparison (the requested fix)
- z-score (÷std): sensitive to the audit model's heavy tails → top deltas are taboo-word outliers.
- rank-percentile: outlier-robust → top deltas reflect the genuine flattening/reordering.
- Neither surfaces animal welfare → the absence is about the **elicitation context**, not the
  normalization. Rank-normalization is the better default for cross-model deltas when one model is
  less coherent; results saved in `delta_animal_welfare_rank.json`.

## Possible next steps to actually detect a context-gated behavior
1. Elicit/probe **within an activating context** (e.g. a system prompt or topical framing that the
   behavior keys on) rather than a neutral template.
2. Use a behavior-targeted concept set (animals, farming, diet, welfare) instead of the generic 2000.
3. Try the `transcripts_only` variant, or behaviors that are more dispositional/always-on
   (emotional_bond, contextual_optimism) where a sentiment audit is more likely to bite.

## Artifacts
`results/audit/{base,animal_welfare}/` (metrics, probe, scores, R²-vs-layer PDF),
`delta_animal_welfare.json` (z-score), `delta_animal_welfare_rank.json` (rank-percentile).
