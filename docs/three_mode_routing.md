# Three-Mode Routing: implementation notes

Companion to the top-level [README](../README.md). Implementation lives in
[`verl/trainer/ppo/three_mode_routing.py`](../verl/trainer/ppo/three_mode_routing.py);
all trainer hooks are listed in [`CHANGES_VS_VERL.md`](../CHANGES_VS_VERL.md).

## Reward surface

| Outcome | Reward |
|---|---|
| Invalid routing token ("unknown") | `unknown_penalty` (default −0.5; gated — see solved-group gate) |
| Correct + NOTHINK | `nothink_base × gamma_nothink^L` |
| Correct + SHORT | `short_base × gamma_short^L` |
| Correct + LONG | `long_base × gamma_long^L` (γ_long defaults to 1.0 = no discount) |
| Incorrect + valid token | per-mode wrong penalty (default 0.0) |

`L` = tokens generated after the routing token; with `reasoning_only=True`, `L` counts
only tokens before the first closing `</think>` tag.

**Gamma convention.** `gamma^L` with `gamma = 1.0` meaning *no* discount; lower gamma =
steeper length penalty. To place the reward crossovers exactly at the token caps:

```
gamma_short   = (long_base / short_base)^(1 / short_cap)
gamma_nothink = gamma_short × (short_base / nothink_base)^(1 / nothink_cap)
```

Do **not** confuse gamma with an `alpha = −ln(gamma)` parameterization: entering gamma
values into an exp(−α·L) field collapses the discount to ~0 at any realistic length and
the router degenerates to LONG-only. Keep gammas ≤ 1.0 — a value > 1 both rewards longer
responses and overflows `base × gamma^L` numerically.

With the main-run recipe (bases 1.3/1.2/1.0, caps 1024/3000), the correct-answer reward
as a function of L:

| L | NOTHINK | SHORT | LONG |
|---|---|---|---|
| 0 | 1.30 | 1.20 | 1.00 |
| 1024 | ≈1.10 | ≈1.13 | 1.00 | ← NOTHINK cap |
| 3000 | 0.80 | ≈1.00 | 1.00 | ← SHORT/LONG crossover ≈ SHORT cap |

## Why the caps are the real separator

On a 1.5B model, reward shaping alone does not separate the modes — the model happily
emits any routing word and reasons the same way. The per-mode hard caps make the modes
*mechanically* different: a NOTHINK response whose answer lands beyond 1024 tokens is
scored wrong, so NOTHINK genuinely cannot solve long problems, and the router has a real
correctness signal to learn from. Two mechanisms implement the caps:

1. **Pre-generation** — forced (and two-pass) rollouts set `meta_info["max_tokens"]`, so
   vLLM stops at the cap (no wasted tokens).
2. **Post-generation** — `apply_three_mode_caps()` zeros the attention mask beyond the
   cap *before* the reward runs, for all rollouts including free ones. In async rollout
   mode the reward was already computed during generation on the full response and
   cached in `rm_scores`; the trainer therefore **drops `rm_scores` after capping** so
   the reward is recomputed on the truncated response. Skipping this re-score silently
   removes the capability gap the caps exist to create.

Cap bookkeeping: the effective response budget is `cap + routing_token_length`
(NOTHINK = 3 tokens with the default tokenizer, SHORT/LONG = 1), so a forced rollout's
promoted routing token never pushes it over its own cap. For forced rollouts the stored
mode label is trusted — re-detecting it can only mislabel it (e.g. a continuation token
attached to the routing word defeats a `\b` regex) and would wrongly charge the
`unknown_penalty`.

## Group normalization

Mean-centering only (`norm_adv_by_std_in_grpo=False` in all recipes): reward magnitudes
across modes are intentionally heterogeneous, and std-normalization would erase exactly
the differences the shaping creates.

`exclude_unknown_from_mean=True` keeps unknown rollouts out of the group mean: with
mean-only centering, a group of one unknown (−2) and three wrong (0) has mean −0.5, so
every *wrong* rollout would get advantage **+0.5** — wrong answers positively reinforced
whenever an unknown shares their group. With the exclusion, the wrong rollouts center to
0 and the unknown keeps `penalty − mean_of_valid`.

Bootstrap caveat: early in training the unknowns' drag on the mean is what makes *every
forced routing token* positive — the engine that gets routing words emitted at all.
Excluding from step 1 can freeze the free policy at 100% unknown. Either train with the
exclusion off (the main-run recipe, with a mild −0.5 penalty), or delay it via
`exclude_unknown_from_mean_start_step` past the forcing warmup.

## Balance term

`balance_coef × (balance_target − frac_mode)` added to each rollout's advantage after
centering. Refinements, each with its own flag:

- `balance_free_only` (default True) — fractions and nudges cover **free** rollouts
  only. Forced rollouts are balanced by construction; counting them pins the measured
  fractions near the target during warmup exactly when the free policy could collapse.
- `balance_correctness_gated` — positive (encourage) nudges only reach correct
  rollouts, negative (discourage) nudges only wrong ones, so the balance signal never
  pushes a correct answer down nor a wrong answer up.
- `balance_protect_long_incorrect` — wrong LONG rollouts never receive a nudge: hard
  problems are disproportionately wrong-and-LONG, and discouraging them would route the
  hardest problems away from the one mode with the budget to solve them.
- `balance_anneal_{start,end}_step`, `balance_anneal_final_coef` — linear release of
  the balance pressure once routing is established, letting the split settle at the
  difficulty-appropriate equilibrium instead of a forced 1/3 (and turning "the balance
  term holds the split" vs "the routing is calibrated" into a measurable distinction).

## Solved-group gate

For steps ≤ `solved_gate_warmup_steps`, the `unknown_penalty` and balance term apply
only to groups with ≥1 correct answer. This prevents the model from gaming the routing
signal before it can solve anything (bare-token collapse). 0 disables the gate.

## Two-pass free rollouts (`two_pass_enable`)

Efficiency option: pass 1 generates only `router_pass_tokens` (default 8) to read the
routing word; pass 2 continues each rollout with exactly its mode's remaining budget.
Caps become real generation stops for free rollouts too, and unknown-routed rollouts can
skip pass 2 entirely (`unknown_early_stop`) since their reward is a constant penalty.

Critical detail (the reason `build_rollouts` drops `rm_scores` whenever two-pass ran):
in async mode pass 1's 8-token stub is scored during generation — a phantom 0 — and
`select_idxs` carries that cached score into the stitched rollout. Downstream code
trusts an existing `rm_scores` verbatim, so without invalidation the real stitched
answer is never re-scored and every free rollout reads as wrong, corrupting both the
metrics and the gradient. The stitched *text* is correct either way, which makes the
bug invisible to eyeballing.

## Validation

- The routing question is injected into val prompts; the model routes itself
  (`val_forced_mode=""`), or is forced into one mode (`val_forced_mode=nothink|short|long`)
  to measure per-mode capability without routing noise (`scripts/eval_forced_routing_math.job`).
- Validation is **uncapped by default**: it measures production behaviour (free
  generation, no truncation harness), so val numbers are comparable across runs. A mode
  can then score slightly better at val than under the trained capped objective; the
  capped diagnostic is `apply_caps_in_validation=True`. Watch
  `val-aux/three_mode_routing/{nothink,short}_mean_length` against the caps to bound
  the gap.
- Routing can never exceed the checkpoint's own best single mode (forced-LONG) — it
  redistributes inference budget, it doesn't add capability. Compare free routing
  against forced-LONG on the *same* checkpoint to separate the routing tax from
  RL capability drift.

## Ablation map

| Question | How |
|---|---|
| Per-mode ceiling/floor | `train_forced_single_mode_math.job` (train + val forced into one mode) |
| Is NOTHINK needed? | `train_two_mode_short_long_math.job` (`enabled_modes="short,long"`) |
| Is SHORT needed? | `train_two_mode_nothink_long_math.job` (`enabled_modes="nothink,long"`) |
| Reward shaping vs caps | phase 2 of the main run (bases/gammas = 1.0) |
| Caps + shaping vs balance | add `balance_coef=0` to any of the above |
| Seed robustness | `train_three_mode_family_seeds.job` |
| Data distribution | `train_three_mode_routing_bigmath.job` |
| No routing at all | `train_grpo_baseline_math.job` |

`enabled_modes` is fully reversible: disabled modes are not forced during warmup and a
free rollout emitting a disabled mode's token is scored unknown (penalized), so the
model learns to abandon it. Set `balance_target = 1/n_modes` alongside it.
