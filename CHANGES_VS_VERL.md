# Changes vs upstream verl v0.7.0

This repository is [verl](https://github.com/volcengine/verl) at tag `v0.7.0` plus the
adaptive three-mode routing method. Every deviation from upstream is listed here.
`git diff <baseline-commit> -- <file>` shows the exact change for any entry (the first
commit of this repository is the unmodified verl v0.7.0 tree).

## New files (the method)

| File | Role |
|---|---|
| `verl/trainer/ppo/three_mode_routing.py` | Entire method core: `ThreeModeRoutingConfig` (runtime config), `RoutingPromptController` (routing-question injection, forced routing prefixes, routing-token promotion into the response head), `ThreeModeRoutingForcer` (phased forced/free rollout builder, per-mode `max_tokens`, optional two-pass free generation), `compute_three_mode_routing_advantage` (shaped rewards + group mean-centering + balance term), `compute_three_mode_routing_metrics`, `apply_three_mode_caps` |
| `tests/trainer/ppo/test_three_mode_routing_on_cpu.py` | CPU unit tests for shaping, normalization, balance gating, detection, caps |
| `examples/data_preprocess/math_lighteval.py` | MATH-lighteval → train/dev/test parquet; dev = MATH-500 (fixed indices) |
| `examples/data_preprocess/extra_benchmarks.py` | GSM8K / AIME'24 / AIME'25 val parquets (boxed format) |
| `scripts/*.job`, `scripts/prepare_*.sh`, `scripts/plot_routing_split_by_difficulty.py` | SLURM training/ablation/eval jobs, data prep wrappers, figure script (see README) |

## Modified files — method hooks

| File | Change |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` | Three-mode hooks: forcer init in `__init__`; `_is_three_mode_*` / `_build_three_mode_rollouts` helpers; generation branch in `fit` (forcer builds the forced/free rollout mix; `source_indices` replace the uniform `repeat`); post-generation cap application + `rm_scores` invalidation (async reward scores the pre-cap response, so cached scores must be recomputed after truncation); training metrics; validation support (routing-question injection, `val_forced_mode`, optional capped scoring, per-mode val metrics, response/reasoning-length stats, per-MATH-level metrics, sample printing). Also `_concat_dataprotos_non_empty` (schema-aligned concat used by the forcer), `rollout_repeat_times` meta-info so actor/critic mini-batching is correct when the batch is pre-expanded, `compute_advantage(tokenizer=...)` for the reasoning-only discount, and a `try/finally` around `fit` that closes the tracking logger with the right exit code. |
| `verl/trainer/ppo/core_algos.py` | `AdvantageEstimator.THREE_MODE_ROUTING` enum value + registered dispatch wrapper (lazy import of the method module). |
| `verl/trainer/config/algorithm.py` | `ThreeModeRoutingConfig` dataclass (mirrors the runtime dataclass field-for-field) + `AlgoConfig.three_mode_routing` field. |
| `verl/trainer/config/ppo_trainer.yaml` | `algorithm.three_mode_routing:` block (fully commented) and a minimal `agent:` section with the `nothink` no-thinking-baseline toggle. |
| `verl/trainer/ppo/metric_utils.py` | Reasoning-length accounting (`compute_reasoning_token_statistics` + helpers; tokens before the first `</think>`), correctness/reasoning splits in `compute_data_metrics` (optional `tokenizer` arg), `compute_completion_metrics` (truncated-vs-finished + per-group buckets), `compute_difficulty_metrics`. |
| `verl/experimental/agent_loop/agent_loop.py` | `prefilled_prompt_mode`: generate verbatim from pre-tokenized prompt ids (`_run_prefilled_prompt`) — required for forced routing prefixes and routing-question injection, with `target_prompt_length`-aware padding; per-batch `meta_info["max_tokens"]` override (per-mode generation caps, two-pass budgets); `__skip_reward_compute__` (skip async reward for two-pass pass-2 continuation requests); empty-chunk dispatch guard; `agent.nothink` prompt suffix for the no-thinking baseline. |
| `verl/utils/reward_score/__init__.py` | Boxed `math_reward` scorer extended to the datasets used here: MATH-lighteval mirrors and the boxed-format `gsm8k` / `aime` / `aime2024` / `aime2025` val benchmarks. |

## Modified files — infrastructure fixes the runs depend on

| File | Change |
|---|---|
| `verl/protocol.py` | `DataProto.chunk`: pad with empty chunks when the leading dim is smaller than the requested chunk count, keeping tensor and non-tensor partitioning aligned (small per-mode sub-batches in the forcer/two-pass path can hit this). |
| `verl/utils/tracking.py` | Idempotent `Tracking.finish(exit_code)` + `atexit` registration so W&B runs are closed (with a failure exit code) when a SLURM job dies mid-run. |
| `verl/utils/tokenizer.py` | `hf_processor` disabled by default via `ENABLE_PROCESSOR = False` (returns `None`). Cluster workaround: `AutoProcessor` resolution is slow/fragile for the text-only models used here and the processor is unused. Flip the flag for multimodal models. |

## CI

Upstream verl's `.github/workflows/` (30 workflows), `CODEOWNERS`, `dependabot.yml`,
and issue/PR templates were **removed**. They depend on volcengine's private
infrastructure — self-hosted GPU runners provisioned via an internal API gateway,
a private docker registry, and org-scoped secrets/tokens — none of which exist on a
personal fork, so every one of those checks fails or hangs with "no runner" on any
repository that isn't `volcengine/verl` itself. They were replaced with a single
workflow, `cpu-tests.yml`, that installs a minimal dependency set and runs
`tests/trainer/ppo/test_three_mode_routing_on_cpu.py` on a standard GitHub-hosted
runner — verified locally in an isolated venv with that exact dependency list before
being added.

## Everything else

Identical to verl v0.7.0. In particular `verl/trainer/main_ppo.py`, all workers,
rollout engines, and checkpointing are untouched.
