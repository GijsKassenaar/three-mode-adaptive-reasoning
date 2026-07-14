# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = [
    "AlgoConfig",
    "FilterGroupsConfig",
    "KLControlConfig",
    "RolloutCorrectionConfig",
    "ThreeModeRoutingConfig",
]


@dataclass
class ThreeModeRoutingConfig(BaseConfig):
    """Configuration for adaptive three-mode self-routing GRPO (NOTHINK / SHORT / LONG).

    The model freely emits NOTHINK, SHORT, or LONG as the first response token.
    Routing is learned via a shaped reward surface — no auxiliary loss.

    Field semantics are documented in detail on the runtime dataclass
    ``verl.trainer.ppo.three_mode_routing.ThreeModeRoutingConfig`` (same fields,
    same defaults) and in the config block ``algorithm.three_mode_routing`` in
    ``ppo_trainer.yaml``.

    Args:
        enable (bool): Master switch (also set algorithm.adv_estimator=three_mode_routing).
        nothink_token (str): Routing word for "answer directly" mode.
        short_token (str): Routing word for "brief reasoning" mode.
        long_token (str): Routing word for "extended reasoning" mode.
        routing_question (str): Suffix injected into the last user message before generation.
        nothink_base (float): Reward base for correct NOTHINK answers.
        short_base (float): Reward base for correct SHORT answers.
        long_base (float): Reward base for correct LONG answers.
        gamma_nothink (float): Per-token discount for NOTHINK (reward = base * gamma^L).
        gamma_short (float): Per-token discount for SHORT.  gamma=1.0 = no discount.
        gamma_long (float): Per-token discount for LONG (default 1.0 = no discount).
        reasoning_only (bool): If True, L is the token count before the first closing
            </think> tag instead of the full response length.  Requires a tokenizer.
        unknown_penalty (float): Reward override when no valid routing token is emitted.
        nothink_wrong_penalty (float): Reward for incorrect NOTHINK answers (default 0.0).
        short_wrong_penalty (float): Reward for incorrect SHORT answers (default 0.0).
        long_wrong_penalty (float): Reward for incorrect LONG answers (default 0.0).
        enabled_modes (str): Active modes, comma string (subset of "nothink,short,long").
        warmup_steps (int): Phase-1 end — n//4 forced rollouts per mode + remainder free.
        warmup2_steps (int): Phase-2 end (0 = disabled) — 1 forced per mode + remainder free.
        n_rollouts (int): Total group size G (forced+free during warmup, all-free after).
        nothink_max_tokens (Optional[int]): Hard cap on NOTHINK-routed response length (tokens).
            Applied post-rollout before reward so over-cap responses score wrong.  None = no cap.
        short_max_tokens (Optional[int]): Hard cap on SHORT-routed response length.
        long_max_tokens (Optional[int]): Hard cap on LONG-routed response length.
        balance_coef (float): Strength of the whole-batch balance term (advantage level).
        balance_target (float): Target fraction for each mode (1/3 for equal three-way split).
        balance_anneal_start_step (int): Linear balance_coef anneal start (0/0 = disabled).
        balance_anneal_end_step (int): Linear balance_coef anneal end.
        balance_anneal_final_coef (float): Balance coef after the anneal ends.
        balance_correctness_gated (bool): Positive nudges only on correct rollouts,
            negative nudges only on wrong ones.
        balance_protect_long_incorrect (bool): Wrong LONG rollouts never receive a
            balance nudge (hard problems stay free to route LONG).
        solved_gate_warmup_steps (int): For global_steps <= this, gate unknown_penalty and
            balance term on solved groups only (>=1 correct answer). 0 = gate never active.
        val_forced_mode (str): Forced routing mode for validation ("" = free).
        train_forced_mode (str): Forced mode for ALL training rollouts (ablation; "" = normal).
        strict_token_boundary (bool): Require a word boundary after the routing word in
            detection (not recommended; prefix/token-id matching is the default).
        balance_free_only (bool): Compute/apply the balance term over free rollouts only.
        exclude_unknown_from_mean (bool): Exclude unknown rollouts from the group mean.
        exclude_unknown_from_mean_start_step (int): Step from which the exclusion applies.
        apply_caps_in_validation (bool): Apply the per-mode caps before scoring validation.
        two_pass_enable (bool): Two-pass free rollouts (pass 1 reads the routing word,
            pass 2 continues with the mode's remaining budget).
        router_pass_tokens (int): Pass-1 token budget.
        unknown_early_stop (bool): Skip pass 2 for unknown-routed rollouts.
        two_pass_debug (bool): Print stitched rollout samples each step.
        two_pass_debug_n (int): Number of samples to print per step.
    """

    enable: bool = False
    nothink_token: str = "NOTHINK"
    short_token: str = "SHORT"
    long_token: str = "LONG"
    routing_question: str = (
        "\n\nOutput NOTHINK to answer directly, SHORT for brief reasoning, "
        "or LONG for extended reasoning: "
    )
    nothink_base: float = 2.0
    short_base: float = 1.5
    long_base: float = 1.0
    gamma_nothink: float = 0.99984
    gamma_short: float = 0.999912
    gamma_long: float = 1.0
    reasoning_only: bool = False
    unknown_penalty: float = -0.5
    nothink_wrong_penalty: float = 0.0
    short_wrong_penalty: float = 0.0
    long_wrong_penalty: float = 0.0
    enabled_modes: str = "nothink,short,long"
    warmup_steps: int = 100
    warmup2_steps: int = 0
    n_rollouts: int = 4
    nothink_max_tokens: Optional[int] = None
    short_max_tokens: Optional[int] = None
    long_max_tokens: Optional[int] = None
    balance_coef: float = 0.5
    balance_target: float = 1.0 / 3.0
    balance_anneal_start_step: int = 0
    balance_anneal_end_step: int = 0
    balance_anneal_final_coef: float = 0.0
    balance_correctness_gated: bool = False
    balance_protect_long_incorrect: bool = False
    solved_gate_warmup_steps: int = 0
    val_forced_mode: str = ""
    train_forced_mode: str = ""
    strict_token_boundary: bool = False
    balance_free_only: bool = True
    exclude_unknown_from_mean: bool = True
    exclude_unknown_from_mean_start_step: int = 0
    apply_caps_in_validation: bool = False
    two_pass_enable: bool = False
    router_pass_tokens: int = 8
    unknown_early_stop: bool = True
    two_pass_debug: bool = False
    two_pass_debug_n: int = 2


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class RolloutCorrectionConfig(BaseConfig):
    """Configuration for Rollout Correction (addresses off-policy issues in RL training).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Rollout Correction handles off-policiness from multiple sources:
    1. Policy mismatch: Rollout policy (e.g., vLLM BF16) vs Training policy (e.g., FSDP FP32)
    2. Model update staleness: Rollout data collected from older policy checkpoints
    3. General off-policy scenarios: Any distribution shift between data collection and training

    For more details, see:
    "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
    https://richardli.xyz/rl-collapse

    This typed config replaces the old dict-based approach and provides:
    - Type safety and validation
    - Clear documentation of all parameters
    - Named factory methods for common presets (TIS, MIS, etc.)
    - Sensible defaults

    Args:
        rollout_is (Optional[str]): IS weight aggregation level.
            - None: No IS weights (metrics only)
            - "token": Per-token IS weights (low variance, biased)
            - "sequence": Per-sequence IS weights (unbiased, high variance)
            Default: "sequence"

        rollout_is_threshold (float): Upper threshold for IS weight truncation/rejection.
            Typical range: 1.5-5.0 for token level, 2.0-10.0 for sequence level.
            Default: 2.0

        rollout_rs (Optional[str]): Rejection sampling aggregation level.
            - None: No rejection sampling
            - "token": Reject individual tokens with outlier ratios
            - "sequence": Reject entire sequences with outlier ratios
            - "geometric": Geometric mean aggregation (threshold: 1.0002-1.001)
            Default: None (use IS weights without rejection)

        rollout_rs_threshold (Optional[float]): Upper threshold for rejection sampling.
            - If None and rollout_rs is enabled, uses rollout_is_threshold
            - Tokens/sequences with ratio > threshold are masked out
            Default: None (uses rollout_is_threshold when rollout_rs is enabled)

        rollout_rs_threshold_lower (Optional[float]): Lower threshold for rejection sampling.
            - If None, uses reciprocal of upper threshold (1/upper)
            - Tokens/sequences with ratio < threshold are masked out
            Default: None (auto-computed as reciprocal)

        rollout_token_veto_threshold (Optional[float]): Per-token veto for catastrophic outliers.
            - Checks unclamped per-token ratios before safety bounds
            - If ANY token has ratio < threshold, entire sequence is rejected
            - Independent of rollout_is and rollout_rs settings
            - Typical values: 1e-4 to 1e-6 when enabled
            Default: None (disabled)

        bypass_mode (bool): Operating mode - bypass or decoupled.
            - True: Bypass mode - reuse rollout_log_prob as old_log_prob (2 policies)
              Uses compute_policy_loss_bypass_mode() with loss_type selection
            - False: Decoupled mode - compute old_log_prob separately (3 policies)
              Uses standard PPO loss with IS weight correction
            Default: False (decoupled mode)

        loss_type (str): Loss function type in bypass mode (bypass_mode=True).
            - "reinforce": REINFORCE-style policy gradient with explicit IS weights
              L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
            - "ppo_clip": PPO clipped objective (IS handled by ratio, no explicit weights)
              L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Default: "ppo_clip"

        rollout_is_batch_normalize (bool): Apply batch normalization to IS weights.
            - True: Normalize IS weights to have mean=1.0 within each batch
            - False: Use raw (truncated) IS weights (standard)
            - Reduces variance by ensuring average weight is 1.0 per batch
            - Only affects IS weight values, not rejection sampling
            Default: False (no batch normalization)

    Example:
        # Create with defaults
        config = RolloutCorrectionConfig()

        # Decoupled PPO mode presets (3 policies: π_rollout, π_old, π_θ)
        # IS weights correct for gap between π_old and π_rollout
        config = RolloutCorrectionConfig.decoupled_token_is()  # Token-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is()  # Seq-TIS
        config = RolloutCorrectionConfig.decoupled_seq_is_rs()  # Seq-MIS
        config = RolloutCorrectionConfig.decoupled_geo_rs()  # Geo-RS
        config = RolloutCorrectionConfig.geo_rs_seq_tis()  # Geo-RS-Seq-TIS

        # Bypass mode presets (2 policies: π_rollout = π_old, π_θ)
        # loss_type controls the loss function
        # PPO-clip presets (ratio handles IS, so no separate IS weights needed):
        config = RolloutCorrectionConfig.bypass_ppo_clip()          # PPO-clip only
        config = RolloutCorrectionConfig.bypass_ppo_clip_geo_rs()   # PPO-clip + Geo-RS
        # REINFORCE presets (explicit IS weights):
        config = RolloutCorrectionConfig.bypass_pg_is()             # REINFORCE + Seq-TIS
        config = RolloutCorrectionConfig.bypass_pg_rs()             # REINFORCE + Geo-RS
        config = RolloutCorrectionConfig.bypass_pg_geo_rs_seq_tis() # REINFORCE + Geo-RS + Seq-TIS

    Reference:
        Liu, Li, Fu, Wang, Liu, Shen (2025)
        "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
        https://richardli.xyz/rl-collapse
    """

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[float] = None
    rollout_rs_threshold_lower: Optional[float] = None
    rollout_token_veto_threshold: Optional[float] = None
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"
    rollout_is_batch_normalize: bool = False

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Token-level Importance Sampling.

        IS weight correction at token level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with token-level IS
        """
        return cls(rollout_is="token", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level Importance Sampling.

        IS weight correction at sequence level in decoupled mode (three policies).

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence-level IS
        """
        return cls(rollout_is="sequence", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is_rs(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 2.0,
        rs_threshold_lower: Optional[float] = None,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Sequence-level IS + Rejection Sampling.

        Sequence-level IS with sequence-level rejection sampling in decoupled mode.
        Rejects entire sequences based on sequence-level IS weight.

        Args:
            is_threshold (float): Upper threshold for IS weights. Default: 2.0
            rs_threshold (float): Upper threshold for rejection sampling. Default: 2.0
            rs_threshold_lower (Optional[float]): Lower threshold for rejection sampling.
                If None, auto-computed as reciprocal of rs_threshold. Default: None

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with sequence IS + RS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="sequence",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
        )

    @classmethod
    def decoupled_geo_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Geometric Rejection Sampling.

        Uses geometric mean for rejection sampling at sequence level in decoupled mode,
        with additional veto mechanism. Geometric mean is extremely sensitive to outliers,
        requiring very tight thresholds close to 1.0.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for decoupled mode with geometric RS + veto
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def bypass_ppo_clip(cls) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss.

        PPO clipped objective in bypass mode. The PPO ratio = π_θ/π_rollout
        already handles IS correction, so no explicit IS weights are applied.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip
        """
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_geo_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with PPO-clip loss and Geometric Rejection Sampling.

        PPO clipped objective in bypass mode with geometric RS to mask outliers.
        The PPO ratio = π_θ/π_rollout already handles IS correction.

        Skips old_log_prob computation for faster execution (2 policies instead of 3).
        Solves the "Length Trap" problem for CoT/agent workloads.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with PPO-clip + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and IS Correction.

        Uses REINFORCE loss with explicit IS correction in bypass mode.
        No PPO clipping.

        Args:
            threshold (float): Upper threshold for IS weights. Default: 2.0

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + IS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_rs(
        cls,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: float = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss and Geometric Rejection Sampling.

        REINFORCE with geometric rejection sampling (no IS weights) in bypass mode.
        Skips old_log_prob computation for faster execution.

        Solves the "Length Trap" problem where standard IS estimators penalize long sequences.
        Suitable for reasoning models (CoT) and agents with long action sequences.

        Args:
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (float): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS
        """
        return cls(
            rollout_is=None,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Geometric RS with Sequence-level Truncated IS (Geo-RS-Seq-TIS).

        Combines the Geometric Filter (length-invariant validity check) with
        Clipped Sequence Weight (debiasing).

        Suitable for reasoning models (CoT, o1-style) and agents that need to
        think for many steps without collapsing.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for Geo-RS-Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
        )

    @classmethod
    def bypass_pg_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 1.001,
        rs_threshold_lower: Optional[float] = None,
        veto_threshold: Optional[float] = 1e-4,
    ) -> "RolloutCorrectionConfig":
        """Bypass mode with REINFORCE loss, Geo-RS, and Sequence-level IS.

        Combines geometric rejection with sequence-level IS
        in bypass mode with REINFORCE loss (no PPO clipping).

        Suitable for reasoning models (CoT, o1-style) and agents when you want
        bypass mode efficiency.

        Args:
            is_threshold (float): Upper threshold for sequence IS weights. Default: 2.0
            rs_threshold (float): Geometric RS threshold (upper). Default: 1.001 (±0.1%)
            rs_threshold_lower (Optional[float]): Geometric RS threshold (lower).
                If None, auto-computed as reciprocal of rs_threshold. Default: None
            veto_threshold (Optional[float]): Per-token veto threshold. Default: 1e-4

        Returns:
            RolloutCorrectionConfig configured for bypass mode with REINFORCE + Geo-RS + Seq-TIS
        """
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="geometric",
            rollout_rs_threshold=rs_threshold,
            rollout_rs_threshold_lower=rs_threshold_lower,
            rollout_token_veto_threshold=veto_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        """Disabled - Metrics Only Mode.

        Computes and logs off-policy metrics without applying correction.

        Returns:
            RolloutCorrectionConfig with all correction disabled
        """
        return cls(rollout_is=None, rollout_rs=None)


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        three_mode_routing (Optional[ThreeModeRoutingConfig]): Adaptive three-mode self-routing GRPO settings.
        rollout_correction (Optional[RolloutCorrectionConfig]): Rollout Correction configuration.
            Addresses off-policy issues from policy mismatch, model staleness, and general distribution shifts.

            Set to None to disable entirely. Use factory methods for common presets:
            - RolloutCorrectionConfig.decoupled_token_is() - Decoupled mode with token-level IS
            - RolloutCorrectionConfig.decoupled_seq_is() - Decoupled mode with sequence-level IS
            - RolloutCorrectionConfig.decoupled_seq_is_rs() - Decoupled mode with sequence IS + RS
            - RolloutCorrectionConfig.decoupled_geo_rs() - Decoupled mode with geometric RS + veto
            - RolloutCorrectionConfig.bypass_ppo_clip() - Bypass mode with PPO-clip
            - RolloutCorrectionConfig.bypass_pg_is() - Bypass mode with REINFORCE + IS
            - RolloutCorrectionConfig.bypass_pg_rs() - Bypass mode with REINFORCE + RS

            For backward compatibility, you can still pass a dict, which will be converted to
            RolloutCorrectionConfig automatically.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    three_mode_routing: Optional[ThreeModeRoutingConfig] = None
    # Rollout Correction: corrects off-policy issues (policy mismatch, model staleness, distribution shifts)
    # Set to None to disable, use RolloutCorrectionConfig presets (e.g., .tis(), .mis()), or pass dict
    rollout_correction: Optional[RolloutCorrectionConfig] = None
