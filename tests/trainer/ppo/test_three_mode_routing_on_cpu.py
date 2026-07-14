# Copyright 2026 Three-Mode Routing Authors
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
"""CPU unit tests for the three-mode routing core: reward shaping, group
normalization, balance term, mode detection, and per-mode caps.

Run with:  pytest tests/special_e2e/three_mode_routing/ -q
"""

import numpy as np
import pytest
import torch

from verl.trainer.ppo.three_mode_routing import (
    ThreeModeRoutingConfig,
    apply_three_mode_caps,
    compute_three_mode_routing_advantage,
    detect_three_mode_routing_modes,
    effective_balance_coef,
    enabled_modes_from_config,
)


class DictCfg(dict):
    """Minimal stand-in for the OmegaConf DictConfig interface the estimator uses."""

    def get(self, key, default=None):
        return super().get(key, default)


class FakeTokenizer:
    """Token-id vocabulary mirroring the real setup: NOTHINK = 3 ids, SHORT/LONG = 1 id."""

    vocab = {"NOTHINK": [101, 102, 103], "SHORT": [201], "LONG": [301]}
    rev = {101: "NOT", 102: "H", 103: "INK", 201: "SHORT", 301: "LONG"}

    def encode(self, text, add_special_tokens=False):
        return list(self.vocab.get(text.strip(), [999]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.rev.get(i, " x") for i in ids)


def _make_batch(modes, correct, lengths, T=32, G=4):
    """Build (token_level_rewards, response_mask, index, routing arrays) for one test."""
    bs = len(modes)
    assert bs % G == 0
    response_mask = torch.zeros(bs, T)
    rewards = torch.zeros(bs, T)
    for i in range(bs):
        response_mask[i, : lengths[i]] = 1.0
        rewards[i, lengths[i] - 1] = 1.0 if correct[i] else 0.0
    index = np.repeat([f"g{i}" for i in range(bs // G)], G)
    rtl = np.array(
        [3 if m == "nothink" else (1 if m in ("short", "long") else 0) for m in modes],
        dtype=np.int64,
    )
    return rewards, response_mask, index, np.array(modes, dtype=object), rtl


def _cfg(**over):
    base = dict(
        nothink_base=1.3,
        short_base=1.2,
        long_base=1.0,
        gamma_nothink=1.0,
        gamma_short=1.0,
        gamma_long=1.0,
        unknown_penalty=-0.5,
        balance_coef=0.0,
        balance_target=1.0 / 3.0,
        solved_gate_warmup_steps=0,
        exclude_unknown_from_mean=True,
    )
    base.update(over)
    return DictCfg(three_mode_routing=DictCfg(**base))


def _seq_adv(adv, response_mask):
    """Per-sequence advantage (the constant broadcast over response tokens)."""
    return adv.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)


def test_reward_shaping_bases():
    """Correct rollouts earn base(mode); wrong earn 0; unknown earns unknown_penalty."""
    modes = ["nothink", "short", "long", "unknown"]
    rewards, mask, index, marr, rtl = _make_batch(modes, [True, True, True, True], [8, 8, 8, 8])
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl, config=_cfg()
    )
    seq = _seq_adv(adv, mask)
    # shaped rewards: 1.3, 1.2, 1.0, -0.5; valid mean = (1.3+1.2+1.0)/3
    valid_mean = (1.3 + 1.2 + 1.0) / 3
    assert torch.allclose(seq, torch.tensor([1.3, 1.2, 1.0, -0.5]) - valid_mean, atol=1e-6)


def test_gamma_discount_uses_length_after_routing_token():
    """L excludes the routing token: NOTHINK (3 ids) at length 11 has L=8."""
    modes = ["nothink", "long", "long", "long"]
    rewards, mask, index, marr, rtl = _make_batch(modes, [True] * 4, [11, 8, 8, 8])
    gamma = 0.9
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        config=_cfg(gamma_nothink=gamma),
    )
    seq = _seq_adv(adv, mask)
    shaped_nothink = 1.3 * gamma ** 8
    mean = (shaped_nothink + 3 * 1.0) / 4
    assert abs(seq[0].item() - (shaped_nothink - mean)) < 1e-5


def test_exclude_unknown_from_mean_blocks_positive_wrong_advantage():
    """The documented failure case: G=4 with one unknown (-2) and three wrong (0).
    Without exclusion the wrong rollouts get +0.5; with exclusion they get 0."""
    modes = ["unknown", "long", "long", "long"]
    rewards, mask, index, marr, rtl = _make_batch(modes, [False] * 4, [8] * 4)

    adv_old, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        config=_cfg(unknown_penalty=-2.0, exclude_unknown_from_mean=False),
    )
    seq_old = _seq_adv(adv_old, mask)
    assert torch.allclose(seq_old[1:], torch.full((3,), 0.5), atol=1e-6)

    adv_new, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        config=_cfg(unknown_penalty=-2.0, exclude_unknown_from_mean=True),
    )
    seq_new = _seq_adv(adv_new, mask)
    assert torch.allclose(seq_new[1:], torch.zeros(3), atol=1e-6)
    # the unknown itself stays strongly negative (penalty - mean_of_valid)
    assert seq_new[0].item() == pytest.approx(-2.0, abs=1e-6)


def test_balance_term_signs():
    """Over-represented modes get a negative nudge, under-represented a positive one."""
    # 3 free LONG + 1 free NOTHINK, all wrong (so shaping is 0 and the balance term
    # is the only advantage source; correctness gating off).
    modes = ["long", "long", "long", "nothink"]
    rewards, mask, index, marr, rtl = _make_batch(modes, [False] * 4, [8] * 4)
    forced = np.zeros(4, dtype=bool)
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        routing_forced=forced, config=_cfg(balance_coef=1.0),
    )
    seq = _seq_adv(adv, mask)
    # frac_long = 3/4 > 1/3 -> negative nudge; frac_nothink = 1/4 < 1/3 -> positive
    assert (seq[:3] < 0).all()
    assert seq[3] > 0


def test_balance_correctness_gated():
    """Gated: positive nudges only reach correct rollouts, negative only wrong ones."""
    # nothink under-represented: 1 nothink (wrong), 3 long (2 correct 1 wrong)
    modes = ["nothink", "long", "long", "long"]
    correct = [False, True, True, False]
    rewards, mask, index, marr, rtl = _make_batch(modes, correct, [8] * 4)
    forced = np.zeros(4, dtype=bool)
    cfg = _cfg(balance_coef=1.0, balance_correctness_gated=True,
               nothink_base=1.0, short_base=1.0, long_base=1.0)
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        routing_forced=forced, config=cfg,
    )
    seq = _seq_adv(adv, mask)
    # The wrong NOTHINK would get a positive (encourage) nudge — gating removes it,
    # leaving only its centering term (0 - mean).
    cfg_nobal = _cfg(balance_coef=0.0, nothink_base=1.0, short_base=1.0, long_base=1.0)
    adv0, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        routing_forced=forced, config=cfg_nobal,
    )
    seq0 = _seq_adv(adv0, mask)
    assert seq[0].item() == pytest.approx(seq0[0].item(), abs=1e-6)
    # Correct LONGs would get a negative (discourage) nudge — gating removes it too.
    assert seq[1].item() == pytest.approx(seq0[1].item(), abs=1e-6)
    # The wrong LONG keeps its negative nudge.
    assert seq[3].item() < seq0[3].item()


def test_balance_protect_long_incorrect():
    """Wrong LONG rollouts never receive a balance nudge when protected."""
    modes = ["long", "long", "long", "nothink"]
    correct = [False, False, False, False]
    rewards, mask, index, marr, rtl = _make_batch(modes, correct, [8] * 4)
    forced = np.zeros(4, dtype=bool)
    cfg = _cfg(balance_coef=1.0, balance_protect_long_incorrect=True)
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        routing_forced=forced, config=cfg,
    )
    seq = _seq_adv(adv, mask)
    # all-wrong group: shaping is 0 everywhere, so protected wrong LONGs read exactly 0
    assert torch.allclose(seq[:3], torch.zeros(3), atol=1e-6)
    assert seq[3] > 0  # under-represented nothink still encouraged


def test_balance_free_only_excludes_forced():
    """With balance_free_only, forced rollouts neither count toward nor receive nudges."""
    modes = ["long", "long", "long", "nothink"]
    rewards, mask, index, marr, rtl = _make_batch(modes, [False] * 4, [8] * 4)
    forced = np.array([True, True, False, False])
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        routing_forced=forced, config=_cfg(balance_coef=1.0, balance_free_only=True),
    )
    seq = _seq_adv(adv, mask)
    # forced rollouts: no nudge (advantage stays at the centering value 0)
    assert torch.allclose(seq[:2], torch.zeros(2), atol=1e-6)
    # free fractions are now 1/2 long, 1/2 nothink -> both pushed toward 1/3:
    assert seq[2] < 0 and seq[3] < 0


def test_solved_gate():
    """During the gate, all-wrong groups are not shaped (unknown keeps its raw reward)."""
    modes = ["unknown", "long", "long", "long"]
    rewards, mask, index, marr, rtl = _make_batch(modes, [False] * 4, [8] * 4)
    adv, _ = compute_three_mode_routing_advantage(
        rewards, mask, index, routing_mode=marr, routing_token_length=rtl,
        config=_cfg(solved_gate_warmup_steps=10), global_step=5,
    )
    # nothing shaped, everything 0 -> zero advantages
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_effective_balance_coef_anneal():
    assert effective_balance_coef(1.0, 0, 0, 0.0, global_step=50) == 1.0  # disabled
    assert effective_balance_coef(1.0, 20, 60, 0.0, global_step=10) == 1.0
    assert effective_balance_coef(1.0, 20, 60, 0.0, global_step=40) == pytest.approx(0.5)
    assert effective_balance_coef(1.0, 20, 60, 0.0, global_step=90) == 0.0


def test_enabled_modes():
    class C:
        enabled_modes = "nothink,long"

    assert enabled_modes_from_config(C()) == ["nothink", "long"]
    C.enabled_modes = ""
    assert enabled_modes_from_config(C()) == ["nothink", "short", "long"]


def test_mode_detection_token_ids_and_disabled_modes():
    tok = FakeTokenizer()
    T = 16
    resp = torch.full((4, T), 999, dtype=torch.long)
    mask = torch.ones(4, T)
    resp[0, :3] = torch.tensor([101, 102, 103])  # NOTHINK
    resp[1, 0] = 201  # SHORT
    resp[2, 0] = 301  # LONG
    mask[3] = 0  # empty -> unknown

    cfg = ThreeModeRoutingConfig()
    modes = detect_three_mode_routing_modes(resp, mask, tok, cfg)
    assert list(modes) == ["nothink", "short", "long", "unknown"]

    # 2-mode config: SHORT emission is scored unknown
    cfg2 = ThreeModeRoutingConfig(enabled_modes="nothink,long")
    modes2 = detect_three_mode_routing_modes(resp, mask, tok, cfg2)
    assert list(modes2) == ["nothink", "unknown", "long", "unknown"]


def test_apply_three_mode_caps_truncates_and_trusts_forced_labels():
    tok = FakeTokenizer()
    P, T = 4, 40

    class B:
        pass

    b = B()
    resp = torch.full((3, T), 999, dtype=torch.long)
    attn = torch.zeros(3, P + T, dtype=torch.long)
    attn[:, :P] = 1
    # row 0: NOTHINK, 30 tokens (cap 8 + 3 rt = 11 -> truncated)
    resp[0, :3] = torch.tensor([101, 102, 103])
    attn[0, P : P + 30] = 1
    # row 1: LONG, uncapped
    resp[1, 0] = 301
    attn[1, P : P + 30] = 1
    # row 2: junk head but FORCED short label -> label trusted, capped at 16 + 1
    attn[2, P : P + 30] = 1
    b.batch = {"responses": resp, "attention_mask": attn}

    cfg = ThreeModeRoutingConfig(nothink_max_tokens=8, short_max_tokens=16, long_max_tokens=None)
    known = np.array(["nothink", "long", "short"], dtype=object)
    forced = np.array([False, False, True])
    rtl = np.array([3, 1, 1], dtype=np.int64)
    modes, truncated = apply_three_mode_caps(
        b, tok, cfg, known_modes=known, known_forced=forced, routing_token_lengths=rtl
    )
    assert list(modes) == ["nothink", "long", "short"]
    assert list(truncated) == [True, False, True]
    resp_attn = b.batch["attention_mask"][:, P:]
    assert int(resp_attn[0].sum()) == 8 + 3   # cap + routing token allowance
    assert int(resp_attn[1].sum()) == 30      # LONG uncapped
    assert int(resp_attn[2].sum()) == 16 + 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
