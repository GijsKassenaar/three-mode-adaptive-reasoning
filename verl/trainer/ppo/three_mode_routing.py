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
"""
Adaptive Three-Mode Routing for GRPO (NOTHINK / SHORT / LONG).

The model freely chooses one of three reasoning modes as its first response token.
Routing is learned implicitly through a shaped reward surface — no auxiliary loss.

  - NOTHINK : highest base reward, steepest per-token discount (gamma_nothink^L).
              Best for prompts solvable with little or no reasoning.
  - SHORT   : medium base, moderate discount (gamma_short^L).
              Best for prompts needing brief reasoning.
  - LONG    : base reward, no discount.
              Best for hard prompts needing extended reasoning.
  - unknown : invalid routing token → fixed penalty reward (unknown_penalty).

Reward surface (all values configurable, see ThreeModeRoutingConfig):
  correct:   base(mode) × gamma(mode)^L      [LONG: no discount by default]
  incorrect: per-mode wrong penalty (default 0.0)
  unknown:   unknown_penalty (regardless of correctness)

where L = response length − routing token length.

Gamma convention is gamma^L (gamma=1.0 = no discount).  To place the reward
crossovers at the per-mode token caps:
  gamma_short   = (long_base / short_base)^(1 / short_cap)
  gamma_nothink = gamma_short × (short_base / nothink_base)^(1 / nothink_cap)
Do NOT confuse with an alpha = -ln(gamma) parameterization — entering gamma
values in alpha fields makes the discount collapse to ~0 at any realistic length.

Advantage: GRPO mean-centering within each group; std normalization is DISABLED
(reward magnitudes are intentionally heterogeneous).  By default unknown rollouts
are EXCLUDED from the group mean (exclude_unknown_from_mean) so a large
unknown_penalty cannot push the advantage of incorrect-but-valid groupmates
positive.

Rollout protocol:
  Phase 1 (steps 0…warmup_steps):          n//4 forced per mode + remainder free.
  Phase 2 (steps warmup_steps+1…warmup2_steps, if warmup2_steps>warmup_steps):
                                           1 forced per mode + remainder free.
  Phase 3 (post-warmup2):                  n free rollouts.

The balance term is computed over and applied to FREE rollouts only by default
(balance_free_only) — forced rollouts are balanced by construction and would
otherwise dilute the measured fractions toward the target during warmup, masking
a collapse of the free policy.

Module layout:
  ThreeModeRoutingConfig   — runtime configuration (mirrored by the Hydra config
                             block algorithm.three_mode_routing).
  RoutingPromptController  — prompt augmentation (routing question + optional
                             forced routing token) and routing-token promotion.
  ThreeModeRoutingForcer   — rollout builder (forced/free split, per-mode
                             max_tokens, optional two-pass free generation).
  compute_three_mode_routing_advantage — registered advantage estimator body.
  compute_three_mode_routing_metrics   — W&B diagnostics.
  apply_three_mode_caps    — post-generation per-mode token caps.
"""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import DataProtoConfig


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ThreeModeRoutingConfig:
    """Runtime configuration for adaptive three-mode self-routing GRPO."""

    enable: bool = False

    # Routing words emitted as the first response token.
    nothink_token: str = "NOTHINK"
    short_token: str = "SHORT"
    long_token: str = "LONG"

    # Routing question injected into the last user message before generation.
    routing_question: str = (
        "\n\nOutput NOTHINK to answer directly, SHORT for brief reasoning, "
        "or LONG for extended reasoning: "
    )

    # Reward bases for correct answers.
    nothink_base: float = 2.0
    short_base: float = 1.5
    long_base: float = 1.0

    # Per-token exponential discount factors (gamma^L).
    # reward = base * gamma^L, where L = tokens after the routing token.
    # gamma=1.0 means no discount. Lower gamma = steeper penalty for long responses.
    # Set so crossovers land at your token caps:
    #   gamma_short  = (long_base / short_base)^(1/short_cap)
    #   gamma_nothink = gamma_short * (short_base / nothink_base)^(1/nothink_cap)
    # Example for bases 1.4/1.3/1.0, caps 1024/3000: gamma_nothink≈0.99984, gamma_short≈0.999912
    gamma_nothink: float = 0.99984
    gamma_short: float = 0.999912
    # Discount for LONG (default 1.0 = no discount, the original design).  <1 adds
    # gentle length pressure within LONG as well.
    gamma_long: float = 1.0

    # When True, L (used for gamma^L) is computed only from tokens before the first
    # closing </think> tag instead of the full response length.  Requires a tokenizer
    # at the ray_trainer level (raises if missing).  Responses that never close
    # </think> get L=0 (i.e. no discount).
    reasoning_only: bool = False

    # Reward override for rollouts with an invalid routing token.
    unknown_penalty: float = -0.5

    # Reward for incorrect answers, per mode.  0.0 = no penalty (original behaviour).
    # Setting e.g. nothink_wrong_penalty=-0.3 creates a risk-reward tradeoff:
    # choosing an aggressive mode (high base) but getting it wrong costs more than
    # falling back to LONG, so the router learns to be calibrated.
    nothink_wrong_penalty: float = 0.0
    short_wrong_penalty: float = 0.0
    long_wrong_penalty: float = 0.0

    # Active routing modes (comma-separated, canonical subset of nothink,short,long).
    # Default = all three.  Set e.g. "nothink,long" to run a 2-mode router: only the
    # listed modes are forced during warmup and detected in free rollouts; a free
    # rollout that emits a disabled mode's token is treated as unknown (penalized), so
    # the model learns to avoid it.  Remember to set balance_target=1/n_modes
    # accordingly (e.g. 0.5 for two modes).  Fully reversible (default keeps 3-mode).
    enabled_modes: str = "nothink,short,long"

    # Rollout protocol
    warmup_steps: int = 100   # phase 1 end: n//4 forced per mode + remainder free
    warmup2_steps: int = 0    # phase 2 end (0 = disabled): 1 forced per mode + remainder free
    n_rollouts: int = 4       # total group size G

    # Per-mode generation caps.  Applied post-rollout (before reward + training) so
    # that a SHORT-routed response with more than short_max_tokens is scored wrong —
    # creating a real capability gap the router can learn from.  None = no cap.
    nothink_max_tokens: Optional[int] = None
    short_max_tokens: Optional[int] = None
    long_max_tokens: Optional[int] = None

    # Whole-batch load-balance term added at the ADVANTAGE level:
    #   balance_coef * (balance_target - frac_mode)
    # frac_mode is the fraction of each mode among all valid (non-unknown) rollouts.
    # Over-represented modes get a negative nudge; under-represented get a positive one.
    balance_coef: float = 0.5
    balance_target: float = 1.0 / 3.0  # equal 1/3 split across three modes

    # Linear annealing of balance_coef (0/0 = disabled, coef stays constant).
    # For global_step <= start: full balance_coef.  For step >= end: final_coef.
    # Linear interpolation in between.  Anneal to 0 after routing is established
    # (e.g. start=45, end=75) to test whether the mode split is genuinely
    # calibrated to problem difficulty rather than held at the target by the
    # balance term — and to let the split settle at the difficulty-appropriate
    # equilibrium instead of a forced 1/3.
    balance_anneal_start_step: int = 0
    balance_anneal_end_step: int = 0
    balance_anneal_final_coef: float = 0.0

    # When True, gate the balance nudge by correctness so its sign agrees with the
    # answer.  A positive nudge (under-represented mode → encourage) is applied only
    # to CORRECT rollouts; a negative nudge (over-represented mode → discourage) only
    # to WRONG rollouts.  This stops the default behaviour where a correct rollout in
    # a popular mode is pushed down (a muddy "good answer, but stop doing that" signal)
    # and where a wrong rollout in a rare mode is pushed up.  Pressure to abandon an
    # over-used mode then concentrates where that mode is actually failing.
    # False (default) keeps the original symmetric nudge over all rollouts in a mode.
    balance_correctness_gated: bool = False

    # When True, LONG rollouts with an INCORRECT answer never receive a balance nudge.
    # Rationale: LONG is the mode for the hardest problems, which are precisely the ones
    # most often answered wrong.  If an over-used LONG mode were discouraged on its wrong
    # rollouts, the model would learn to route hard problems away from LONG — the opposite
    # of what we want.  This zeroes the balance term for LONG∧incorrect so a wrong-but-hard
    # LONG attempt is never penalized by the load-balancer.  False (default) = no exemption.
    balance_protect_long_incorrect: bool = False

    # Solved-group gate: for global_steps <= solved_gate_warmup_steps, both the
    # unknown_penalty and the balance term are applied ONLY to groups with >=1 correct
    # answer (prevents bare-token gaming before the model can solve anything).
    # After the warmup ALL groups are shaped.  0 = gate never active.
    solved_gate_warmup_steps: int = 0

    # Forced mode for val_only evaluation runs.  "" or "free" = free routing (model chooses).
    # "nothink" / "short" / "long" = inject routing question + token into prompt so the
    # model generates entirely in that mode.  Use with trainer.val_only=True.
    val_forced_mode: str = ""

    # Forced mode for TRAINING rollouts (ablation: fully-forced single-mode training).
    # "" = normal phased forcing (n//4 forced per mode + free, per warmup_steps/warmup2_steps).
    # "nothink" / "short" / "long" = every rollout, every step, is forced into that one mode
    # (n_forced=n_rollouts, n_free=0) — no free/self-routing behaviour at all, for the whole
    # run. Overrides enabled_modes for the forced loop (only this one mode is ever forced,
    # regardless of what enabled_modes lists) so no extra config coordination is needed.
    # Reward shaping (base/gamma/wrong_penalty for the forced mode) still applies normally;
    # the balance term is a no-op (balance_free_only excludes forced rollouts and there are
    # no free ones). Use with trainer.val_only=False for real training ablations.
    train_forced_mode: str = ""

    # ── Detection / normalization behaviour flags.  Defaults are the recommended
    # settings; each flag can be flipped to reproduce earlier behaviour. ─────────

    # When True, mode detection requires a word boundary after the routing token.
    # False (default) uses token-id + prefix matching, which does not misclassify
    # e.g. "LONGI need…" as unknown (a \b regex drops a large fraction of valid
    # routing words whose continuation attaches a word character with no space).
    strict_token_boundary: bool = False

    # When True (default), the balance term computes mode fractions over FREE
    # rollouts only and nudges only free rollouts.  False restores fractions over
    # all rollouts, which dilutes the balance signal toward the target whenever
    # forced rollouts are present.
    balance_free_only: bool = True

    # When True (default), unknown rollouts are excluded from the group mean used
    # for centering (their own advantage is unknown_penalty − mean_of_valid).
    # False restores the behaviour where a large unknown_penalty drags the
    # group mean down and gives incorrect-but-valid groupmates a positive advantage.
    exclude_unknown_from_mean: bool = True

    # Step from which exclude_unknown_from_mean takes effect (0 = immediately).
    # The unknowns' drag on the group mean is what gives ALL forced routing tokens
    # a positive advantage during warmup — the engine that bootstraps routing-word
    # emission.  With exclusion active from step 1, free rollouts may never emit a
    # routing word.  Set this past the forcing warmup (e.g. 30): bootstrap runs
    # with the drag active, then the clean mean takes over once unknowns are rare.
    exclude_unknown_from_mean_start_step: int = 0

    # When True, the per-mode token caps are also applied during validation
    # (before scoring), matching the training objective — useful as a diagnostic.
    # Default False: validation measures production behaviour, where the model
    # generates freely with no truncation harness.  Note this means a mode can
    # score slightly better at val than under the trained (capped) objective.
    apply_caps_in_validation: bool = False

    # ── Two-pass free rollout generation (efficiency) ─────────────────────────
    # Pass 1 generates only router_pass_tokens tokens (enough to read the routing
    # word), then pass 2 continues each rollout with exactly its mode's remaining
    # budget (mode_cap − router_pass_tokens).  This turns the per-mode caps into
    # real generation stops instead of post-hoc truncation: no tokens are generated
    # past a binding cap, and unknown-routed rollouts can stop immediately
    # (unknown_early_stop) since their reward is a constant penalty regardless of
    # what they would have generated.  False (default) = single-pass generation.
    # Applies to FREE rollouts only; forced rollouts already generate with their
    # mode's max_tokens.
    two_pass_enable: bool = False
    # Pass-1 token budget.  Must cover leading whitespace + the longest routing
    # word (NOTHINK = 3 tokens); 8 is a safe default.
    router_pass_tokens: int = 8
    # Skip pass 2 for unknown-routed rollouts (their reward does not depend on the
    # continuation, so generating it is pure waste — and the penalty gradient then
    # concentrates on the routing decision instead of a full reasoning trace).
    unknown_early_stop: bool = True
    # Debug: decode and print a few (pass-1 | pass-2 | stitched) examples per step so
    # the stitched continuation can be eyeballed for coherence / answer-format intactness.
    two_pass_debug: bool = False
    two_pass_debug_n: int = 2


# ─── Prompt controller ────────────────────────────────────────────────────────


class RoutingPromptController:
    """Prompt augmentation and routing-token promotion for routed rollouts.

    Two prompt variants are produced from the dataset's untokenized ``raw_prompt``
    (new async-rollout pipeline):

    - :meth:`augment_batch` — routing question + FORCED routing token appended
      after the chat template's assistant generation prefix, so vLLM
      deterministically generates in the requested mode (training warmup and
      forced-mode evaluation).
    - :meth:`append_routing_question` — routing question only; the model predicts
      the routing token itself (free rollouts and validation).

    After forced-prefix generation, :meth:`_promote_routing_tokens_to_response`
    relocates the forced routing token from the prompt tail into the response
    head so it receives gradient like any other generated token — this is what
    teaches the model to EMIT the routing word itself.
    """

    def __init__(
        self,
        tokenizer: Any,
        nothink_token: str = "NOTHINK",
        short_token: str = "SHORT",
        long_token: str = "LONG",
        nothink_suffix: str = "",
        routing_question: str = "",
        apply_chat_template_kwargs: Optional[dict] = None,
    ):
        """Parameters
        ----------
        tokenizer : HF tokenizer for chat-template / routing-token encoding.
        nothink_token / short_token / long_token : routing words.
        nothink_suffix : text appended after nothink_token in the forced prefix
            ("" for three-mode routing — the model closes </think> itself).
        routing_question : question injected into the last user message.
        apply_chat_template_kwargs : extra kwargs forwarded to
            `tokenizer.apply_chat_template`.  Must match what the AgentLoop uses
            in standard rollouts so our pre-tokenized prompts are bit-identical
            to what the agent loop would produce on its own.
        """
        self.tokenizer = tokenizer
        self.routing_question = routing_question
        self._apply_chat_template_kwargs: dict = dict(apply_chat_template_kwargs or {})

        self._short_token_ids: list[int] = self._tokenize(short_token)
        self._long_token_ids: list[int] = self._tokenize(long_token)
        self._nothink_token_ids: list[int] = self._tokenize(nothink_token + nothink_suffix)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        tok = self.tokenizer
        if hasattr(tok, "encode"):
            return list(tok.encode(text, add_special_tokens=False))
        result = tok(text, add_special_tokens=False, return_attention_mask=False)
        return list(result["input_ids"])

    def _get_pad_token_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            eos = getattr(self.tokenizer, "eos_token_id", 0)
            return eos if eos is not None else 0
        return int(pad)

    def _apply_chat_template_to_messages(self, messages: list) -> list[int]:
        """Apply chat template + tokenize, matching what the AgentLoop does.

        Always uses `add_generation_prompt=True` so the template appends the assistant
        prefix (e.g. "<|im_start|>assistant\\n").  After this call, the forced routing
        token is concatenated *after* the assistant prefix so that vLLM continues from
        the routing token as the first generated content.
        """
        prompt_ids = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            **self._apply_chat_template_kwargs,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        # In rare cases apply_chat_template returns a 2-D batch; flatten to 1-D
        if len(prompt_ids) > 0 and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        return list(prompt_ids)

    def _inject_routing_question_into_messages(self, messages: list) -> list:
        """Append routing question to the last user message content.

        Placing the routing question in the user turn means the assistant prefix
        (<think>) immediately follows it.  Returns a new list; the input is not
        modified.
        """
        messages = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            role = str(messages[i].get("role", "")).lower()
            if role in ("user", "human"):
                msg = dict(messages[i])
                msg["content"] = str(msg.get("content", "")) + self.routing_question
                messages[i] = msg
                break
        return messages

    def _set_prompt_tensors(
        self,
        gen_batch: DataProto,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> DataProto:
        """Write prompt tensors into gen_batch.batch, creating it if it is None.

        After `ray_trainer._get_gen_batch`, the resulting DataProto has `batch=None`
        (no batch tensors are popped — they stay on the source batch).  We need to
        construct a fresh TensorDict here.
        """
        N = input_ids.shape[0]
        augmented = deepcopy(gen_batch)
        if augmented.batch is None:
            augmented.batch = TensorDict({}, batch_size=N)
        augmented.batch["input_ids"] = input_ids
        augmented.batch["attention_mask"] = attention_mask
        augmented.batch["position_ids"] = position_ids
        return augmented

    def _build_left_padded_ids(
        self,
        full_ids: list[int],
        max_prompt_length: int,
    ) -> tuple[list[int], list[int]]:
        """Truncate-from-left if too long, then left-pad to max_prompt_length.

        Returns (input_ids, attention_mask) as Python lists of length max_prompt_length.
        Suffix tokens (routing question + routing token) are at the very end of full_ids
        and are always preserved by truncating the prefix instead.
        """
        pad_id = self._get_pad_token_id()
        if len(full_ids) > max_prompt_length:
            full_ids = full_ids[-max_prompt_length:]

        pad_len = max_prompt_length - len(full_ids)
        if pad_len > 0:
            new_ids = [pad_id] * pad_len + list(full_ids)
            new_mask = [0] * pad_len + [1] * len(full_ids)
        else:
            new_ids = list(full_ids)
            new_mask = [1] * max_prompt_length
        return new_ids, new_mask

    # ── Prompt augmentation ───────────────────────────────────────────────────

    def augment_batch(
        self,
        gen_batch: DataProto,
        routing_mode: str,  # "nothink", "short" or "long"
        max_prompt_length: int,
    ) -> DataProto:
        """Tokenize raw_prompt + chat template, then append the routing suffix.

        Operates on the async-rollout pipeline where the dataset returns
        `raw_prompt` (a list of message dicts) in `non_tensor_batch` rather than
        pre-tokenized `input_ids`.  We do the tokenization ourselves so that we
        can append `routing_question + routing_token` *after* the assistant
        generation prefix that the chat template adds — vLLM then continues
        generation with the routing token as the first response content.

        The result is emitted in the AgentLoop's `prefilled_prompt_mode` format:
            - input_ids, attention_mask, position_ids in `batch.batch`
            - meta_info["prefilled_prompt_mode"] = True
        which makes the agent loop bypass its own chat-template step and use
        our token ids directly.

        If the suffix would push the sequence past `max_prompt_length`, the prompt
        prefix is truncated from the left (suffix is always preserved).
        """
        raw_prompts = gen_batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError(
                "augment_batch requires `raw_prompt` in gen_batch.non_tensor_batch. "
                "This DataProto came from the dataset before tokenization (async-rollout pipeline)."
            )

        if routing_mode == "short":
            suffix_ids = self._short_token_ids
        elif routing_mode == "nothink":
            suffix_ids = self._nothink_token_ids
        else:
            suffix_ids = self._long_token_ids
        N = len(raw_prompts)

        new_ids_list: list[list[int]] = []
        new_mask_list: list[list[int]] = []
        for i in range(N):
            messages = self._inject_routing_question_into_messages(list(raw_prompts[i]))
            prompt_ids = self._apply_chat_template_to_messages(messages)
            full_ids = prompt_ids + list(suffix_ids)
            ids, mask = self._build_left_padded_ids(full_ids, max_prompt_length)
            new_ids_list.append(ids)
            new_mask_list.append(mask)

        new_input_ids = torch.tensor(new_ids_list, dtype=torch.long)
        new_attention_mask = torch.tensor(new_mask_list, dtype=torch.long)
        position_ids = (new_attention_mask.cumsum(dim=-1) - 1).clamp(min=0) * new_attention_mask

        augmented_batch = self._set_prompt_tensors(
            gen_batch, new_input_ids, new_attention_mask, position_ids
        )

        # Tag every sample with its routing mode for advantage computation downstream
        augmented_batch.non_tensor_batch["routing_mode"] = np.full(N, routing_mode, dtype=object)

        # Tell the agent loop to use these prompt ids verbatim (skip chat templating).
        if augmented_batch.meta_info is None:
            augmented_batch.meta_info = {}
        augmented_batch.meta_info["prefilled_prompt_mode"] = True

        return augmented_batch

    def append_routing_question(
        self,
        gen_batch: DataProto,
        max_prompt_length: int,
    ) -> DataProto:
        """Free-rollout / validation variant: inject routing question, no routing token.

        Injects the routing question into the last user message before applying
        the chat template.  The model then predicts "NOTHINK", "SHORT", or "LONG"
        itself as the first response token.

        Emits prefilled_prompt_mode=True like augment_batch.
        """
        raw_prompts = gen_batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError(
                "append_routing_question requires `raw_prompt` in non_tensor_batch."
            )

        N = len(raw_prompts)

        new_ids_list: list[list[int]] = []
        new_mask_list: list[list[int]] = []
        for i in range(N):
            messages = self._inject_routing_question_into_messages(list(raw_prompts[i]))
            prompt_ids = self._apply_chat_template_to_messages(messages)
            ids, mask = self._build_left_padded_ids(prompt_ids, max_prompt_length)
            new_ids_list.append(ids)
            new_mask_list.append(mask)

        new_input_ids = torch.tensor(new_ids_list, dtype=torch.long)
        new_attention_mask = torch.tensor(new_mask_list, dtype=torch.long)
        position_ids = (new_attention_mask.cumsum(dim=-1) - 1).clamp(min=0) * new_attention_mask

        augmented_batch = self._set_prompt_tensors(
            gen_batch, new_input_ids, new_attention_mask, position_ids
        )

        if augmented_batch.meta_info is None:
            augmented_batch.meta_info = {}
        augmented_batch.meta_info["prefilled_prompt_mode"] = True

        return augmented_batch

    # ── Routing-token promotion ───────────────────────────────────────────────

    def _promote_routing_tokens_to_response(
        self,
        output: DataProto,
        routing_mode_arr: np.ndarray,
    ) -> DataProto:
        """Move routing token(s) from the end of the prompt into the start of responses.

        After forced-prefix generation the routing token sits at the tail of input_ids
        (inside the prompt portion of the sequence). This method relocates it so that:
          - The prompt stored in the DataProto ends at the routing_question (no routing token).
          - The response begins with the routing token(s) followed by the generated text.
          - response_mask covers the routing token positions → gradient flows through them.

        The full-sequence (prompt + response) layout is maintained at constant length:
          OLD: [PAD... | prompt | routing_q | routing_t || gen_text | PAD...]
          NEW: [PAD(+n) | prompt | routing_q || routing_t | gen_text | PAD(-n)]
               ^^^^^^^^ prompt portion ^^^^^^^  ^^^^^^^^^ response portion ^^^^^^

        The transformation for full-sequence tensors (input_ids, attention_mask) is:
            new[i] = cat([zeros/pad * n_rt, old[i, :-n_rt]])
        which shifts routing_t tokens from the prompt boundary into the response region.
        The last n_rt positions that are dropped are response padding (0/EOS) in practice.

        Parameters
        ----------
        output : DataProto returned by generate_fn; contains:
            - input_ids        [N, prompt_len + response_len] — full sequence
            - prompts          [N, prompt_len]                — prompt portion (optional key)
            - responses        [N, response_len]              — generated response
            - attention_mask   [N, prompt_len + response_len] — full sequence mask
            - position_ids     [N, prompt_len + response_len] — optional, recomputed if present
        routing_mode_arr : np.ndarray[str], shape (N,)
            "nothink", "short" or "long" per rollout — determines which token ids to promote.

        Returns
        -------
        DataProto with updated tensors. response_mask is always (re)computed from the
        updated attention_mask so gradient correctly includes the routing token positions.
        """
        result = deepcopy(output)

        input_ids = result.batch["input_ids"]         # full seq  [N, total_len]
        responses = result.batch["responses"]          # [N, response_len]
        attention_mask = result.batch["attention_mask"]  # full seq  [N, total_len]
        has_prompts = "prompts" in result.batch
        if has_prompts:
            prompts = result.batch["prompts"]          # [N, prompt_len]

        max_response_len = responses.shape[1]
        max_prompt_len = input_ids.shape[1] - max_response_len
        N = input_ids.shape[0]
        pad_id = self._get_pad_token_id()

        for i in range(N):
            mode = str(routing_mode_arr[i]).lower()
            if mode == "short":
                n_rt = len(self._short_token_ids)
            elif mode == "nothink":
                n_rt = len(self._nothink_token_ids)
            else:
                n_rt = len(self._long_token_ids)
            if n_rt == 0:
                continue

            # Routing token values live at the last n_rt positions of the prompt portion
            # of the full sequence.  Extract BEFORE modifying in-place.
            routing_vals = input_ids[i, max_prompt_len - n_rt : max_prompt_len].clone()

            # Full-sequence shift: prepend n_rt padding, drop last n_rt (response padding)
            input_ids[i] = torch.cat([
                input_ids.new_full((n_rt,), pad_id),
                input_ids[i, :-n_rt],
            ])
            attention_mask[i] = torch.cat([
                attention_mask.new_zeros(n_rt),
                attention_mask[i, :-n_rt],
            ])

            # Prepend routing tokens to response, truncate trailing (padding) positions
            responses[i] = torch.cat([routing_vals, responses[i, :-n_rt]])

            if has_prompts:
                prompts[i] = torch.cat([
                    prompts.new_full((n_rt,), pad_id),
                    prompts[i, :-n_rt],
                ])

        # Derive response_mask from the updated full attention_mask
        # (attention_mask[:, -response_len:] is now the response portion with routing tokens included)
        response_mask = attention_mask[:, -max_response_len:].clone()

        # Recompute position_ids from the new attention_mask if the key exists
        if "position_ids" in result.batch:
            pos_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
            result.batch["position_ids"] = pos_ids.to(result.batch["position_ids"].dtype)

        result.batch["input_ids"] = input_ids
        result.batch["responses"] = responses
        result.batch["attention_mask"] = attention_mask
        result.batch["response_mask"] = response_mask
        if has_prompts:
            result.batch["prompts"] = prompts

        return result


# ─── Mode detection ───────────────────────────────────────────────────────────


def _build_mode_regex(token: str, strict_boundary: bool = False) -> Optional[re.Pattern]:
    if not token:
        return None
    # Prefix match by default (NO trailing \b): the routing word is the literal first
    # token of the response, and the next generated token can attach a word character
    # with no space (e.g. "LONGI need…").  Requiring a boundary misclassifies those as
    # unknown.  strict_boundary=True restores the \b behaviour.
    boundary = r"\b" if strict_boundary else ""
    return re.compile(rf"^\s*{re.escape(token)}{boundary}", flags=re.IGNORECASE)


_CANONICAL_MODES = ("nothink", "short", "long")


def enabled_modes_from_config(config: Any) -> list:
    """Active routing modes in canonical order.

    Reads `config.enabled_modes` (comma string like "nothink,long"); empty/None
    falls back to all three modes (default 3-mode behaviour).  Used by detection,
    the forcer's forced loop, and the trainer's forced/free split arithmetic.
    """
    raw = getattr(config, "enabled_modes", None) if config is not None else None
    if not raw:
        return list(_CANONICAL_MODES)
    if isinstance(raw, str):
        sel = {m.strip().lower() for m in raw.split(",") if m.strip()}
    else:
        sel = {str(m).strip().lower() for m in raw}
    modes = [m for m in _CANONICAL_MODES if m in sel]
    return modes or list(_CANONICAL_MODES)


def detect_three_mode_routing_modes(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    config: ThreeModeRoutingConfig,
) -> np.ndarray:
    """Detect the routing mode of each rollout from the leading word of its response.

    Detection is token-id-level first (the promoted routing token is an exact id
    sequence, so this is 100% reliable for forced rollouts), with a prefix-regex
    fallback for free rollouts that emit a spelling variant.

    Returns np.ndarray (bs,) with entries in {"nothink", "short", "long", "unknown"}.
    """
    bsz = responses.size(0)
    lengths = response_mask.sum(dim=-1).to(dtype=torch.long)

    strict = bool(getattr(config, "strict_token_boundary", False))
    enabled = set(enabled_modes_from_config(config))
    # Disabled modes never match → their token falls through to "unknown".
    nothink_re = _build_mode_regex(config.nothink_token, strict) if "nothink" in enabled else None
    short_re = _build_mode_regex(config.short_token, strict) if "short" in enabled else None
    long_re = _build_mode_regex(config.long_token, strict) if "long" in enabled else None

    id_candidates = []
    for mode, token in (
        ("nothink", config.nothink_token),
        ("short", config.short_token),
        ("long", config.long_token),
    ):
        if mode not in enabled:
            continue
        ids = tokenizer.encode(token, add_special_tokens=False) if token else []
        if ids:
            id_candidates.append((mode, ids))

    # Peek at ≤20 tokens — enough to read any routing word.
    peek = 20

    modes = np.empty((bsz,), dtype=object)
    for i in range(bsz):
        valid_len = int(lengths[i].item())
        if valid_len <= 0:
            modes[i] = "unknown"
            continue
        head_ids = responses[i, : min(valid_len, peek)].tolist()
        # Token-id level: exact match of the leading ids (robust for promoted tokens).
        matched = None
        if not strict:
            for mode, ids in id_candidates:
                if head_ids[: len(ids)] == ids:
                    matched = mode
                    break
        if matched is not None:
            modes[i] = matched
            continue
        text = tokenizer.decode(head_ids, skip_special_tokens=True)
        if nothink_re is not None and nothink_re.match(text):
            modes[i] = "nothink"
        elif short_re is not None and short_re.match(text):
            modes[i] = "short"
        elif long_re is not None and long_re.match(text):
            modes[i] = "long"
        else:
            modes[i] = "unknown"
    return modes


def _routing_token_length_for_mode(mode: str, ctrl: RoutingPromptController) -> int:
    """Number of token IDs in the routing token for a given mode."""
    mode = mode.lower()
    if mode == "nothink":
        return len(ctrl._nothink_token_ids)
    if mode == "short":
        return len(ctrl._short_token_ids)
    if mode == "long":
        return len(ctrl._long_token_ids)
    return 0  # unknown


# ─── Forcer ───────────────────────────────────────────────────────────────────


class ThreeModeRoutingForcer:
    """Manages rollout generation for three-mode routing.

    Phase 1 (steps 0..warmup_steps): n//4 forced per mode + remainder free.
    Phase 2 (steps warmup_steps+1..warmup2_steps, if set): 1 forced per mode + remainder free.
    Phase 3 (post-warmup2): all n rollouts are free.
    """

    def __init__(
        self,
        config: ThreeModeRoutingConfig,
        tokenizer: Any,
        apply_chat_template_kwargs: Optional[dict] = None,
    ):
        self._ctrl = RoutingPromptController(
            tokenizer=tokenizer,
            nothink_token=config.nothink_token,
            short_token=config.short_token,
            long_token=config.long_token,
            nothink_suffix="",  # model closes </think> itself; we don't force-close it
            routing_question=config.routing_question,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        self.config = config

    def append_routing_question(self, gen_batch, max_prompt_length: int):
        """Inject routing question into the last user message without a forced routing token.

        Used during validation so the model predicts the routing token itself.
        Delegates to the underlying RoutingPromptController.
        """
        return self._ctrl.append_routing_question(gen_batch, max_prompt_length)

    # ── Rollout builder ───────────────────────────────────────────────────────

    def build_rollouts(
        self,
        gen_batch,
        generate_fn,
        n_forced: int,
        n_free: int,
        max_prompt_length: int,
        dp_size: Optional[int] = None,
        forced_modes: Optional[list] = None,
    ):
        """Generate forced-prefix and free rollouts, return (mixed_output, source_indices, metrics).

        Parameters
        ----------
        n_forced : int
            Number of forced rollouts per mode (0 post-warmup, >0 during warmup).
        n_free : int
            Number of free rollouts per prompt.
        forced_modes : list, optional
            Modes to force, overriding enabled_modes_from_config(self.config) for the forced
            loop only. Used for train_forced_mode (fully-forced single-mode training) so that
            exactly one mode is forced regardless of what enabled_modes lists. None (default)
            = enabled_modes_from_config(self.config), the standard behaviour.
        """
        from verl.trainer.ppo.ray_trainer import _concat_dataprotos_non_empty

        ctrl = self._ctrl
        N = len(gen_batch)
        chunks = []
        src_chunks = []

        _mode_max_tokens = {
            "nothink": self.config.nothink_max_tokens,
            "short": self.config.short_max_tokens,
            "long": self.config.long_max_tokens,
        }

        def _copy_meta(dst, max_tokens=None):
            if dst.meta_info is None:
                dst.meta_info = {}
            dst.meta_info[DataProtoConfig.auto_padding_key] = True
            for k, v in (gen_batch.meta_info or {}).items():
                dst.meta_info.setdefault(k, v)
            if max_tokens is not None:
                dst.meta_info["max_tokens"] = int(max_tokens)

        # ── Forced rollouts (n_forced per mode) ───────────────────────────────
        # Only force the enabled modes (default all three; e.g. "nothink,long" for 2-mode).
        # forced_modes overrides this (train_forced_mode: force exactly one mode).
        for mode in (forced_modes if forced_modes is not None else enabled_modes_from_config(self.config)):
            if n_forced <= 0:
                break
            aug = ctrl.augment_batch(gen_batch, mode, max_prompt_length)
            gen = aug.repeat(repeat_times=n_forced, interleave=True)
            _copy_meta(gen, max_tokens=_mode_max_tokens.get(mode))
            gen.meta_info["prefilled_prompt_mode"] = True
            out = generate_fn(gen)
            out.meta_info.pop("timing", None)

            modes_arr = np.full(N * n_forced, mode, dtype=object)
            out = ctrl._promote_routing_tokens_to_response(out, modes_arr)
            n_rt = _routing_token_length_for_mode(mode, ctrl)
            out.non_tensor_batch["routing_mode"] = modes_arr
            out.non_tensor_batch["routing_token_length"] = np.full(N * n_forced, n_rt, dtype=np.int64)
            out.non_tensor_batch["routing_forced"] = np.ones(N * n_forced, dtype=bool)

            chunks.append(out)
            src_chunks.append(np.repeat(np.arange(N, dtype=np.int64), n_forced))

        # ── Free rollouts ─────────────────────────────────────────────────────
        two_pass_metrics: dict[str, float] = {}
        if n_free > 0:
            if self.config.two_pass_enable:
                free_chunks, free_srcs, two_pass_metrics = self._build_free_two_pass(
                    gen_batch=gen_batch,
                    generate_fn=generate_fn,
                    n_free=n_free,
                    max_prompt_length=max_prompt_length,
                    copy_meta=_copy_meta,
                    dp_size=dp_size,
                )
                chunks.extend(free_chunks)
                src_chunks.extend(free_srcs)
            else:
                free_aug = ctrl.append_routing_question(gen_batch, max_prompt_length)
                free_gen = free_aug.repeat(repeat_times=n_free, interleave=True)
                _copy_meta(free_gen)
                out_free = generate_fn(free_gen)
                out_free.meta_info.pop("timing", None)

                responses = out_free.batch["responses"]
                resp_mask = out_free.batch.get(
                    "response_mask",
                    out_free.batch["attention_mask"][:, -responses.shape[1]:],
                )
                free_modes = detect_three_mode_routing_modes(
                    responses=responses,
                    response_mask=resp_mask,
                    tokenizer=ctrl.tokenizer,
                    config=self.config,
                )
                free_rt_lengths = np.array(
                    [_routing_token_length_for_mode(m, ctrl) for m in free_modes], dtype=np.int64
                )
                out_free.non_tensor_batch["routing_mode"] = free_modes
                out_free.non_tensor_batch["routing_token_length"] = free_rt_lengths
                out_free.non_tensor_batch["routing_forced"] = np.zeros(N * n_free, dtype=bool)

                chunks.append(out_free)
                src_chunks.append(np.repeat(np.arange(N, dtype=np.int64), n_free))

        mixed = _concat_dataprotos_non_empty(chunks)
        source_indices = np.concatenate(src_chunks)

        # CRITICAL (two-pass): in async/agent-loop mode the reward is computed DURING
        # generation and stashed in `rm_scores`.  Two-pass free rollouts are scored in
        # pass 1 — on the few-token routing stub ("SHORT for brief reasoning"), before any
        # answer exists — so their rm_scores is a phantom 0 that select_idxs carries into
        # the stitched rollout.  Downstream (_compute_or_extract_reward) trusts an existing
        # rm_scores verbatim, so without this the real stitched answer is never scored and
        # every free rollout reads as wrong.  Drop rm_scores whenever two-pass actually ran
        # so the reward is recomputed on the final responses (forced rollouts re-score
        # identically, so dropping it for the whole batch is safe).
        if two_pass_metrics and "rm_scores" in mixed.batch.keys():
            mixed.batch.pop("rm_scores")

        # DP padding
        if dp_size is not None and dp_size > 0 and len(mixed) > 0 and len(mixed) % dp_size != 0:
            pad = int(dp_size - len(mixed) % dp_size)
            pad_idx = np.arange(pad, dtype=np.int64) % len(mixed)
            mixed = _concat_dataprotos_non_empty([mixed, mixed.select_idxs(pad_idx)])
            source_indices = np.concatenate([source_indices, source_indices[pad_idx]])

        metrics = {
            "three_mode_routing/n_forced_per_mode": float(n_forced),
            "three_mode_routing/n_free": float(n_free),
            **two_pass_metrics,
        }
        return mixed, source_indices, metrics

    # ── Two-pass free rollout generation ──────────────────────────────────────

    def _build_free_two_pass(
        self,
        gen_batch,
        generate_fn,
        n_free: int,
        max_prompt_length: int,
        copy_meta,
        dp_size: Optional[int] = None,
    ):
        """Two-pass free rollouts: pass 1 reads the routing word, pass 2 continues
        each rollout with its mode's exact remaining budget.

        Pass 1: routing question injected, generation capped at router_pass_tokens.
        Pass 2 (per detected mode): continuation prompt = valid prompt tokens +
            pass-1 tokens, max_tokens = mode_cap − pass-1 len.
        Rollouts that finished in pass 1 (EOS within the router budget) and —
        when unknown_early_stop — unknown-routed rollouts skip pass 2 entirely;
        their final response is just the pass-1 tokens.

        Returns (chunks, src_index_arrays, metrics).  Each chunk has the standard
        (prompt_width P, response_width R) layout of a single-pass generation
        output, so downstream code is unchanged.
        """
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        cfg = self.config
        ctrl = self._ctrl
        tok = ctrl.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_id = tok.eos_token_id
        B1 = max(1, int(cfg.router_pass_tokens))
        N = len(gen_batch)

        # ── Pass 1: routing decision only ────────────────────────────────────
        free_aug = ctrl.append_routing_question(gen_batch, max_prompt_length)
        free_gen = free_aug.repeat(repeat_times=n_free, interleave=True)
        copy_meta(free_gen, max_tokens=B1)
        free_gen.meta_info["prefilled_prompt_mode"] = True
        out1 = generate_fn(free_gen)
        out1.meta_info.pop("timing", None)

        responses1 = out1.batch["responses"]              # (M, R) — padded to response_length
        M, R = responses1.shape
        attn1 = out1.batch["attention_mask"]               # (M, P + R)
        P = attn1.shape[1] - R
        resp_attn1 = attn1[:, P:]
        valid1 = resp_attn1.sum(dim=-1).long()
        modes1 = detect_three_mode_routing_modes(responses1, resp_attn1, tok, cfg)
        mode_l = np.array([str(m).lower() for m in modes1], dtype=object)

        src_base = np.repeat(np.arange(N, dtype=np.int64), n_free)

        # Finished in pass 1: emitted EOS within the router budget (or empty).
        finished = np.zeros(M, dtype=bool)
        for i in range(M):
            L = int(valid1[i].item())
            if L == 0 or int(responses1[i, L - 1].item()) == eos_id:
                finished[i] = True

        cap_map = {
            "nothink": cfg.nothink_max_tokens,
            "short": cfg.short_max_tokens,
            "long": cfg.long_max_tokens,
        }

        cont_groups: dict[str, list[int]] = {}
        leftover: list[int] = []
        for i in range(M):
            m = mode_l[i]
            if finished[i]:
                leftover.append(i)
            elif m == "unknown" and cfg.unknown_early_stop:
                leftover.append(i)
            else:
                cont_groups.setdefault(str(m), []).append(i)

        chunks = []
        srcs = []

        # ── Pass 2 per mode group ────────────────────────────────────────────
        for m, rows in cont_groups.items():
            rows_np = np.array(rows, dtype=np.int64)
            cap = cap_map.get(m)  # unknown (early_stop=False) → None → full window
            budget = int(cap) - B1 if cap is not None else R - B1
            budget = max(1, min(budget, R - B1))

            # Continuation prompts: valid prompt + valid pass-1 tokens, left-padded
            # to a common width of P + B1 (budget-forcing style).
            ids_list, mask_list = [], []
            for i in rows:
                p_row = out1.batch["input_ids"][i, :P]
                p_valid = p_row[attn1[i, :P].bool()]
                L1 = int(valid1[i].item())
                seq = torch.cat([p_valid, responses1[i, :L1]])
                pad_n = P + B1 - seq.shape[0]
                ids_list.append(torch.cat([seq.new_full((pad_n,), pad_id), seq]))
                mask_list.append(
                    torch.cat([seq.new_zeros(pad_n), torch.ones_like(seq)]).long()
                )
            ids2 = torch.stack(ids_list)
            mask2 = torch.stack(mask_list)
            pos2 = (mask2.cumsum(dim=-1) - 1).clamp(min=0) * mask2

            gen2 = DataProto(
                batch=TensorDict(
                    {"input_ids": ids2, "attention_mask": mask2, "position_ids": pos2},
                    batch_size=[len(rows)],
                ),
                non_tensor_batch={
                    "__skip_reward_compute__": np.ones(len(rows), dtype=bool),
                },
                meta_info={},
            )
            copy_meta(gen2, max_tokens=budget)
            gen2.meta_info["prefilled_prompt_mode"] = True

            if dp_size is not None and dp_size > 0 and len(gen2) % dp_size != 0:
                gen2_padded, pad_sz = pad_dataproto_to_divisor(gen2, dp_size)
                out2 = generate_fn(gen2_padded)
                out2 = unpad_dataproto(out2, pad_size=pad_sz)
            else:
                out2 = generate_fn(gen2)
            out2.meta_info.pop("timing", None)

            resp2 = out2.batch["responses"]
            R2 = resp2.shape[1]
            valid2 = out2.batch["attention_mask"][:, -R2:].sum(dim=-1).long()

            # Stitch back to the standard (P, R) layout: response = pass1 + pass2.
            n_rows = len(rows)
            f_resp = responses1.new_full((n_rows, R), pad_id)
            f_resp_attn = resp_attn1.new_zeros((n_rows, R))
            for j, i in enumerate(rows):
                L1 = int(valid1[i].item())
                L2 = min(int(valid2[j].item()), R - L1)
                f_resp[j, :L1] = responses1[i, :L1]
                if L2 > 0:
                    f_resp[j, L1 : L1 + L2] = resp2[j, :L2]
                f_resp_attn[j, : L1 + L2] = 1
            f_prompts = out1.batch["input_ids"][rows_np.tolist()][:, :P]
            f_prompt_attn = attn1[rows_np.tolist()][:, :P]
            f_input = torch.cat([f_prompts, f_resp], dim=1)
            f_attn = torch.cat([f_prompt_attn, f_resp_attn], dim=1)

            sub = out1.select_idxs(rows_np)  # inherits non-tensor rows + meta keys
            sub.batch["responses"] = f_resp
            sub.batch["input_ids"] = f_input
            sub.batch["attention_mask"] = f_attn
            if "position_ids" in sub.batch and sub.batch["position_ids"].dim() == 2:
                sub.batch["position_ids"] = (
                    (f_attn.cumsum(dim=-1) - 1).clamp(min=0) * f_attn
                ).to(sub.batch["position_ids"].dtype)
            if "prompts" in sub.batch:
                sub.batch["prompts"] = f_prompts
            if "response_mask" in sub.batch:
                sub.batch["response_mask"] = f_resp_attn

            n_rt = _routing_token_length_for_mode(m, ctrl)
            sub.non_tensor_batch["routing_mode"] = np.full(n_rows, m, dtype=object)
            sub.non_tensor_batch["routing_token_length"] = np.full(n_rows, n_rt, dtype=np.int64)
            sub.non_tensor_batch["routing_forced"] = np.zeros(n_rows, dtype=bool)
            sub.non_tensor_batch.pop("__skip_reward_compute__", None)

            # ── Debug: eyeball the stitched continuation ─────────────────────────
            # Verifies pass 2 CONTINUES (no restart/duplication) and the answer format
            # survives the split.  Prints pass-1 text, pass-2 head, stitched tail.
            if getattr(cfg, "two_pass_debug", False):
                for j in range(min(int(getattr(cfg, "two_pass_debug_n", 2)), n_rows)):
                    L1 = int(valid1[rows[j]].item())
                    L2 = min(int(valid2[j].item()), R - L1)
                    pass1_txt = tok.decode(responses1[rows[j], :L1].tolist(), skip_special_tokens=False)
                    pass2_txt = tok.decode(resp2[j, : min(L2, 40)].tolist(), skip_special_tokens=False)
                    full_txt = tok.decode(f_resp[j, : L1 + L2].tolist(), skip_special_tokens=False)
                    boxed_tok = "\\boxed"
                    has_think = "</think>" in full_txt
                    has_answer = "<answer>" in full_txt
                    has_boxed = boxed_tok in full_txt
                    tail = full_txt[-200:]
                    print(
                        f"[two_pass_debug mode={m}] L1={L1} L2={L2} "
                        f"has</think>={has_think} has<answer>={has_answer} has_boxed={has_boxed}\n"
                        f"  PASS1     : {pass1_txt!r}\n"
                        f"  PASS2[:40]: {pass2_txt!r}\n"
                        f"  TAIL      : {tail!r}",
                        flush=True,
                    )

            chunks.append(sub)
            srcs.append(src_base[rows_np])

        # ── Pass-1-only rows (finished or early-stopped unknown) ─────────────
        if leftover:
            rows_np = np.array(leftover, dtype=np.int64)
            sub = out1.select_idxs(rows_np)
            sub.non_tensor_batch["routing_mode"] = mode_l[rows_np]
            sub.non_tensor_batch["routing_token_length"] = np.array(
                [_routing_token_length_for_mode(str(mode_l[i]), ctrl) for i in leftover],
                dtype=np.int64,
            )
            sub.non_tensor_batch["routing_forced"] = np.zeros(len(leftover), dtype=bool)
            chunks.append(sub)
            srcs.append(src_base[rows_np])

        unknown_stopped = sum(1 for i in leftover if mode_l[i] == "unknown" and not finished[i])
        metrics = {
            "three_mode_routing/two_pass_active": 1.0,
            "three_mode_routing/two_pass_continued_nothink": float(len(cont_groups.get("nothink", []))),
            "three_mode_routing/two_pass_continued_short": float(len(cont_groups.get("short", []))),
            "three_mode_routing/two_pass_continued_long": float(len(cont_groups.get("long", []))),
            "three_mode_routing/two_pass_unknown_stopped": float(unknown_stopped),
            "three_mode_routing/two_pass_finished_pass1": float(finished.sum()),
        }
        return chunks, srcs, metrics


# ─── Balance-coef annealing ───────────────────────────────────────────────────


def effective_balance_coef(
    balance_coef: float,
    anneal_start_step: int,
    anneal_end_step: int,
    anneal_final_coef: float,
    global_step: int,
) -> float:
    """Balance coefficient at `global_step` under the linear anneal schedule.

    Disabled (returns balance_coef unchanged) unless anneal_end > anneal_start > 0.
    Constant at balance_coef up to anneal_start, linearly interpolated to
    anneal_final_coef at anneal_end, constant afterwards.
    """
    start = int(anneal_start_step)
    end = int(anneal_end_step)
    if not (end > start > 0):
        return float(balance_coef)
    if global_step <= start:
        return float(balance_coef)
    if global_step >= end:
        return float(anneal_final_coef)
    frac = (global_step - start) / float(end - start)
    return float(balance_coef) + frac * (float(anneal_final_coef) - float(balance_coef))


# ─── Advantage computation ────────────────────────────────────────────────────


def compute_three_mode_routing_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    routing_mode: Optional[np.ndarray] = None,
    routing_token_length: Optional[np.ndarray] = None,
    routing_forced: Optional[np.ndarray] = None,
    reasoning_lengths: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
    global_step: int = 0,
    epsilon: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token advantages for three-mode routing GRPO.

    Reward shaping (pre group-normalization):
      - unknown routing token → unknown_penalty (gated on solved groups during warmup)
      - correct + nothink    → nothink_base × gamma_nothink^L
      - correct + short      → short_base   × gamma_short^L
      - correct + long       → long_base × gamma_long^L (gamma_long defaults to 1.0)
      - incorrect + nothink  → nothink_wrong_penalty (default 0.0)
      - incorrect + short    → short_wrong_penalty   (default 0.0)
      - incorrect + long     → long_wrong_penalty    (default 0.0)

    where L = response_length − routing_token_length.  When three_mode_routing.
    reasoning_only is True, response_length is replaced by reasoning_lengths (token
    count before the first closing </think> tag, computed upstream via
    compute_reasoning_token_statistics) — so only tokens inside the think block are
    discounted.

    Group normalization: mean-centering only (std normalization disabled).  When
    exclude_unknown_from_mean (default True), unknown rollouts do not contribute to
    the group mean — so a large unknown_penalty cannot give incorrect-but-valid
    groupmates a positive advantage.  Unknowns are still centered against the
    valid-rollout mean, keeping their own advantage strongly negative.

    Balance term (advantage level): balance_coef × (balance_target − frac_mode) added
    after normalization, pushing the three-way split toward balance_target (default
    1/3 each).  When balance_free_only (default True) the fractions are computed over
    and the nudge applied to FREE rollouts only — forced rollouts are balanced by
    construction and would otherwise dilute the signal during warmup.
    """
    tmr_cfg = config.get("three_mode_routing") if config is not None else None
    nothink_base = float(tmr_cfg.get("nothink_base", 2.0)) if tmr_cfg is not None else 2.0
    short_base = float(tmr_cfg.get("short_base", 1.5)) if tmr_cfg is not None else 1.5
    long_base = float(tmr_cfg.get("long_base", 1.0)) if tmr_cfg is not None else 1.0
    gamma_nothink = float(tmr_cfg.get("gamma_nothink", 0.99984)) if tmr_cfg is not None else 0.99984
    gamma_short = float(tmr_cfg.get("gamma_short", 0.999912)) if tmr_cfg is not None else 0.999912
    gamma_long = float(tmr_cfg.get("gamma_long", 1.0)) if tmr_cfg is not None else 1.0
    reasoning_only = bool(tmr_cfg.get("reasoning_only", False)) if tmr_cfg is not None else False
    unknown_penalty = float(tmr_cfg.get("unknown_penalty", -0.5)) if tmr_cfg is not None else -0.5
    balance_coef = float(tmr_cfg.get("balance_coef", 0.5)) if tmr_cfg is not None else 0.5
    balance_target = float(tmr_cfg.get("balance_target", 1.0 / 3.0)) if tmr_cfg is not None else 1.0 / 3.0
    solved_gate_warmup = int(tmr_cfg.get("solved_gate_warmup_steps", 0)) if tmr_cfg is not None else 0
    nothink_wrong_penalty = float(tmr_cfg.get("nothink_wrong_penalty", 0.0)) if tmr_cfg is not None else 0.0
    short_wrong_penalty = float(tmr_cfg.get("short_wrong_penalty", 0.0)) if tmr_cfg is not None else 0.0
    long_wrong_penalty = float(tmr_cfg.get("long_wrong_penalty", 0.0)) if tmr_cfg is not None else 0.0
    balance_free_only = bool(tmr_cfg.get("balance_free_only", True)) if tmr_cfg is not None else True
    balance_correctness_gated = (
        bool(tmr_cfg.get("balance_correctness_gated", False)) if tmr_cfg is not None else False
    )
    balance_protect_long_incorrect = (
        bool(tmr_cfg.get("balance_protect_long_incorrect", False)) if tmr_cfg is not None else False
    )
    exclude_unknown_from_mean = (
        bool(tmr_cfg.get("exclude_unknown_from_mean", True)) if tmr_cfg is not None else True
    )
    exclude_unknown_start = (
        int(tmr_cfg.get("exclude_unknown_from_mean_start_step", 0)) if tmr_cfg is not None else 0
    )
    # Exclusion only after the start step: during bootstrap the unknowns' drag on
    # the mean is the positive push on forced routing tokens (see config docstring).
    exclude_unknown_from_mean = exclude_unknown_from_mean and global_step >= exclude_unknown_start

    # Balance-coef annealing (no-op unless end > start > 0).
    balance_coef = effective_balance_coef(
        balance_coef=balance_coef,
        anneal_start_step=int(tmr_cfg.get("balance_anneal_start_step", 0)) if tmr_cfg is not None else 0,
        anneal_end_step=int(tmr_cfg.get("balance_anneal_end_step", 0)) if tmr_cfg is not None else 0,
        anneal_final_coef=float(tmr_cfg.get("balance_anneal_final_coef", 0.0)) if tmr_cfg is not None else 0.0,
        global_step=global_step,
    )

    with torch.no_grad():
        raw_scores = token_level_rewards.sum(dim=-1)  # (bs,)
        response_lengths = response_mask.sum(dim=-1)
        bsz = raw_scores.shape[0]

        if reasoning_only and reasoning_lengths is not None:
            discount_base_lengths = reasoning_lengths.to(device=response_lengths.device)
        else:
            discount_base_lengths = response_lengths

        mode_arr = np.array(
            [str(routing_mode[i]).lower() if routing_mode is not None else "unknown" for i in range(bsz)],
            dtype=object,
        )
        nt_mask_np = mode_arr == "nothink"
        sh_mask_np = mode_arr == "short"
        lg_mask_np = mode_arr == "long"
        unk_mask_np = mode_arr == "unknown"

        # ── Solved-group gate ──────────────────────────────────────────────────
        # During steps 0..solved_gate_warmup: only shape groups with >=1 correct answer.
        # After warmup: shape ALL groups (so all-wrong hard groups get balance gradient).
        gate_active = solved_gate_warmup > 0 and global_step <= solved_gate_warmup
        is_correct_np = (raw_scores > 0.5).detach().cpu().numpy()
        group_has_correct: dict[Any, bool] = defaultdict(bool)
        for i in range(bsz):
            if is_correct_np[i]:
                group_has_correct[index[i]] = True
        solved_np = np.array([group_has_correct[index[i]] for i in range(bsz)], dtype=bool)

        # ── Shape rewards ──────────────────────────────────────────────────────
        shaped = raw_scores.clone()
        for i in range(bsz):
            mode = mode_arr[i]
            n_rt = int(routing_token_length[i]) if routing_token_length is not None else 0
            L = max(int(discount_base_lengths[i].item()) - n_rt, 0)
            correct = bool(is_correct_np[i])
            in_shaped_group = not gate_active or solved_np[i]

            if mode == "unknown":
                shaped[i] = unknown_penalty if in_shaped_group else raw_scores[i]
            elif correct:
                if mode == "nothink":
                    shaped[i] = nothink_base * (gamma_nothink ** L)
                elif mode == "short":
                    shaped[i] = short_base * (gamma_short ** L)
                else:  # long (gamma_long defaults to 1.0 = no discount)
                    shaped[i] = long_base * (gamma_long ** L)
            else:
                if mode == "nothink":
                    shaped[i] = shaped.new_tensor(nothink_wrong_penalty)
                elif mode == "short":
                    shaped[i] = shaped.new_tensor(short_wrong_penalty)
                else:  # long
                    shaped[i] = shaped.new_tensor(long_wrong_penalty)

        # ── Group mean-centering (no std division) ─────────────────────────────
        # When exclude_unknown_from_mean, unknown rollouts are left out of the mean:
        # otherwise a large unknown_penalty drags the group mean down and incorrect
        # valid rollouts in the same group come out with a POSITIVE advantage
        # (e.g. G=4, one unknown at -2 + three wrong at 0 → mean -0.5 → wrong get +0.5).
        # Groups whose rollouts are all unknown fall back to the full-group mean.
        id2scores: dict[Any, list] = defaultdict(list)
        id2valid_scores: dict[Any, list] = defaultdict(list)
        for i in range(bsz):
            id2scores[index[i]].append(shaped[i])
            if not unk_mask_np[i]:
                id2valid_scores[index[i]].append(shaped[i])

        id2mean: dict[Any, torch.Tensor] = {}
        for uid, sc_list in id2scores.items():
            if len(sc_list) <= 1:
                id2mean[uid] = shaped.new_tensor(0.0)
                continue
            valid_list = id2valid_scores.get(uid, [])
            mean_list = valid_list if (exclude_unknown_from_mean and len(valid_list) > 0) else sc_list
            id2mean[uid] = torch.stack(mean_list).mean()

        seq_advantages = shaped.new_zeros(bsz)
        for i in range(bsz):
            seq_advantages[i] = shaped[i] - id2mean[index[i]]

        # ── Balance term (advantage level) ─────────────────────────────────────
        # Applied only to valid-routing rollouts in shaped groups, so the push
        # toward the target split is meaningful only where routing shaping applies.
        # With balance_free_only, fractions and nudges cover free rollouts only:
        # forced rollouts are balanced by construction and would otherwise dilute
        # the fractions toward the target during warmup, hiding a collapse of the
        # free policy.
        if balance_coef != 0.0:
            shaped_mask = solved_np if gate_active else np.ones(bsz, dtype=bool)
            if balance_free_only and routing_forced is not None:
                shaped_mask = shaped_mask & ~np.asarray(routing_forced, dtype=bool)
            bal_nt = nt_mask_np & shaped_mask
            bal_sh = sh_mask_np & shaped_mask
            bal_lg = lg_mask_np & shaped_mask
            n_nt = int(bal_nt.sum())
            n_sh = int(bal_sh.sum())
            n_lg = int(bal_lg.sum())
            n_valid = n_nt + n_sh + n_lg
            if n_valid > 0:
                frac_nt = n_nt / n_valid
                frac_sh = n_sh / n_valid
                frac_lg = n_lg / n_valid
                balance = seq_advantages.new_zeros(bsz)
                balance[torch.as_tensor(bal_nt, device=seq_advantages.device)] = (
                    balance_coef * (balance_target - frac_nt)
                )
                balance[torch.as_tensor(bal_sh, device=seq_advantages.device)] = (
                    balance_coef * (balance_target - frac_sh)
                )
                balance[torch.as_tensor(bal_lg, device=seq_advantages.device)] = (
                    balance_coef * (balance_target - frac_lg)
                )
                # Correctness-gated nudge: keep a positive (encourage) nudge only on
                # correct rollouts and a negative (discourage) nudge only on wrong
                # ones, so the balance signal never penalizes a correct answer nor
                # rewards a wrong one.  Pressure to drop an over-used mode concentrates
                # where it is failing; pressure to grow an under-used mode lands where
                # it is succeeding.
                if balance_correctness_gated:
                    correct_t = torch.as_tensor(is_correct_np, device=seq_advantages.device)
                    keep = ((balance > 0) & correct_t) | ((balance < 0) & ~correct_t)
                    balance = torch.where(keep, balance, balance.new_zeros(()))
                # Never let the load-balancer penalize a wrong LONG attempt: the hardest
                # problems (most often wrong) should remain free to route to LONG.
                if balance_protect_long_incorrect:
                    long_wrong = torch.as_tensor(
                        lg_mask_np & ~is_correct_np, device=seq_advantages.device
                    )
                    balance = torch.where(long_wrong, balance.new_zeros(()), balance)
                seq_advantages = seq_advantages + balance

        token_advantages = seq_advantages.unsqueeze(-1) * response_mask

    return token_advantages, token_advantages


# ─── Metrics ──────────────────────────────────────────────────────────────────


def compute_three_mode_routing_metrics(
    routing_mode: np.ndarray,
    is_correct: torch.Tensor,
    response_lengths: torch.Tensor,
    index: np.ndarray,
    advantages: Optional[torch.Tensor] = None,
    routing_forced: Optional[np.ndarray] = None,
    gate_active: bool = False,
    was_truncated_by_cap: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Per-step diagnostics for the three-mode routing protocol.

    When routing_forced is provided, mode distribution and unknown_fraction
    are computed over free rollouts only (excluding forced ones).
    advantages is optional — omit during validation to skip mean_advantage metrics.
    """
    is_correct_np = is_correct.detach().cpu().numpy().astype(bool)
    lengths_np = response_lengths.detach().cpu().numpy()
    advantages_np = advantages.sum(dim=-1).detach().cpu().numpy() if advantages is not None else None

    mode_arr = np.array([str(m).lower() for m in routing_mode], dtype=object)

    # Masks per mode (all rollouts)
    nt_mask = mode_arr == "nothink"
    sh_mask = mode_arr == "short"
    lg_mask = mode_arr == "long"
    unk_mask = mode_arr == "unknown"

    # Free-only masks for distribution and entropy
    if routing_forced is not None:
        free_mask = ~routing_forced.astype(bool)
    else:
        free_mask = np.ones(len(mode_arr), dtype=bool)

    total_free = float(free_mask.sum()) if free_mask.sum() > 0 else 1.0
    free_mode = mode_arr[free_mask]
    nt_frac = float((free_mode == "nothink").sum()) / total_free
    sh_frac = float((free_mode == "short").sum()) / total_free
    lg_frac = float((free_mode == "long").sum()) / total_free
    unk_frac = float((free_mode == "unknown").sum()) / total_free

    # Routing entropy over free rollouts
    probs = np.array([nt_frac, sh_frac, lg_frac])
    probs_safe = np.clip(probs, 1e-9, 1.0)
    routing_entropy = float(-np.sum(probs * np.log(probs_safe)))

    def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        return float(arr[mask].mean()) if mask.any() else 0.0

    metrics = {
        # Mode distribution (free rollouts)
        "three_mode_routing/nothink_fraction": nt_frac,
        "three_mode_routing/short_fraction": sh_frac,
        "three_mode_routing/long_fraction": lg_frac,
        "three_mode_routing/unknown_fraction": unk_frac,
        "three_mode_routing/routing_entropy": routing_entropy,
        # Correctness per mode (all rollouts)
        "three_mode_routing/nothink_correct_rate": _safe_mean(is_correct_np.astype(float), nt_mask),
        "three_mode_routing/short_correct_rate": _safe_mean(is_correct_np.astype(float), sh_mask),
        "three_mode_routing/long_correct_rate": _safe_mean(is_correct_np.astype(float), lg_mask),
        # Mean length per mode
        "three_mode_routing/nothink_mean_length": _safe_mean(lengths_np, nt_mask),
        "three_mode_routing/short_mean_length": _safe_mean(lengths_np, sh_mask),
        "three_mode_routing/long_mean_length": _safe_mean(lengths_np, lg_mask),
        # Mean advantage per mode (omitted when advantages not provided, e.g. validation)
        **({
            "three_mode_routing/nothink_mean_advantage": _safe_mean(advantages_np, nt_mask),
            "three_mode_routing/short_mean_advantage": _safe_mean(advantages_np, sh_mask),
            "three_mode_routing/long_mean_advantage": _safe_mean(advantages_np, lg_mask),
            "three_mode_routing/unknown_mean_advantage": _safe_mean(advantages_np, unk_mask),
        } if advantages_np is not None else {}),
        # Schedule / state
        "three_mode_routing/gate_active": float(gate_active),
    }

    # Truncation per mode (only when cap application info is provided, i.e. training)
    if was_truncated_by_cap is not None:
        trunc = was_truncated_by_cap.astype(float)
        metrics["three_mode_routing/nothink_truncated_fraction"] = _safe_mean(trunc, nt_mask)
        metrics["three_mode_routing/short_truncated_fraction"] = _safe_mean(trunc, sh_mask)
        metrics["three_mode_routing/long_truncated_fraction"] = _safe_mean(trunc, lg_mask)

    # Group mixing: among free rollouts, how diverse are the mode choices per prompt?
    # Requires >=2 free rollouts per group to be meaningful (i.e. post-warmup).
    free_mask_for_mixing = ~routing_forced.astype(bool) if routing_forced is not None else np.ones(len(mode_arr), dtype=bool)
    free_index = index[free_mask_for_mixing]
    free_modes_mix = mode_arr[free_mask_for_mixing]
    valid_mode_set = {"nothink", "short", "long"}
    unique_groups = np.unique(free_index) if len(free_index) > 0 else []
    unique_modes_counts = []
    for g in unique_groups:
        g_modes = free_modes_mix[free_index == g]
        valid = [m for m in g_modes if m in valid_mode_set]
        if len(valid) >= 2:
            unique_modes_counts.append(len(set(valid)))
    if unique_modes_counts:
        metrics["three_mode_routing/mean_unique_modes_per_group"] = float(np.mean(unique_modes_counts))
        metrics["three_mode_routing/frac_mixed_groups"] = float(np.mean([c > 1 for c in unique_modes_counts]))

    return metrics


# ─── Per-mode response cap ────────────────────────────────────────────────────


def apply_three_mode_caps(
    batch,
    tokenizer: Any,
    config: ThreeModeRoutingConfig,
    known_modes: Optional[np.ndarray] = None,
    known_forced: Optional[np.ndarray] = None,
    routing_token_lengths: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncate each rollout's response to its mode's token cap, in place.

    Mutates batch.batch["attention_mask"] so that responses beyond the per-mode
    cap are masked out.  Applied before reward and training so a response that
    exceeds the cap is scored wrong — creating a real capability gap the router
    can learn from.  unknown rollouts are left uncapped.

    The cap measures tokens AFTER the routing token (same L as the reward
    formula), so the effective response budget is cap + routing_token_length.
    Without this allowance, a forced rollout that used its full generation
    budget gets its promoted routing token pushed past the cap and is flagged
    truncated + clipped by n_rt tokens.

    known_modes / known_forced: previously assigned routing labels and the forced
    flag per rollout.  Where known_forced is True the stored label is TRUSTED and
    re-detection is skipped — a forced rollout's mode is certain by construction,
    and re-detecting it can only mislabel it (e.g. an attached continuation token
    defeating the regex), which would wrongly charge forced rollouts the
    unknown_penalty.  Free rollouts are (re-)detected as before.

    routing_token_lengths: per-rollout routing token length for the cap
    allowance.  Falls back to the canonical token length of the detected mode.
    Pass zeros when the routing token is NOT part of the response (forced-mode
    validation, where the token sits in the prompt).

    Returns (routing_mode array, was_truncated bool array).
    """
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    bsz, resp_len = responses.shape
    prompt_len = attention_mask.shape[1] - resp_len

    resp_attn = attention_mask[:, prompt_len:]
    modes = detect_three_mode_routing_modes(responses, resp_attn, tokenizer, config)
    if known_modes is not None and known_forced is not None:
        forced_arr = np.asarray(known_forced, dtype=bool)
        for i in range(bsz):
            if forced_arr[i]:
                modes[i] = known_modes[i]

    cap_map = {
        "nothink": config.nothink_max_tokens,
        "short": config.short_max_tokens,
        "long": config.long_max_tokens,
    }
    canonical_rt_len = {
        "nothink": len(tokenizer.encode(config.nothink_token, add_special_tokens=False)) if config.nothink_token else 0,
        "short": len(tokenizer.encode(config.short_token, add_special_tokens=False)) if config.short_token else 0,
        "long": len(tokenizer.encode(config.long_token, add_special_tokens=False)) if config.long_token else 0,
    }
    valid_len = resp_attn.sum(dim=-1)
    was_truncated = np.zeros(bsz, dtype=bool)
    for i in range(bsz):
        mode = str(modes[i]).lower()
        cap = cap_map.get(mode)
        if cap is None:
            continue
        if routing_token_lengths is not None:
            n_rt = int(routing_token_lengths[i])
        else:
            n_rt = canonical_rt_len.get(mode, 0)
        effective_cap = int(cap) + max(0, n_rt)
        if int(valid_len[i].item()) > effective_cap:
            attention_mask[i, prompt_len + effective_cap:] = 0
            was_truncated[i] = True

    batch.batch["attention_mask"] = attention_mask

    return modes, was_truncated
