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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
import re
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


_THINKING_CLOSE_TAG_PATTERN = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)


def _tokenize_text_length(text: str, tokenizer: Any) -> int:
    """Return token length for text without adding special tokens."""
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return len(tokenized["input_ids"])


def _extract_reasoning_spans(response_text: str) -> tuple[list[str], bool]:
    """Extract reasoning text for prompt-prefilled <think> format.

    The opening think tag is expected in the prompt prefix, so response text starts
    inside think and ends reasoning at the first closing think tag.
    """
    close_match = _THINKING_CLOSE_TAG_PATTERN.search(response_text)
    if close_match is not None:
        prefix = response_text[: close_match.start()]
        if prefix.strip():
            return [prefix], True

        return [], True

    return [], False


def _compute_reasoning_token_lengths(
    batch: DataProto, response_length: torch.Tensor, tokenizer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sample reasoning token counts and whether reasoning is fully observed.

    A sample is considered fully observed when either:
    1) A closing think tag is present, or
    2) The response is clipped at max response length without a closing think tag.
    """
    responses = batch.batch["responses"]
    max_response_length = responses.size(1)
    lengths = []
    closed_think = []

    for i in range(responses.size(0)):
        valid_len = int(response_length[i].item())
        if valid_len <= 0:
            lengths.append(0.0)
            closed_think.append(False)
            continue

        response_ids = responses[i, :valid_len].tolist()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        spans, has_close_tag = _extract_reasoning_spans(response_text)
        reasoning_tokens = 0
        reasoning_fully_observed = bool(has_close_tag)
        if has_close_tag:
            for span in spans:
                reasoning_tokens += _tokenize_text_length(span, tokenizer)
        elif valid_len >= max_response_length:
            # Truncated at max length without a close tag: treat full response as reasoning.
            reasoning_tokens = valid_len
            reasoning_fully_observed = True

        lengths.append(float(reasoning_tokens))
        closed_think.append(reasoning_fully_observed)

    return (
        torch.tensor(lengths, dtype=torch.float32, device=response_length.device),
        torch.tensor(closed_think, dtype=torch.bool, device=response_length.device),
    )


def compute_reasoning_token_statistics(
    batch: DataProto,
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reasoning-token mask/statistics from prompt-prefilled think responses.

    Used by three_mode_routing.reasoning_only: the gamma^L discount is then based on
    the token count before the first closing </think> tag instead of the full
    response length.  Responses that never close </think> get length 0 (no discount).

    Returns:
        reasoning_mask: Bool tensor of shape (bs, response_len_max) with reasoning tokens set True.
        reasoning_lengths: Float tensor of shape (bs,) containing reasoning token counts K.
        closed_think: Bool tensor of shape (bs,) indicating whether a closing think tag was found.
        valid_response_lengths: Long tensor of shape (bs,) with valid response lengths from attention mask.
    """
    responses = batch.batch["responses"]
    response_len_max = responses.size(1)
    response_mask = batch.batch["attention_mask"][:, -response_len_max:]
    valid_response_lengths = response_mask.sum(-1).to(dtype=torch.long)

    reasoning_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    reasoning_lengths = torch.zeros(responses.size(0), dtype=torch.float32, device=responses.device)
    closed_think = torch.zeros(responses.size(0), dtype=torch.bool, device=responses.device)

    for i in range(responses.size(0)):
        valid_len = int(valid_response_lengths[i].item())
        if valid_len <= 0:
            continue

        response_ids = responses[i, :valid_len].tolist()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        spans, has_close_tag = _extract_reasoning_spans(response_text)
        closed_think[i] = bool(has_close_tag)
        if not has_close_tag:
            continue

        reasoning_token_count = 0
        for span in spans:
            reasoning_token_count += _tokenize_text_length(span, tokenizer)

        reasoning_token_count = min(int(reasoning_token_count), valid_len)
        reasoning_lengths[i] = float(reasoning_token_count)
        if reasoning_token_count > 0:
            reasoning_mask[i, :reasoning_token_count] = True

    return reasoning_mask, reasoning_lengths, closed_think, valid_response_lengths


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True, tokenizer: Any | None = None) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - response_length/mean_correct, mean_incorrect: Mean response length split by correctness
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
                        - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
                        - response_length/mean_reasoning: Mean token length inside thinking tags
                        - response_length/mean_reasoning_correct, mean_reasoning_incorrect:
                            Mean reasoning token length split by correctness
                        - response_length/incorrect_non_aborted_mean: Mean response length for incorrect,
                            non-aborted samples
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    # Incorrect samples excluding aborted responses.
    incorrect_mask = sequence_score < 0.5
    correct_mask = ~incorrect_mask
    correct_non_aborted_mask = correct_mask & non_aborted_mask
    incorrect_non_aborted_mask = incorrect_mask & non_aborted_mask

    correct_non_aborted_response_length = response_length[correct_non_aborted_mask]
    if correct_non_aborted_response_length.numel() > 0:
        correct_non_aborted_response_length_mean = torch.mean(correct_non_aborted_response_length).detach().item()
    else:
        correct_non_aborted_response_length_mean = 0.0

    incorrect_non_aborted_response_length = response_length[incorrect_non_aborted_mask]
    if incorrect_non_aborted_response_length.numel() > 0:
        incorrect_non_aborted_response_length_mean = (
            torch.mean(incorrect_non_aborted_response_length).detach().item()
        )
    else:
        incorrect_non_aborted_response_length_mean = 0.0

    # Mean token length inside <think>/<thinking> blocks, excluding aborted samples.
    if tokenizer is not None:
        reasoning_token_lengths, reasoning_closed_mask = _compute_reasoning_token_lengths(batch, response_length, tokenizer)
        reasoning_valid_mask = non_aborted_mask & reasoning_closed_mask
        valid_reasoning_lengths = reasoning_token_lengths[reasoning_valid_mask]
        if valid_reasoning_lengths.numel() > 0:
            mean_reasoning_length = torch.mean(valid_reasoning_lengths).detach().item()
        else:
            mean_reasoning_length = 0.0

        reasoning_correct_mask = reasoning_valid_mask & correct_mask
        valid_reasoning_lengths_correct = reasoning_token_lengths[reasoning_correct_mask]
        if valid_reasoning_lengths_correct.numel() > 0:
            mean_reasoning_length_correct = torch.mean(valid_reasoning_lengths_correct).detach().item()
        else:
            mean_reasoning_length_correct = 0.0

        reasoning_incorrect_mask = reasoning_valid_mask & incorrect_mask
        valid_reasoning_lengths_incorrect = reasoning_token_lengths[reasoning_incorrect_mask]
        if valid_reasoning_lengths_incorrect.numel() > 0:
            mean_reasoning_length_incorrect = torch.mean(valid_reasoning_lengths_incorrect).detach().item()
        else:
            mean_reasoning_length_incorrect = 0.0
    else:
        mean_reasoning_length = 0.0
        mean_reasoning_length_correct = 0.0
        mean_reasoning_length_incorrect = 0.0

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/mean_correct": correct_non_aborted_response_length_mean,
        "response_length/mean_incorrect": incorrect_non_aborted_response_length_mean,
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        "response_length/mean_reasoning": mean_reasoning_length,
        "response_length/mean_reasoning_correct": mean_reasoning_length_correct,
        "response_length/mean_reasoning_incorrect": mean_reasoning_length_incorrect,
        "response_length/incorrect_non_aborted_mean": incorrect_non_aborted_response_length_mean,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_completion_metrics(batch: DataProto, generation_budget: int) -> dict[str, Any]:
    """Compute fractions of truncated / finished responses and their correctness.

    Args:
        batch: DataProto after rollout + reward computation.
        generation_budget: The effective max number of response tokens allowed
            during generation (e.g., max_response_length or adaptive window).
    """
    response_info = _compute_response_info(batch)
    response_length = response_info["response_length"]  # (batch_size,)

    # Identify responses that hit (or exceed) the generation budget.
    truncated_mask = response_length >= generation_budget
    finished_mask = ~truncated_mask

    batch_size = response_length.shape[0]
    if batch_size == 0:
        return {
            "completion/truncated_frac": 0.0,
            "completion/finished_frac": 0.0,
            "completion/truncated_correct_frac": 0.0,
            "completion/finished_correct_frac": 0.0,
            "completion/correct_mean_length": 0.0,
            "completion/success_rate": 0.0,
        }

    truncated_frac = truncated_mask.float().mean().item()
    finished_frac = finished_mask.float().mean().item()

    # Sequence-level raw task score (before KL), e.g. 0, 0.1, 1.0 for countdown.
    if "token_level_scores" in batch.batch:
        token_level_scores = batch.batch["token_level_scores"]
        seq_scores = token_level_scores.sum(-1)
        correct_mask = seq_scores >= 0.99  # treat ~1.0 as strictly correct
    else:
        # Fallback: no notion of correctness
        correct_mask = torch.zeros_like(response_length, dtype=torch.bool)

    truncated_count = truncated_mask.float().sum().item()
    finished_count = finished_mask.float().sum().item()

    correct_count = correct_mask.float().sum().item()
    if correct_count > 0:
        correct_mean_length = response_length[correct_mask].float().mean().item()
    else:
        correct_mean_length = 0.0

    if truncated_count > 0:
        truncated_correct_frac = (truncated_mask & correct_mask).float().sum().item() / truncated_count
    else:
        truncated_correct_frac = 0.0

    if finished_count > 0:
        finished_correct_frac = (finished_mask & correct_mask).float().sum().item() / finished_count
    else:
        finished_correct_frac = 0.0

    # Overall success rate (fraction of correct answers)
    overall_success_rate = correct_count / batch_size if batch_size > 0 else 0.0

    metrics = {
        "completion/truncated_frac": truncated_frac,
        "completion/finished_frac": finished_frac,
        "completion/truncated_correct_frac": truncated_correct_frac,
        "completion/finished_correct_frac": finished_correct_frac,
        "completion/correct_mean_length": correct_mean_length,
        "completion/success_rate": overall_success_rate,
    }

    # Add UUID-based group metrics if available
    if "uid" in batch.non_tensor_batch:
        group_metrics = compute_group_success_metrics(batch, correct_mask)
        metrics.update(group_metrics)

    return metrics


def compute_group_success_metrics(batch: DataProto, correct_mask: torch.Tensor) -> dict[str, Any]:
    """Compute success metrics for GRPO groups (responses sharing the same prompt).

    Adaptive groups can produce variable group sizes per step (e.g. 2 for easy
    prompts and 30 for hard ones), so we bucket groups by their *correctness
    fraction* (constant 4 buckets) rather than by absolute correct count
    (which used to emit 2 metrics per possible group size, blowing up the
    completion tab).

    Buckets:
        all_wrong         frac == 0          (no learning signal)
        minority_correct  0 < frac < 0.5
        majority_correct  0.5 <= frac < 1
        all_correct       frac == 1          (no learning signal in vanilla GRPO)
    """
    uids = batch.non_tensor_batch["uid"]

    uid_to_correct = defaultdict(list)
    response_info = _compute_response_info(batch)
    response_lengths = response_info["response_length"]
    uid_to_response_lengths = defaultdict(list)

    for i, uid in enumerate(uids):
        is_correct = correct_mask[i].item() if torch.is_tensor(correct_mask[i]) else correct_mask[i]
        uid_to_correct[uid].append(bool(is_correct))
        uid_to_response_lengths[uid].append(float(response_lengths[i].item()))

    total_groups = len(uid_to_correct)
    if total_groups == 0:
        return {}

    bucket_counts = {"all_wrong": 0, "minority_correct": 0, "majority_correct": 0, "all_correct": 0}
    bucket_lengths: dict[str, list[float]] = {k: [] for k in bucket_counts}
    correct_fractions: list[float] = []
    group_sizes: list[int] = []

    for uid, correctness_list in uid_to_correct.items():
        n = len(correctness_list)
        if n == 0:
            continue
        frac = sum(correctness_list) / n
        correct_fractions.append(frac)
        group_sizes.append(n)
        if frac == 0.0:
            bucket = "all_wrong"
        elif frac == 1.0:
            bucket = "all_correct"
        elif frac < 0.5:
            bucket = "minority_correct"
        else:
            bucket = "majority_correct"
        bucket_counts[bucket] += 1
        group_lengths = uid_to_response_lengths[uid]
        if group_lengths:
            bucket_lengths[bucket].append(float(np.mean(group_lengths)))

    metrics: dict[str, Any] = {}
    for bucket, count in bucket_counts.items():
        metrics[f"completion/groups/{bucket}_pct"] = 100.0 * count / total_groups
        lengths = bucket_lengths[bucket]
        metrics[f"completion/groups/{bucket}_mean_length"] = (
            float(np.mean(lengths)) if lengths else 0.0
        )

    metrics["completion/groups/mean_correct_frac"] = float(np.mean(correct_fractions))
    metrics["completion/groups/mean_size"] = float(np.mean(group_sizes))

    return metrics


def compute_difficulty_metrics(batch: DataProto) -> dict[str, Any]:
    """Compute accuracy broken down by difficulty level (e.g. 3 vs 4).

    We assume that non_tensor_batch['reward_model'][i]['ground_truth'] contains
    a 'difficulty' field for each sample.
    """
    if "token_level_scores" not in batch.batch:
        return {}

    # Sequence-level raw scores to determine correctness.
    token_level_scores = batch.batch["token_level_scores"]
    seq_scores = token_level_scores.sum(-1)
    correct_mask = seq_scores >= 0.99  # strictly correct

    rm_info = batch.non_tensor_batch.get("reward_model", None)
    if rm_info is None:
        return {}

    # Extract per-sample difficulty if available.
    difficulties = []
    for i in range(len(seq_scores)):
        info = rm_info[i]
        gt = info.get("ground_truth", {})
        if isinstance(gt, dict):
            difficulties.append(gt.get("difficulty", None))
        else:
            difficulties.append(None)

    # Compute counts and accuracies for difficulty 3 and 4.
    diff3_total = 0
    diff3_correct = 0
    diff4_total = 0
    diff4_correct = 0

    for i, d in enumerate(difficulties):
        if d == 3:
            diff3_total += 1
            if correct_mask[i]:
                diff3_correct += 1
        elif d == 4:
            diff4_total += 1
            if correct_mask[i]:
                diff4_correct += 1

    metrics = {}
    if diff3_total > 0:
        metrics["difficulty/3_acc"] = diff3_correct / diff3_total
        metrics["difficulty/3_count"] = float(diff3_total)
    else:
        metrics["difficulty/3_acc"] = 0.0
        metrics["difficulty/3_count"] = 0.0

    if diff4_total > 0:
        metrics["difficulty/4_acc"] = diff4_correct / diff4_total
        metrics["difficulty/4_count"] = float(diff4_total)
    else:
        metrics["difficulty/4_acc"] = 0.0
        metrics["difficulty/4_count"] = 0.0

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    metric[f"all_correct@{n_resps}"] = float(np.min(var_vals) >= 1.0)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
