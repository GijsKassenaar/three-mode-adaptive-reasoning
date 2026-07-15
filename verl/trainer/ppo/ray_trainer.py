# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import DataProtoConfig, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_completion_metrics,
    compute_data_metrics,
    compute_difficulty_metrics,
    compute_reasoning_token_statistics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.three_mode_routing import (
    ThreeModeRoutingConfig,
    ThreeModeRoutingForcer,
    apply_three_mode_caps,
    compute_three_mode_routing_metrics,
    effective_balance_coef,
    enabled_modes_from_config,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    tokenizer: Any = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.
        tokenizer: Optional tokenizer, required by three_mode_routing.reasoning_only (the gamma^L
            discount over reasoning tokens needs to decode/re-tokenize responses).

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        if adv_estimator == AdvantageEstimator.THREE_MODE_ROUTING:
            if "routing_mode" not in data.non_tensor_batch:
                raise ValueError(
                    "routing_mode is required in non_tensor_batch for three_mode_routing estimator. "
                    "Enable algorithm.three_mode_routing.enable=True to activate rollout building."
                )
            adv_kwargs["routing_mode"] = data.non_tensor_batch["routing_mode"]
            if "routing_token_length" in data.non_tensor_batch:
                adv_kwargs["routing_token_length"] = data.non_tensor_batch["routing_token_length"]
            if "routing_forced" in data.non_tensor_batch:
                # Lets the balance term restrict itself to free rollouts (balance_free_only).
                adv_kwargs["routing_forced"] = data.non_tensor_batch["routing_forced"]
            adv_kwargs["global_step"] = int((data.meta_info or {}).get("global_steps", 0))

            tmr_cfg_rt = config.get("three_mode_routing") if config is not None else None
            if tmr_cfg_rt is not None and bool(tmr_cfg_rt.get("reasoning_only", False)):
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer is required when algorithm.three_mode_routing.reasoning_only=True"
                    )
                _, reasoning_lengths_tmr, _, _ = compute_reasoning_token_statistics(data, tokenizer)
                adv_kwargs["reasoning_lengths"] = reasoning_lengths_tmr

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def _concat_dataprotos_non_empty(chunks: list[DataProto]) -> DataProto:
    """Concatenate only non-empty DataProto chunks, aligning schemas across chunks first."""
    non_empty = [chunk for chunk in chunks if len(chunk) > 0]
    if not non_empty:
        return chunks[0][:0]
    if len(non_empty) == 1:
        return non_empty[0]

    # DataProto.concat requires identical tensor and non-tensor key schemas.
    # Hybrid branches can differ in optional keys (e.g., rm_scores/response_mask),
    # so we align by intersecting keys present in every chunk.
    common_batch_keys = None
    common_non_tensor_keys = None
    for chunk in non_empty:
        batch_keys = set(chunk.batch.keys()) if chunk.batch is not None else set()
        non_tensor_keys = set(chunk.non_tensor_batch.keys())
        if common_batch_keys is None:
            common_batch_keys = batch_keys
            common_non_tensor_keys = non_tensor_keys
        else:
            common_batch_keys &= batch_keys
            common_non_tensor_keys &= non_tensor_keys

    aligned = [
        chunk.select(
            batch_keys=sorted(common_batch_keys) if common_batch_keys else [],
            non_tensor_batch_keys=sorted(common_non_tensor_keys) if common_non_tensor_keys else [],
        )
        for chunk in non_empty
    ]
    return DataProto.concat(aligned)


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # Three-mode self-routing forcer (NOTHINK / SHORT / LONG).
        self._three_mode_routing_forcer: Optional[ThreeModeRoutingForcer] = None
        tmr_cfg_init = OmegaConf.select(self.config, "algorithm.three_mode_routing")
        if tmr_cfg_init is not None and tmr_cfg_init.get("enable", False):
            data_chat_kwargs_tmr = OmegaConf.select(self.config, "data.apply_chat_template_kwargs")
            if data_chat_kwargs_tmr is not None:
                data_chat_kwargs_tmr = OmegaConf.to_container(data_chat_kwargs_tmr, resolve=True)
            self._three_mode_routing_forcer = ThreeModeRoutingForcer(
                config=ThreeModeRoutingConfig(**{
                    k: v for k, v in OmegaConf.to_container(tmr_cfg_init, resolve=True).items()
                    if k in ThreeModeRoutingConfig.__dataclass_fields__
                }),
                tokenizer=tokenizer,
                apply_chat_template_kwargs=data_chat_kwargs_tmr,
            )

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _is_three_mode_active(self) -> bool:
        """True when three_mode_routing is enabled (always routes through forcer)."""
        return self._three_mode_routing_forcer is not None

    def _is_three_mode_warmup(self) -> bool:
        """True during phase 1 (heavy forcing: n//4 per mode)."""
        if not self._is_three_mode_active():
            return False
        return self.global_steps <= self._three_mode_routing_forcer.config.warmup_steps

    def _is_three_mode_warmup2(self) -> bool:
        """True during phase 2 (light forcing: 1 per mode).

        Active when warmup2_steps > warmup_steps and we are past phase 1.
        """
        if not self._is_three_mode_active():
            return False
        cfg = self._three_mode_routing_forcer.config
        warmup2 = int(cfg.warmup2_steps)
        return warmup2 > int(cfg.warmup_steps) and self.global_steps <= warmup2

    def _build_three_mode_rollouts(
        self,
        gen_batch,
        generate_fn,
        dp_size=None,
    ):
        """Delegate to ThreeModeRoutingForcer with the correct forced/free split.

        Phase 1 (steps 0..warmup_steps):       n//4 forced per mode, remainder free.
        Phase 2 (warmup_steps..warmup2_steps):  1 forced per mode, remainder free.
        Phase 3 (post-warmup2):                all n rollouts free.

        n_modes = number of enabled modes (3 by default, 2 for e.g. "nothink,long"),
        so the forced total is n_forced * n_modes and the free count fills the rest.
        """
        if self._three_mode_routing_forcer is None:
            raise RuntimeError("ThreeModeRoutingForcer is not initialized")
        cfg = self._three_mode_routing_forcer.config
        n_rollouts = cfg.n_rollouts

        # train_forced_mode (ablation): every rollout, every step, forced into one mode —
        # bypasses the phased warmup/free split entirely. Only this mode is ever forced,
        # regardless of enabled_modes.
        train_forced_mode = str(getattr(cfg, "train_forced_mode", "") or "").lower().strip()
        if train_forced_mode in ("nothink", "short", "long"):
            return self._three_mode_routing_forcer.build_rollouts(
                gen_batch=gen_batch,
                generate_fn=generate_fn,
                n_forced=n_rollouts,
                n_free=0,
                max_prompt_length=int(self.config.data.max_prompt_length),
                dp_size=dp_size,
                forced_modes=[train_forced_mode],
            )

        n_modes = len(enabled_modes_from_config(cfg))
        if self._is_three_mode_warmup():
            n_forced = max(1, n_rollouts // 4)  # phase 1: n=4→1, n=8→2
            n_free = max(0, n_rollouts - n_modes * n_forced)
        elif self._is_three_mode_warmup2():
            n_forced = 1  # phase 2: 1 per enabled mode
            n_free = max(0, n_rollouts - n_modes)
        else:
            n_forced = 0
            n_free = n_rollouts
        return self._three_mode_routing_forcer.build_rollouts(
            gen_batch=gen_batch,
            generate_fn=generate_fn,
            n_forced=n_forced,
            n_free=n_free,
            max_prompt_length=int(self.config.data.max_prompt_length),
            dp_size=dp_size,
        )

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _print_val_samples(self, inputs, outputs, gts, scores, routing_modes=None, n: int = 5):
        """Print N validation samples to stdout so they show up in the SLURM log.

        Uses a step-offset seed so each validation surfaces a different subset
        but the choice is reproducible.  Truncates very long prompts/outputs
        so the log stays readable; the full text is still in the W&B table /
        validation_data_dir dump if those are enabled.
        """
        if not inputs:
            return

        import numpy as np

        n = min(n, len(inputs))
        rng = np.random.RandomState(42 + int(self.global_steps))
        indices = list(range(len(inputs)))
        rng.shuffle(indices)
        indices = indices[:n]

        bar = "=" * 80
        print(f"\n{bar}\n[Validation samples — step {self.global_steps}, showing {n} of {len(inputs)}]\n{bar}")
        for rank, i in enumerate(indices):
            print(f"\n--- sample {rank + 1} / {n} (idx {i}) ---")
            if routing_modes is not None and i < len(routing_modes):
                print(f"routing_mode: {routing_modes[i]}")
            print(f"score: {float(scores[i]):.3f}")
            if i < len(gts) and gts[i] is not None:
                print(f"ground_truth: {gts[i]}")

            inp = str(inputs[i])
            out = str(outputs[i])
            if len(inp) > 1200:
                inp = inp[:600] + "\n...[truncated]...\n" + inp[-600:]
            if len(out) > 2000:
                out = out[:1000] + "\n...[truncated]...\n" + out[-1000:]
            print(f"input:\n{inp}")
            print(f"output:\n{out}")
        print(f"{bar}\n", flush=True)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        correctness_threshold = 0.5

        # Three-mode routing: inject routing question at validation time (no forced token),
        # then detect NOTHINK/SHORT/LONG from response prefix.
        three_mode_routing_val_enabled = (
            self._three_mode_routing_forcer is not None
            and self.config.algorithm.adv_estimator == AdvantageEstimator.THREE_MODE_ROUTING
        )
        if three_mode_routing_val_enabled:
            tmr_val_cfg = self._three_mode_routing_forcer.config
            tmr_val_forced_mode = str(getattr(tmr_val_cfg, "val_forced_mode", "") or "").lower().strip()
            # Prefix match without \b: the next token can attach a word character with
            # no space, causing \b to miss valid routing words.
            tmr_nt_re = re.compile(rf"^\s*{re.escape(tmr_val_cfg.nothink_token)}", flags=re.IGNORECASE)
            tmr_sh_re = re.compile(rf"^\s*{re.escape(tmr_val_cfg.short_token)}", flags=re.IGNORECASE)
            tmr_lg_re = re.compile(rf"^\s*{re.escape(tmr_val_cfg.long_token)}", flags=re.IGNORECASE)
        else:
            tmr_val_forced_mode = ""
            tmr_nt_re = tmr_sh_re = tmr_lg_re = None

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_response_lengths = []
        sample_reasoning_lengths = []
        sample_is_correct = []
        sample_routing_modes = []
        sample_truncated_by_length = []
        sample_levels = []  # MATH difficulty "Level 1".."Level 5" (when present in the data)

        for val_batch_idx, test_data in enumerate(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Capture MATH difficulty level BEFORE _get_gen_batch (which pops non-reward
            # keys out of test_batch). Aligned with sample_gts → aligned with sample_is_correct.
            if "level" in test_batch.non_tensor_batch:
                sample_levels.extend(list(test_batch.non_tensor_batch["level"]))

            test_gen_batch = self._get_gen_batch(test_batch)

            # Three-mode routing: inject routing question.  In forced-mode eval the routing
            # token is also appended so the model generates purely in that mode.
            # prefilled_prompt_mode is re-set after the base_meta_info overwrite below.
            if three_mode_routing_val_enabled and test_gen_batch.non_tensor_batch.get("raw_prompt") is not None:
                if tmr_val_forced_mode in ("nothink", "short", "long"):
                    test_gen_batch = self._three_mode_routing_forcer._ctrl.augment_batch(
                        gen_batch=test_gen_batch,
                        routing_mode=tmr_val_forced_mode,
                        max_prompt_length=int(self.config.data.max_prompt_length),
                    )
                else:
                    test_gen_batch = self._three_mode_routing_forcer.append_routing_question(
                        gen_batch=test_gen_batch,
                        max_prompt_length=int(self.config.data.max_prompt_length),
                    )

            base_meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            test_gen_batch.meta_info = dict(base_meta_info)
            if three_mode_routing_val_enabled and test_gen_batch.non_tensor_batch.get("raw_prompt") is not None:
                # Same: routing question was injected above; re-set flag after meta_info overwrite
                test_gen_batch.meta_info["prefilled_prompt_mode"] = True
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            def _gen(batch):
                padded, pad = pad_dataproto_to_divisor(batch, size_divisor)
                if not self.async_rollout_mode:
                    out_padded = self.actor_rollout_wg.generate_sequences(padded)
                else:
                    out_padded = self.async_rollout_manager.generate_sequences(padded)
                return unpad_dataproto(out_padded, pad_size=pad)

            test_output_gen_batch = _gen(test_gen_batch)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            if three_mode_routing_val_enabled and tmr_nt_re is not None:
                for text in output_texts:
                    if tmr_val_forced_mode in ("nothink", "short", "long"):
                        # Routing token was injected into prompt; response starts after it.
                        # Mode is known — no regex needed.
                        mode = tmr_val_forced_mode
                    elif tmr_nt_re.match(text):
                        mode = "nothink"
                    elif tmr_sh_re.match(text):
                        mode = "short"
                    elif tmr_lg_re.match(text):
                        mode = "long"
                    else:
                        mode = "unknown"
                    sample_routing_modes.append(mode)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Three-mode routing: optionally apply the per-mode token caps before
            # scoring so validation reflects the trained (capped) objective — a
            # diagnostic mode.  Default off: validation measures production
            # behaviour (free generation, no truncation harness), at the cost of a
            # small train/val mismatch for over-cap NOTHINK/SHORT responses.
            # Enable via algorithm.three_mode_routing.apply_caps_in_validation=True.
            # Note: the sample texts logged above remain untruncated (cosmetic only);
            # the reward manager reads lengths from the attention mask we mutate here.
            if (
                three_mode_routing_val_enabled
                and bool(getattr(tmr_val_cfg, "apply_caps_in_validation", False))
                and any(
                    x is not None
                    for x in (
                        tmr_val_cfg.nothink_max_tokens,
                        tmr_val_cfg.short_max_tokens,
                        tmr_val_cfg.long_max_tokens,
                    )
                )
            ):
                if tmr_val_forced_mode in ("nothink", "short", "long"):
                    # Forced-mode eval: the routing token sits in the prompt, so the
                    # response carries no routing word to detect — trust the forced mode
                    # and grant no routing-token cap allowance (zeros).
                    val_known_modes = np.full(len(test_batch), tmr_val_forced_mode, dtype=object)
                    val_known_forced = np.ones(len(test_batch), dtype=bool)
                    val_rt_lengths = np.zeros(len(test_batch), dtype=np.int64)
                else:
                    val_known_modes = None
                    val_known_forced = None
                    val_rt_lengths = None
                apply_three_mode_caps(
                    test_batch,
                    tokenizer=self.tokenizer,
                    config=tmr_val_cfg,
                    known_modes=val_known_modes,
                    known_forced=val_known_forced,
                    routing_token_lengths=val_rt_lengths,
                )

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            _, reasoning_lengths_tensor, closed_think_tensor, valid_response_lengths = compute_reasoning_token_statistics(
                batch=test_batch,
                tokenizer=self.tokenizer,
            )
            response_len_max = int(test_batch.batch["responses"].shape[-1])
            truncated_no_close_mask = (valid_response_lengths >= response_len_max) & (~closed_think_tensor)
            if truncated_no_close_mask.any():
                reasoning_lengths_tensor = reasoning_lengths_tensor.clone()
                reasoning_lengths_tensor[truncated_no_close_mask] = valid_response_lengths[
                    truncated_no_close_mask
                ].to(dtype=reasoning_lengths_tensor.dtype)

            # Aborted-by-length: response filled the entire response tensor.  vLLM stops
            # at max_tokens without an EOS in these cases, so the generation was cut off.
            truncated_by_length_mask = valid_response_lengths >= response_len_max
            sample_truncated_by_length.extend(truncated_by_length_mask.float().cpu().tolist())

            sample_response_lengths.extend(valid_response_lengths.float().cpu().tolist())
            sample_reasoning_lengths.extend(reasoning_lengths_tensor.float().cpu().tolist())

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})

            batch_acc_values = reward_extra_info.get("acc", None)
            if batch_acc_values is None:
                batch_is_correct = [1.0 if score > correctness_threshold else 0.0 for score in scores]
            elif isinstance(batch_acc_values, np.ndarray):
                batch_is_correct = np.asarray(batch_acc_values, dtype=np.float32).reshape(-1).tolist()
            elif torch.is_tensor(batch_acc_values):
                batch_is_correct = batch_acc_values.detach().float().cpu().reshape(-1).tolist()
            elif isinstance(batch_acc_values, list):
                batch_is_correct = [float(v) for v in batch_acc_values]
            else:
                batch_is_correct = [float(batch_acc_values)]

            if len(batch_is_correct) != len(scores):
                batch_is_correct = [1.0 if score > correctness_threshold else 0.0 for score in scores]
            sample_is_correct.extend(batch_is_correct)

            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Print N validation samples to stdout (captured in the SLURM .log via tee).
        self._print_val_samples(
            inputs=sample_inputs,
            outputs=sample_outputs,
            gts=sample_gts,
            scores=sample_scores,
            routing_modes=sample_routing_modes if three_mode_routing_val_enabled else None,
            n=5,
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best", "all_correct"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if len(sample_response_lengths) > 0 and len(sample_reasoning_lengths) > 0 and len(sample_is_correct) > 0:
            n = min(len(sample_response_lengths), len(sample_reasoning_lengths), len(sample_is_correct))
            response_lengths_arr = np.asarray(sample_response_lengths[:n], dtype=np.float32)
            reasoning_lengths_arr = np.asarray(sample_reasoning_lengths[:n], dtype=np.float32)
            correct_mask = np.asarray(sample_is_correct[:n], dtype=np.float32) > 0.5

            def _length_stats(arr: np.ndarray, prefix: str) -> None:
                """Emit mean/std/percentile metrics for a length array into val-lengths/ tab."""
                if len(arr) == 0:
                    for sfx in ("mean", "std", "p25", "p50", "p75", "p90"):
                        metric_dict[f"val-lengths/{prefix}/{sfx}"] = 0.0
                    return
                metric_dict[f"val-lengths/{prefix}/mean"] = float(arr.mean())
                metric_dict[f"val-lengths/{prefix}/std"] = float(arr.std())
                p25, p50, p75, p90 = np.percentile(arr, [25, 50, 75, 90])
                metric_dict[f"val-lengths/{prefix}/p25"] = float(p25)
                metric_dict[f"val-lengths/{prefix}/p50"] = float(p50)
                metric_dict[f"val-lengths/{prefix}/p75"] = float(p75)
                metric_dict[f"val-lengths/{prefix}/p90"] = float(p90)

            _length_stats(response_lengths_arr, "response")
            _length_stats(reasoning_lengths_arr, "reasoning")
            # Keep legacy val-aux means for backwards-compat with existing W&B charts
            metric_dict["val-aux/response_length/mean"] = metric_dict["val-lengths/response/mean"]
            metric_dict["val-aux/reasoning_length/mean"] = metric_dict["val-lengths/reasoning/mean"]

            if np.any(correct_mask):
                _length_stats(response_lengths_arr[correct_mask], "response_correct")
                _length_stats(reasoning_lengths_arr[correct_mask], "reasoning_correct")
            else:
                for sfx in ("mean", "std", "p25", "p50", "p75", "p90"):
                    metric_dict[f"val-lengths/response_correct/{sfx}"] = 0.0
                    metric_dict[f"val-lengths/reasoning_correct/{sfx}"] = 0.0
            metric_dict["val-aux/response_length/mean_correct"] = metric_dict.get("val-lengths/response_correct/mean", 0.0)
            metric_dict["val-aux/reasoning_length/mean_correct"] = metric_dict.get("val-lengths/reasoning_correct/mean", 0.0)

            incorrect_mask = ~correct_mask
            if np.any(incorrect_mask):
                _length_stats(response_lengths_arr[incorrect_mask], "response_incorrect")
                _length_stats(reasoning_lengths_arr[incorrect_mask], "reasoning_incorrect")
            else:
                for sfx in ("mean", "std", "p25", "p50", "p75", "p90"):
                    metric_dict[f"val-lengths/response_incorrect/{sfx}"] = 0.0
                    metric_dict[f"val-lengths/reasoning_incorrect/{sfx}"] = 0.0
            metric_dict["val-aux/response_length/mean_incorrect"] = metric_dict.get("val-lengths/response_incorrect/mean", 0.0)
            metric_dict["val-aux/reasoning_length/mean_incorrect"] = metric_dict.get("val-lengths/reasoning_incorrect/mean", 0.0)

            metric_dict["val-lengths/correct_fraction"] = float(correct_mask.mean())

            # Log length histograms directly to W&B (not via metric_dict to avoid breaking
            # non-wandb loggers that can't serialise Histogram objects).
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    hist_data: dict[str, object] = {
                        "val-lengths/response_hist": _wandb.Histogram(response_lengths_arr),
                        "val-lengths/reasoning_hist": _wandb.Histogram(reasoning_lengths_arr),
                    }
                    if np.any(correct_mask):
                        hist_data["val-lengths/response_correct_hist"] = _wandb.Histogram(
                            response_lengths_arr[correct_mask]
                        )
                        hist_data["val-lengths/reasoning_correct_hist"] = _wandb.Histogram(
                            reasoning_lengths_arr[correct_mask]
                        )
                    if np.any(incorrect_mask):
                        hist_data["val-lengths/response_incorrect_hist"] = _wandb.Histogram(
                            response_lengths_arr[incorrect_mask]
                        )
                        hist_data["val-lengths/reasoning_incorrect_hist"] = _wandb.Histogram(
                            reasoning_lengths_arr[incorrect_mask]
                        )
                    _wandb.log(hist_data, step=self.global_steps, commit=False)
            except Exception:
                pass

            # Aborted-by-length metrics: how often did the response hit the max_tokens cap?
            if len(sample_truncated_by_length) >= n:
                truncated_arr = np.asarray(sample_truncated_by_length[:n], dtype=np.float32) > 0.5
                metric_dict["val-aux/truncated_by_length/fraction"] = float(truncated_arr.mean())
                if np.any(correct_mask):
                    metric_dict["val-aux/truncated_by_length/correct_fraction"] = float(
                        truncated_arr[correct_mask].mean()
                    )
        if three_mode_routing_val_enabled and len(sample_routing_modes) > 0 and len(sample_is_correct) > 0:
            n = min(len(sample_routing_modes), len(sample_is_correct), len(sample_response_lengths), len(sample_uids))
            tmr_val_metrics = compute_three_mode_routing_metrics(
                routing_mode=np.asarray(sample_routing_modes[:n], dtype=object),
                is_correct=torch.as_tensor(sample_is_correct[:n], dtype=torch.float32),
                response_lengths=torch.as_tensor(sample_response_lengths[:n], dtype=torch.float32),
                index=np.asarray(sample_uids[:n], dtype=object),
                advantages=None,  # not available at validation time
            )
            for key, val in tmr_val_metrics.items():
                metric_dict[key.replace("three_mode_routing/", "val-aux/three_mode_routing/")] = val

        # ── Per-difficulty validation metrics (val-difficulties/) ────────────────
        # MATH `level` ("Level 1".."Level 5"): accuracy and (when routing is active)
        # the routing-mode split per difficulty bucket. Only emitted when the data
        # carries a level field; for non-routing runs the mode split is skipped.
        if len(sample_levels) > 0 and len(sample_is_correct) > 0:
            n = min(len(sample_levels), len(sample_is_correct))
            lv_correct = np.asarray(sample_is_correct[:n], dtype=np.float32)

            def _lvl(x):
                digits = "".join(ch for ch in str(x) if ch.isdigit())
                return digits if digits else "unknown"

            lv_keys = np.array([_lvl(x) for x in sample_levels[:n]], dtype=object)
            have_modes = len(sample_routing_modes) >= n
            if have_modes:
                lv_modes = np.array([str(m).lower() for m in sample_routing_modes[:n]], dtype=object)
            for lv in sorted(set(lv_keys)):
                mask = lv_keys == lv
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                metric_dict[f"val-difficulties/level{lv}/count"] = float(cnt)
                metric_dict[f"val-difficulties/level{lv}/acc"] = float(lv_correct[mask].mean())
                if have_modes:
                    for mode in ("nothink", "short", "long", "unknown"):
                        metric_dict[f"val-difficulties/level{lv}/{mode}_frac"] = float(
                            (lv_modes[mask] == mode).mean()
                        )

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        rollout_repeat_times = int(batch.meta_info.get("rollout_repeat_times", self.config.actor_rollout_ref.rollout.n))
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * rollout_repeat_times
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        rollout_repeat_times = int(batch.meta_info.get("rollout_repeat_times", self.config.actor_rollout_ref.rollout.n))
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * rollout_repeat_times
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        run_succeeded = False

        self.global_steps = 0
        progress_bar = None

        try:
            # load checkpoint before doing anything
            self._load_checkpoint()

            current_epoch = self.global_steps // len(self.train_dataloader)

            # perform validation before training
            # currently, we only support validation using the reward_function.
            if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    run_succeeded = True
                    return

            if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
                rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
                rollout_skip.wrap_generate_sequences()

            # add tqdm
            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

            # we start from step 1
            self.global_steps += 1
            last_val_metrics = None
            self.max_steps_duration = 0

            prev_step_profile = False
            curr_step_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            next_step_profile = False

            for epoch in range(current_epoch, self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                    metrics = {}
                    timing_raw = {}

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    # add uid to batch
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(batch)

                    # pass global_steps to trace
                    gen_batch.meta_info["global_steps"] = self.global_steps

                    current_adv_estimator = self.config.algorithm.adv_estimator
                    current_rollout_repeat_times = int(self.config.actor_rollout_ref.rollout.n)
                    three_mode_active = self._is_three_mode_active()
                    metrics["three_mode_routing/active"] = float(three_mode_active)

                    if not three_mode_active:
                        gen_batch_output = gen_batch.repeat(
                            repeat_times=current_rollout_repeat_times, interleave=True
                        )
                    else:
                        # The three-mode forcer builds its own forced/free rollout mix.
                        gen_batch_output = gen_batch

                    is_last_step = self.global_steps >= self.total_training_steps
                    three_mode_source_indices = None
                    with marked_timer("step", timing_raw):
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                generate_fn = self.actor_rollout_wg.generate_sequences
                            else:
                                generate_fn = self.async_rollout_manager.generate_sequences

                            if three_mode_active:
                                three_mode_dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
                                (
                                    gen_batch_output,
                                    three_mode_source_indices,
                                    three_mode_rollout_metrics,
                                ) = self._build_three_mode_rollouts(
                                    gen_batch=gen_batch,
                                    generate_fn=generate_fn,
                                    dp_size=three_mode_dp_size,
                                )
                                metrics.update(three_mode_rollout_metrics)
                            else:
                                gen_batch_output = generate_fn(gen_batch_output)

                            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        if current_adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                batch = batch.union(gen_baseline_output)
                                # compute reward model score on batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    if not self.use_reward_loop:
                                        rm_scores = self.rm_wg.compute_rm_score(batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                    batch = batch.union(rm_scores)

                                # Compute or extract reward for REMAX baseline
                                reward_baseline_tensor = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, sum_reward=True
                                )

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                batch.pop(batch_keys=list(keys_to_pop))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        if three_mode_active:
                            if three_mode_source_indices is None:
                                raise RuntimeError(
                                    "Three-mode routing source indices were not returned."
                                )
                            current_rollout_repeat_times = 1
                            batch.meta_info["rollout_repeat_times"] = 1
                            batch = batch.select_idxs(three_mode_source_indices)
                        else:
                            batch.meta_info["rollout_repeat_times"] = current_rollout_repeat_times
                            batch = batch.repeat(repeat_times=current_rollout_repeat_times, interleave=True)
                        batch = batch.union(gen_batch_output)

                        # Three-mode routing per-mode response cap.
                        if (
                            three_mode_active
                            and self._three_mode_routing_forcer is not None
                        ):
                            tmr_cap_cfg = self._three_mode_routing_forcer.config
                            if any(
                                x is not None
                                for x in (
                                    tmr_cap_cfg.nothink_max_tokens,
                                    tmr_cap_cfg.short_max_tokens,
                                    tmr_cap_cfg.long_max_tokens,
                                )
                            ):
                                # Pass the stored labels so forced rollouts keep their
                                # known-certain mode instead of being re-detected (a
                                # re-detection miss used to charge them unknown_penalty).
                                cap_modes, was_truncated_by_cap = apply_three_mode_caps(
                                    batch,
                                    tokenizer=self.tokenizer,
                                    config=tmr_cap_cfg,
                                    known_modes=batch.non_tensor_batch.get("routing_mode"),
                                    known_forced=batch.non_tensor_batch.get("routing_forced"),
                                    routing_token_lengths=batch.non_tensor_batch.get("routing_token_length"),
                                )
                                batch.non_tensor_batch["routing_mode"] = cap_modes
                                batch.non_tensor_batch["was_truncated_by_cap"] = was_truncated_by_cap
                                batch.batch["response_mask"] = compute_response_mask(batch)
                                # CRITICAL: the async reward loop scored the FULL response
                                # during generation and cached it in rm_scores — BEFORE this
                                # cap was applied. Using that stale score means an over-cap
                                # response (answer beyond the cap) is still marked correct, so
                                # the per-mode cap never creates the intended capability gap.
                                # Drop rm_scores so the reward is recomputed below on the
                                # CAPPED response (answer must fall within the cap to count).
                                if "rm_scores" in batch.batch.keys():
                                    batch.batch.pop("rm_scores")

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # Compute or extract reward for training
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, return_dict=False
                                )

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch.meta_info["global_steps"] = self.global_steps
                            batch = compute_advantage(
                                batch,
                                adv_estimator=current_adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=current_rollout_repeat_times,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                                tokenizer=self.tokenizer,
                            )

                            # ── Three-mode routing metrics ─────────────────────────────────
                            if three_mode_active and "routing_mode" in batch.non_tensor_batch:
                                seq_scores_tmr = batch.batch["token_level_rewards"].sum(dim=-1)
                                is_correct_tmr = (seq_scores_tmr > 0.5).float()
                                tmr_adv = batch.batch["advantages"]
                                tmr_forced = batch.non_tensor_batch.get("routing_forced", None)
                                tmr_gate_active = self._is_three_mode_warmup() and (
                                    self._three_mode_routing_forcer.config.solved_gate_warmup_steps > 0
                                    and self.global_steps <= self._three_mode_routing_forcer.config.solved_gate_warmup_steps
                                )
                                tmr_detail_metrics = compute_three_mode_routing_metrics(
                                    routing_mode=batch.non_tensor_batch["routing_mode"],
                                    is_correct=is_correct_tmr,
                                    response_lengths=batch.batch["response_mask"].sum(dim=-1),
                                    index=batch.non_tensor_batch["uid"],
                                    advantages=tmr_adv,
                                    routing_forced=tmr_forced,
                                    gate_active=tmr_gate_active,
                                    was_truncated_by_cap=batch.non_tensor_batch.get("was_truncated_by_cap"),
                                )
                                metrics.update(tmr_detail_metrics)
                                _tmr_cfg_m = self._three_mode_routing_forcer.config
                                metrics["three_mode_routing/balance_coef_effective"] = effective_balance_coef(
                                    balance_coef=_tmr_cfg_m.balance_coef,
                                    anneal_start_step=_tmr_cfg_m.balance_anneal_start_step,
                                    anneal_end_step=_tmr_cfg_m.balance_anneal_end_step,
                                    anneal_final_coef=_tmr_cfg_m.balance_anneal_final_coef,
                                    global_step=self.global_steps,
                                )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, tokenizer=self.tokenizer))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                    # Difficulty-specific accuracy (e.g., levels 3 and 4).
                    difficulty_metrics = compute_difficulty_metrics(batch=batch)
                    metrics.update(difficulty_metrics)

                    # Completion statistics: truncated vs finished and correctness.
                    generation_budget = int(batch.batch["responses"].shape[-1])
                    completion_metrics = compute_completion_metrics(batch=batch, generation_budget=generation_budget)
                    metrics.update(completion_metrics)

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        run_succeeded = True
                        return

                    # this is experimental and may be changed/removed in the future
                    # in favor of a general-purpose data buffer pool
                    if hasattr(self.train_dataset, "on_batch_end"):
                        # The dataset may be changed after each training batch
                        self.train_dataset.on_batch_end(batch=batch)
            run_succeeded = True
        finally:
            if progress_bar is not None:
                progress_bar.close()
            logger.finish(exit_code=0 if run_succeeded else 1)
