"""GRPO training interface.

This module implements the framework-level mechanics that are independent of a
specific neural backend: rollout collection, trajectory reward scoring,
group-wise advantage estimation, and clipped objective statistics. A future
Transformer/PyTorch backend can provide current/reference token log-probability
functions and an optimizer while reusing the same records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterable, List, Optional, Sequence

from agentic_event_forecasting.reward import RewardScorer
from agentic_event_forecasting.rollout import RolloutGenerator
from agentic_event_forecasting.schema import ForecastQuery, Trajectory
from agentic_event_forecasting.trainer.advantage import AdvantageRecord, compute_group_advantages

LogprobFn = Callable[[Trajectory], Sequence[float]]


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 6
    clip_epsilon: float = 0.2
    beta_kl: float = 0.01
    advantage_epsilon: float = 1e-6


@dataclass
class GRPOStepStats:
    num_queries: int
    num_trajectories: int
    mean_reward: float
    mean_outcome_reward: float
    mean_process_reward: float
    mean_advantage: float
    clip_loss: float
    kl: float
    total_loss: float

    def to_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "num_trajectories": self.num_trajectories,
            "mean_reward": self.mean_reward,
            "mean_outcome_reward": self.mean_outcome_reward,
            "mean_process_reward": self.mean_process_reward,
            "mean_advantage": self.mean_advantage,
            "clip_loss": self.clip_loss,
            "kl": self.kl,
            "total_loss": self.total_loss,
        }


class GRPOTrainer:
    """Collect rollouts and compute GRPO training statistics."""

    def __init__(
        self,
        rollout_generator: RolloutGenerator,
        reward_scorer: RewardScorer,
        config: Optional[GRPOConfig] = None,
    ) -> None:
        self.rollout_generator = rollout_generator
        self.reward_scorer = reward_scorer
        self.config = config or GRPOConfig()

    def collect_records(self, queries: Iterable[ForecastQuery]) -> List[AdvantageRecord]:
        records: List[AdvantageRecord] = []
        for query in queries:
            trajectories = self.rollout_generator.generate_for_query(
                query,
                group_size=self.config.group_size,
            )
            rewards = self.reward_scorer.score_group(trajectories)
            records.extend(
                compute_group_advantages(
                    trajectories,
                    rewards,
                    eps=self.config.advantage_epsilon,
                )
            )
        return records

    def train_step(
        self,
        queries: Iterable[ForecastQuery],
        current_logprobs: Optional[LogprobFn] = None,
        reference_logprobs: Optional[LogprobFn] = None,
    ) -> tuple[GRPOStepStats, List[AdvantageRecord]]:
        records = self.collect_records(queries)
        clip_losses: List[float] = []
        kls: List[float] = []
        for record in records:
            trajectory = record.trajectory
            old = trajectory.token_logprobs
            current = list(current_logprobs(trajectory)) if current_logprobs else old
            reference = list(reference_logprobs(trajectory)) if reference_logprobs else old
            clip_losses.append(
                clipped_surrogate_loss(
                    current_logprobs=current,
                    old_logprobs=old,
                    advantage=record.advantage,
                    clip_epsilon=self.config.clip_epsilon,
                )
            )
            kls.append(approximate_kl(current, reference))

        rewards = [record.reward.total for record in records]
        outcomes = [record.reward.outcome for record in records]
        processes = [record.reward.details.get("process_reward", 0.0) for record in records]
        advantages = [record.advantage for record in records]
        clip_loss = mean(clip_losses) if clip_losses else 0.0
        kl = mean(kls) if kls else 0.0
        stats = GRPOStepStats(
            num_queries=len({record.trajectory.query.query_id for record in records}),
            num_trajectories=len(records),
            mean_reward=mean(rewards) if rewards else 0.0,
            mean_outcome_reward=mean(outcomes) if outcomes else 0.0,
            mean_process_reward=mean(processes) if processes else 0.0,
            mean_advantage=mean(advantages) if advantages else 0.0,
            clip_loss=clip_loss,
            kl=kl,
            total_loss=clip_loss + self.config.beta_kl * kl,
        )
        return stats, records


def clipped_surrogate_loss(
    current_logprobs: Sequence[float],
    old_logprobs: Sequence[float],
    advantage: float,
    clip_epsilon: float,
) -> float:
    """Compute token-averaged GRPO/PPO-style clipped surrogate loss."""
    pairs = list(zip(current_logprobs, old_logprobs))
    if not pairs:
        return 0.0
    objectives = []
    for current, old in pairs:
        ratio = math.exp(max(-20.0, min(20.0, current - old)))
        clipped_ratio = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
        objectives.append(min(ratio * advantage, clipped_ratio * advantage))
    return -mean(objectives)


def approximate_kl(current_logprobs: Sequence[float], reference_logprobs: Sequence[float]) -> float:
    pairs = list(zip(current_logprobs, reference_logprobs))
    if not pairs:
        return 0.0
    return mean(abs(current - reference) for current, reference in pairs)
