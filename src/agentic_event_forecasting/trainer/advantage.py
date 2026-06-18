"""Group-wise advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List, Sequence

from agentic_event_forecasting.schema import RewardBreakdown, Trajectory


@dataclass
class AdvantageRecord:
    trajectory: Trajectory
    reward: RewardBreakdown
    advantage: float
    group_mean: float
    group_std: float

    def to_dict(self) -> dict:
        return {
            "query_id": self.trajectory.query.query_id,
            "prediction": None
            if self.trajectory.final_prediction is None
            else self.trajectory.final_prediction.label,
            "reward": self.reward.to_dict(),
            "advantage": self.advantage,
            "group_mean": self.group_mean,
            "group_std": self.group_std,
            "metadata": self.trajectory.metadata,
        }


def compute_group_advantages(
    trajectories: Sequence[Trajectory],
    rewards: Sequence[RewardBreakdown],
    eps: float = 1e-6,
) -> List[AdvantageRecord]:
    """Normalize rewards inside one query-specific trajectory group."""
    if len(trajectories) != len(rewards):
        raise ValueError("trajectory and reward counts must match")
    values = [reward.total for reward in rewards]
    if not values:
        return []
    group_mean = mean(values)
    group_std = pstdev(values)
    denominator = group_std + eps
    return [
        AdvantageRecord(
            trajectory=trajectory,
            reward=reward,
            advantage=(reward.total - group_mean) / denominator,
            group_mean=group_mean,
            group_std=group_std,
        )
        for trajectory, reward in zip(trajectories, rewards)
    ]
