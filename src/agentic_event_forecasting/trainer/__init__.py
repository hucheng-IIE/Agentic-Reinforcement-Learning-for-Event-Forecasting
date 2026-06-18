"""Training utilities for GRPO-style optimization."""

from agentic_event_forecasting.trainer.advantage import AdvantageRecord, compute_group_advantages
from agentic_event_forecasting.trainer.grpo import GRPOConfig, GRPOStepStats, GRPOTrainer

__all__ = [
    "AdvantageRecord",
    "GRPOConfig",
    "GRPOStepStats",
    "GRPOTrainer",
    "compute_group_advantages",
]
