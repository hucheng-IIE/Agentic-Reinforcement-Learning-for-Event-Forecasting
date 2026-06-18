"""Inference pipeline for evidence-grounded event forecasting."""

from __future__ import annotations

from typing import Callable, Optional

from agentic_event_forecasting.agent import BasePolicy, HeuristicReActPolicy
from agentic_event_forecasting.env import ForecastingEnvironment
from agentic_event_forecasting.reward import RewardScorer
from agentic_event_forecasting.rollout import RolloutGenerator
from agentic_event_forecasting.schema import ForecastQuery, Trajectory


class EventForecaster:
    """Run single-trajectory or best-of-N inference."""

    def __init__(
        self,
        env_factory: Callable[[], ForecastingEnvironment],
        policy: Optional[BasePolicy] = None,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy or HeuristicReActPolicy()

    def predict(self, query: ForecastQuery) -> Trajectory:
        env = self.env_factory()
        return env.run(query, self.policy)

    def best_of_n(self, query: ForecastQuery, n: int = 6) -> Trajectory:
        generator = RolloutGenerator(self.env_factory)
        trajectories = generator.generate_for_query(query, group_size=n)
        scorer = RewardScorer()
        scored = [(scorer.score(trajectory).total, trajectory) for trajectory in trajectories]
        return max(scored, key=lambda item: item[0])[1]
