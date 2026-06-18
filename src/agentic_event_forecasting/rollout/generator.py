"""Generate multi-step agent trajectories."""

from __future__ import annotations

from typing import Callable, List, Sequence

from agentic_event_forecasting.agent import BasePolicy, make_policy_family
from agentic_event_forecasting.env import ForecastingEnvironment
from agentic_event_forecasting.schema import ForecastQuery, Trajectory


EnvironmentFactory = Callable[[], ForecastingEnvironment]
PolicyFactory = Callable[[int], BasePolicy]


class RolloutGenerator:
    """Sample complete trajectories for each forecasting query."""

    def __init__(
        self,
        env_factory: EnvironmentFactory,
        policy_factory: PolicyFactory | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.policy_factory = policy_factory

    def generate_for_query(self, query: ForecastQuery, group_size: int = 6) -> List[Trajectory]:
        policies = (
            [self.policy_factory(index) for index in range(group_size)]
            if self.policy_factory is not None
            else make_policy_family(group_size)
        )
        trajectories: List[Trajectory] = []
        for index, policy in enumerate(policies):
            env = self.env_factory()
            trajectory = env.run(query, policy)
            trajectory.metadata["rollout_index"] = index
            trajectory.metadata["policy_name"] = getattr(policy, "name", policy.__class__.__name__)
            trajectory.metadata["policy_strategy"] = getattr(policy, "strategy", "")
            trajectories.append(trajectory)
        return trajectories

    def generate(self, queries: Sequence[ForecastQuery], group_size: int = 6) -> List[List[Trajectory]]:
        return [self.generate_for_query(query, group_size=group_size) for query in queries]
