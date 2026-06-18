"""Outcome and process rewards for agentic event forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from agentic_event_forecasting.schema import RewardBreakdown, Trajectory, TrajectoryStep


@dataclass(frozen=True)
class RewardConfig:
    lambda_process: float = 0.5
    format_weight: float = 0.5
    tool_weight: float = 0.5
    retrieval_weight: float = 0.8
    evidence_weight: float = 1.0
    cost_weight: float = 0.15
    invalid_call_weight: float = 0.4


class RewardScorer:
    """Compute trajectory-level rewards."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def score(self, trajectory: Trajectory) -> RewardBreakdown:
        outcome = self._outcome_reward(trajectory)
        format_reward = self._format_reward(trajectory)
        tool_reward = self._tool_reward(trajectory.steps)
        retrieval_reward = self._retrieval_reward(trajectory.steps)
        evidence_reward = self._evidence_reward(trajectory)
        cost_penalty = self._cost_penalty(trajectory)

        cfg = self.config
        process = (
            cfg.format_weight * format_reward
            + cfg.tool_weight * tool_reward
            + cfg.retrieval_weight * retrieval_reward
            + cfg.evidence_weight * evidence_reward
            - cfg.cost_weight * cost_penalty
        )
        total = outcome + cfg.lambda_process * process
        return RewardBreakdown(
            outcome=outcome,
            format_reward=format_reward,
            tool_reward=tool_reward,
            retrieval_reward=retrieval_reward,
            evidence_reward=evidence_reward,
            cost_penalty=cost_penalty,
            total=total,
            details={"process_reward": process},
        )

    def score_group(self, trajectories: Iterable[Trajectory]) -> List[RewardBreakdown]:
        return [self.score(trajectory) for trajectory in trajectories]

    def _outcome_reward(self, trajectory: Trajectory) -> float:
        truth = trajectory.query.ground_truth
        if truth is None or trajectory.final_prediction is None:
            return 0.0
        return 1.0 if trajectory.final_prediction.label == truth else 0.0

    def _format_reward(self, trajectory: Trajectory) -> float:
        if trajectory.final_prediction is None:
            return 0.0
        final_steps = [step for step in trajectory.steps if step.action.name == "final_answer"]
        if not final_steps:
            return 0.0
        return 1.0 if final_steps[-1].observation.ok and trajectory.final_prediction.is_valid() else 0.0

    def _tool_reward(self, steps: List[TrajectoryStep]) -> float:
        tool_steps = [step for step in steps if step.action.name != "final_answer"]
        if not tool_steps:
            return 0.0
        return sum(1.0 for step in tool_steps if step.observation.ok) / len(tool_steps)

    def _retrieval_reward(self, steps: List[TrajectoryStep]) -> float:
        retrieval_steps = [
            step
            for step in steps
            if step.action.name in {"event_retrieval", "news_retrieval", "relation_statistics", "graph_analysis"}
        ]
        if not retrieval_steps:
            return 0.0
        informative = 0
        for step in retrieval_steps:
            data = step.observation.data
            if data.get("count", 0) > 0:
                informative += 1
            elif data.get("relation_counts") or data.get("direct_relation_counts"):
                informative += 1
            elif data.get("common_neighbors"):
                informative += 1
        return informative / len(retrieval_steps)

    def _evidence_reward(self, trajectory: Trajectory) -> float:
        if trajectory.final_prediction is None or not trajectory.final_prediction.label:
            return 0.0
        label = trajectory.final_prediction.label
        supported = False
        for step in trajectory.steps:
            data = step.observation.data
            if label in data.get("relation_counts", {}):
                supported = True
            if label in data.get("direct_relation_counts", {}):
                supported = True
            for event in data.get("events", []):
                if event.get("relation") == label:
                    supported = True
            for article in data.get("articles", []):
                if label in article.get("relations", []):
                    supported = True
        if supported:
            return 1.0
        return 0.2 if trajectory.final_prediction.evidence else 0.0

    def _cost_penalty(self, trajectory: Trajectory) -> float:
        invalid_penalty = self.config.invalid_call_weight * trajectory.invalid_call_count
        return trajectory.total_cost + invalid_penalty
