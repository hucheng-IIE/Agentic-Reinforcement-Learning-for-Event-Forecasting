"""Agent-environment interaction for event forecasting."""

from __future__ import annotations

from typing import Iterable, List, Optional

from agentic_event_forecasting.schema import (
    Action,
    Event,
    FinalPrediction,
    ForecastQuery,
    NewsArticle,
    Observation,
    Trajectory,
)
from agentic_event_forecasting.tools import (
    EventRetrievalTool,
    GraphAnalysisTool,
    NewsRetrievalTool,
    RelationStatisticsTool,
    ToolContext,
    ToolRegistry,
)


class ForecastingEnvironment:
    """Executes ReAct-style actions against forecasting tools."""

    def __init__(
        self,
        events: Iterable[Event],
        news: Iterable[NewsArticle] = (),
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 6,
        repeated_call_penalty: float = 0.5,
    ) -> None:
        self.events = list(events)
        self.news = list(news)
        self.registry = registry or default_tool_registry()
        self.max_steps = max_steps
        self.repeated_call_penalty = repeated_call_penalty
        self.query: Optional[ForecastQuery] = None
        self.trajectory: Optional[Trajectory] = None
        self._seen_actions: set[str] = set()

    def reset(self, query: ForecastQuery) -> Trajectory:
        self.query = query
        self.trajectory = Trajectory(query=query)
        self._seen_actions = set()
        return self.trajectory

    @property
    def is_done(self) -> bool:
        return bool(self.trajectory and self.trajectory.terminated)

    def step(self, action: Action) -> Observation:
        if self.query is None or self.trajectory is None:
            raise RuntimeError("environment must be reset before step")
        if self.trajectory.terminated:
            return Observation(
                tool_name="environment",
                ok=False,
                message="trajectory already terminated",
                cost=0.0,
            )

        if action.name == "final_answer":
            observation = self._handle_final_answer(action)
            self.trajectory.add_step(action, observation)
            return observation

        if len(self.trajectory.steps) >= self.max_steps:
            self.trajectory.terminated = True
            self.trajectory.termination_reason = "max_steps"
            observation = Observation(
                tool_name="environment",
                ok=False,
                message="maximum interaction step reached",
                cost=0.0,
            )
            self.trajectory.add_step(action, observation)
            return observation

        fingerprint = action.fingerprint()
        repeated = fingerprint in self._seen_actions
        self._seen_actions.add(fingerprint)

        try:
            tool = self.registry.get(action.name)
        except KeyError as exc:
            observation = Observation(
                tool_name=action.name,
                ok=False,
                message=str(exc),
                cost=1.0,
            )
            self.trajectory.add_step(action, observation)
            return observation

        context = ToolContext(
            query=self.query,
            events=self.events,
            news=self.news,
            step_index=len(self.trajectory.steps),
        )
        result = tool.run(context, **action.arguments)
        cost = result.cost + (self.repeated_call_penalty if repeated else 0.0)
        data = dict(result.data)
        if repeated:
            data["repeated_call"] = True
        observation = Observation(
            tool_name=action.name,
            ok=result.ok and not repeated,
            data=data,
            message="repeated action call" if repeated else result.message,
            cost=cost,
        )
        self.trajectory.add_step(action, observation)
        return observation

    def _handle_final_answer(self, action: Action) -> Observation:
        assert self.query is not None
        assert self.trajectory is not None
        label = str(action.arguments.get("label", "")).strip()
        confidence = float(action.arguments.get("confidence", 0.0) or 0.0)
        evidence = action.arguments.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        rationale = str(action.arguments.get("rationale", ""))
        prediction = FinalPrediction(
            label=label,
            confidence=max(0.0, min(confidence, 1.0)),
            evidence=[str(item) for item in evidence],
            rationale=rationale,
        )
        self.trajectory.final_prediction = prediction
        self.trajectory.terminated = True
        self.trajectory.termination_reason = "final_answer"

        valid_candidate = (
            not self.query.candidate_relations
            or prediction.label in self.query.candidate_relations
        )
        return Observation(
            tool_name="final_answer",
            ok=prediction.is_valid() and valid_candidate,
            data=prediction.to_dict(),
            message="final prediction accepted" if valid_candidate else "label not in candidates",
            cost=0.2,
        )

    def run(self, query: ForecastQuery, policy: "PolicyProtocol") -> Trajectory:
        trajectory = self.reset(query)
        while not trajectory.terminated:
            action = policy.act(query=query, steps=trajectory.steps)
            self.step(action)
        return trajectory


class PolicyProtocol:
    def act(self, query: ForecastQuery, steps: List[object]) -> Action:
        raise NotImplementedError


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            EventRetrievalTool(),
            NewsRetrievalTool(),
            RelationStatisticsTool(),
            GraphAnalysisTool(),
        ]
    )


def build_default_environment(
    events: Iterable[Event],
    news: Iterable[NewsArticle] = (),
    max_steps: int = 6,
) -> ForecastingEnvironment:
    return ForecastingEnvironment(
        events=events,
        news=news,
        registry=default_tool_registry(),
        max_steps=max_steps,
    )
