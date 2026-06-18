"""Policy interfaces and a deterministic ReAct baseline.

The baseline policy is intentionally lightweight. It provides a runnable
end-to-end implementation before replacing the policy with an LLM backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from agentic_event_forecasting.schema import Action, ForecastQuery, TrajectoryStep


class BasePolicy(ABC):
    """Abstract policy used by rollout, inference, and training modules."""

    name = "base_policy"

    @abstractmethod
    def act(self, query: ForecastQuery, steps: List[TrajectoryStep]) -> Action:
        raise NotImplementedError

    def update_from_preferences(self, records: Sequence[dict]) -> None:
        """Optional hook for trainable policies."""
        return None


@dataclass
class HeuristicReActPolicy(BasePolicy):
    """A rule-based ReAct policy used to exercise the full framework.

    Different strategies produce different tool-use orders and retrieval scopes,
    which makes group-wise rollout and reward normalization meaningful even
    before a neural policy is plugged in.
    """

    strategy: str = "stats_first"
    window_days: int = 365
    name: str = "heuristic_react_policy"

    def act(self, query: ForecastQuery, steps: List[TrajectoryStep]) -> Action:
        used = [step.action.name for step in steps]
        plan = self._plan()
        for tool_name in plan:
            if tool_name not in used:
                return self._build_tool_action(tool_name, query)
        return self._build_final_answer(query, steps)

    def _plan(self) -> List[str]:
        plans = {
            "stats_first": [
                "relation_statistics",
                "event_retrieval",
                "news_retrieval",
                "graph_analysis",
            ],
            "events_first": [
                "event_retrieval",
                "relation_statistics",
                "graph_analysis",
                "news_retrieval",
            ],
            "news_first": [
                "news_retrieval",
                "event_retrieval",
                "relation_statistics",
                "graph_analysis",
            ],
            "graph_first": [
                "graph_analysis",
                "relation_statistics",
                "event_retrieval",
                "news_retrieval",
            ],
            "broad_context": [
                "event_retrieval",
                "news_retrieval",
                "relation_statistics",
                "graph_analysis",
            ],
            "minimal_stats": [
                "relation_statistics",
            ],
            "event_only": [
                "event_retrieval",
            ],
        }
        return plans.get(self.strategy, plans["stats_first"])

    def _build_tool_action(self, tool_name: str, query: ForecastQuery) -> Action:
        pair_only = self.strategy != "broad_context"
        if tool_name == "event_retrieval":
            return Action(
                name=tool_name,
                arguments={
                    "subject": query.subject,
                    "object": query.object,
                    "pair_only": pair_only,
                    "window_days": 730 if self.strategy == "broad_context" else self.window_days,
                    "limit": 12,
                },
            )
        if tool_name == "news_retrieval":
            return Action(
                name=tool_name,
                arguments={
                    "keywords": [query.subject, query.object, *query.candidate_relations],
                    "limit": 6,
                },
            )
        if tool_name == "relation_statistics":
            return Action(
                name=tool_name,
                arguments={"pair_only": pair_only, "limit": 10},
            )
        if tool_name == "graph_analysis":
            return Action(
                name=tool_name,
                arguments={"max_neighbors": 20},
            )
        return Action(name=tool_name, arguments={})

    def _build_final_answer(self, query: ForecastQuery, steps: List[TrajectoryStep]) -> Action:
        scores, evidence = collect_relation_scores(query, steps)
        if not scores and query.candidate_relations:
            scores[query.candidate_relations[0]] = 1.0
        if not scores:
            label = ""
            confidence = 0.0
        else:
            label, best_score = scores.most_common(1)[0]
            total_score = sum(max(value, 0.0) for value in scores.values())
            confidence = best_score / total_score if total_score else 0.0
        rationale = (
            "Prediction is selected from retrieved historical events, relation "
            "statistics, news evidence, and local graph patterns."
        )
        return Action(
            name="final_answer",
            arguments={
                "label": label,
                "confidence": round(float(confidence), 4),
                "evidence": evidence[:6],
                "rationale": rationale,
            },
        )


def collect_relation_scores(
    query: ForecastQuery,
    steps: Iterable[TrajectoryStep],
) -> tuple[Counter[str], List[str]]:
    """Aggregate label scores from all observations in a trajectory."""
    candidate_set = set(query.candidate_relations)
    scores: Counter[str] = Counter()
    evidence: List[str] = []

    def add_relation(label: str, value: float, source: str) -> None:
        if not label:
            return
        if candidate_set and label not in candidate_set:
            return
        scores[label] += value
        if source and source not in evidence:
            evidence.append(source)

    for step in steps:
        observation = step.observation
        if not observation.ok and step.action.name != "final_answer":
            continue
        data = observation.data
        for relation, count in data.get("relation_counts", {}).items():
            add_relation(relation, float(count) * 1.5, f"relation_stats:{relation}:{count}")
        for relation, count in data.get("direct_relation_counts", {}).items():
            add_relation(relation, float(count) * 1.2, f"graph_direct:{relation}:{count}")
        for event in data.get("events", []):
            relation = str(event.get("relation", ""))
            add_relation(relation, 1.0, event.get("source_id") or event.get("timestamp", "event"))
        for article in data.get("articles", []):
            article_score = float(article.get("score", 1.0))
            for relation in article.get("relations", []):
                add_relation(str(relation), 0.5 + 0.1 * article_score, article.get("source_id", "news"))
    return scores, evidence


def make_policy_family(group_size: int) -> List[HeuristicReActPolicy]:
    """Create diverse policies for group rollout."""
    strategies = [
        "stats_first",
        "events_first",
        "minimal_stats",
        "news_first",
        "graph_first",
        "broad_context",
        "event_only",
    ]
    policies: List[HeuristicReActPolicy] = []
    for index in range(group_size):
        strategy = strategies[index % len(strategies)]
        window = 180 + 90 * (index % 4)
        policies.append(HeuristicReActPolicy(strategy=strategy, window_days=window))
    return policies
