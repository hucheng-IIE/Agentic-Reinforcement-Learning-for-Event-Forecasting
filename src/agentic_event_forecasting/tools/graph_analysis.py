"""Simple temporal graph analysis around the query entities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Set

from agentic_event_forecasting.tools.base import ToolContext, ToolResult


class GraphAnalysisTool:
    name = "graph_analysis"
    description = "Analyze local temporal graph neighborhoods before the target timestamp."
    cost = 1.2

    def run(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        max_neighbors = int(kwargs.get("max_neighbors", 20))
        subject_neighbors: Set[str] = set()
        object_neighbors: Set[str] = set()
        direct_relation_counts: Counter[str] = Counter()
        entity_relation_counts: Dict[str, Counter[str]] = defaultdict(Counter)

        for event in context.events:
            if event.timestamp >= context.query.timestamp:
                continue
            if event.subject == context.query.subject:
                subject_neighbors.add(event.object)
                entity_relation_counts[context.query.subject][event.relation] += 1
            if event.object == context.query.subject:
                subject_neighbors.add(event.subject)
                entity_relation_counts[context.query.subject][event.relation] += 1
            if event.subject == context.query.object:
                object_neighbors.add(event.object)
                entity_relation_counts[context.query.object][event.relation] += 1
            if event.object == context.query.object:
                object_neighbors.add(event.subject)
                entity_relation_counts[context.query.object][event.relation] += 1
            if event.subject == context.query.subject and event.object == context.query.object:
                direct_relation_counts[event.relation] += 1

        common_neighbors = sorted(subject_neighbors & object_neighbors)[:max_neighbors]
        return ToolResult(
            ok=True,
            data={
                "subject_neighbors": sorted(subject_neighbors)[:max_neighbors],
                "object_neighbors": sorted(object_neighbors)[:max_neighbors],
                "common_neighbors": common_neighbors,
                "direct_relation_counts": dict(direct_relation_counts),
                "entity_relation_counts": {
                    entity: dict(counter)
                    for entity, counter in entity_relation_counts.items()
                },
            },
            message="computed local temporal graph statistics",
            cost=self.cost,
        )
