"""Relation distribution statistics for historical evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_event_forecasting.tools.base import ToolContext, ToolResult


class RelationStatisticsTool:
    name = "relation_statistics"
    description = "Compute historical relation frequencies for the target entity pair or entities."
    cost = 0.8

    def run(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        pair_only = bool(kwargs.get("pair_only", True))
        limit = int(kwargs.get("limit", 10))
        counts: Counter[str] = Counter()
        recent_examples = []

        for event in sorted(context.events, key=lambda item: item.timestamp, reverse=True):
            if event.timestamp >= context.query.timestamp:
                continue
            if pair_only:
                if event.subject != context.query.subject or event.object != context.query.object:
                    continue
            elif not (event.matches_entity(context.query.subject) or event.matches_entity(context.query.object)):
                continue
            counts[event.relation] += 1
            if len(recent_examples) < limit:
                recent_examples.append(event.to_dict())

        top_relation = counts.most_common(1)[0][0] if counts else ""
        return ToolResult(
            ok=True,
            data={
                "relation_counts": dict(counts),
                "top_relation": top_relation,
                "total": sum(counts.values()),
                "pair_only": pair_only,
                "recent_examples": recent_examples,
            },
            message=f"computed relation statistics over {sum(counts.values())} events",
            cost=self.cost,
        )
