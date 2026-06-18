"""Structured historical event retrieval."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, List

from agentic_event_forecasting.schema import Event, parse_date
from agentic_event_forecasting.tools.base import ToolContext, ToolResult


class EventRetrievalTool:
    name = "event_retrieval"
    description = "Retrieve historical events before the query timestamp."
    cost = 1.0

    def run(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        subject = kwargs.get("subject", context.query.subject)
        obj = kwargs.get("object", context.query.object)
        relation = kwargs.get("relation")
        pair_only = bool(kwargs.get("pair_only", True))
        limit = int(kwargs.get("limit", 10))
        window_days = kwargs.get("window_days")
        end_time = parse_date(kwargs.get("end_time", context.query.timestamp))
        start_time = kwargs.get("start_time")

        if end_time > context.query.timestamp:
            return ToolResult(False, message="end_time cannot be after target timestamp", cost=self.cost)

        if start_time:
            start = parse_date(start_time)
        elif window_days:
            start = end_time - timedelta(days=int(window_days))
        else:
            start = None

        matched: List[Event] = []
        for event in context.events:
            if event.timestamp >= end_time:
                continue
            if start is not None and event.timestamp < start:
                continue
            if relation and event.relation != relation:
                continue
            if pair_only:
                if event.subject == subject and event.object == obj:
                    matched.append(event)
            elif event.matches_entity(subject) or event.matches_entity(obj):
                matched.append(event)

        matched = sorted(matched, key=lambda item: item.timestamp, reverse=True)[:limit]
        return ToolResult(
            ok=True,
            data={
                "events": [event.to_dict() for event in matched],
                "count": len(matched),
                "query": {
                    "subject": subject,
                    "object": obj,
                    "relation": relation,
                    "pair_only": pair_only,
                    "limit": limit,
                },
            },
            message=f"retrieved {len(matched)} historical events",
            cost=self.cost,
        )
