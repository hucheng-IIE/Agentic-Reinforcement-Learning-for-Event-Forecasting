"""Base classes for environment tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Protocol

from agentic_event_forecasting.schema import Event, ForecastQuery, NewsArticle


@dataclass(frozen=True)
class ToolContext:
    """Read-only context provided to tools."""

    query: ForecastQuery
    events: List[Event]
    news: List[NewsArticle]
    step_index: int = 0


@dataclass
class ToolResult:
    """Structured tool execution result."""

    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    cost: float = 1.0


class Tool(Protocol):
    name: str
    description: str
    cost: float

    def run(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        ...


class ToolRegistry:
    """Name-based tool registry."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schema(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in sorted(self._tools.items())}
