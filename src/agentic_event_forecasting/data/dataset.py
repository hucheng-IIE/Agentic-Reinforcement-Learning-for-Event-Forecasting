"""Dataset utilities for temporal event forecasting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from agentic_event_forecasting.schema import Event, ForecastQuery, NewsArticle, parse_date, unique_relations


@dataclass
class EventDataset:
    """In-memory event forecasting dataset."""

    events: List[Event]
    news: List[NewsArticle]

    @property
    def relations(self) -> Tuple[str, ...]:
        return unique_relations(self.events)

    def sorted_events(self) -> List[Event]:
        return sorted(self.events, key=lambda item: item.timestamp)

    def build_queries(self, events: Sequence[Event] | None = None) -> List[ForecastQuery]:
        source_events = list(events) if events is not None else self.sorted_events()
        candidates = self.relations
        return [ForecastQuery.from_event(event, candidates) for event in source_events]


def _get(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def load_events_csv(path: str | Path) -> List[Event]:
    """Load events from CSV.

    Required logical columns are timestamp/date/time, subject, relation, and object.
    """
    events: List[Event] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            events.append(
                Event(
                    timestamp=parse_date(_get(row, "timestamp", "date", "time")),
                    subject=_get(row, "subject", "s"),
                    relation=_get(row, "relation", "r", "event_type"),
                    object=_get(row, "object", "o"),
                    source_id=_get(row, "source_id", "id", default=f"event-{idx}"),
                    metadata={k: v for k, v in row.items() if k not in {"timestamp", "date", "time", "subject", "s", "relation", "r", "event_type", "object", "o", "source_id", "id"}},
                )
            )
    return events


def load_news_csv(path: str | Path) -> List[NewsArticle]:
    """Load news documents from CSV."""
    articles: List[NewsArticle] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            entities = tuple(x.strip() for x in _get(row, "entities").split(";") if x.strip())
            relations = tuple(x.strip() for x in _get(row, "relations").split(";") if x.strip())
            articles.append(
                NewsArticle(
                    timestamp=parse_date(_get(row, "timestamp", "date", "time")),
                    title=_get(row, "title"),
                    text=_get(row, "text", "body", "content"),
                    source_id=_get(row, "source_id", "id", default=f"news-{idx}"),
                    entities=entities,
                    relations=relations,
                    metadata={k: v for k, v in row.items() if k not in {"timestamp", "date", "time", "title", "text", "body", "content", "source_id", "id", "entities", "relations"}},
                )
            )
    return articles


def chronological_split(
    events: Iterable[Event],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Tuple[List[Event], List[Event], List[Event]]:
    """Split events by time order."""
    ordered = sorted(events, key=lambda item: item.timestamp)
    if not ordered:
        return [], [], []
    train_end = int(len(ordered) * train_ratio)
    valid_end = train_end + int(len(ordered) * valid_ratio)
    train_end = max(1, min(train_end, len(ordered)))
    valid_end = max(train_end, min(valid_end, len(ordered)))
    return ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:]
