"""Lightweight news retrieval based on temporal and lexical relevance."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from agentic_event_forecasting.tools.base import ToolContext, ToolResult


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_\\-]+", text)}


class NewsRetrievalTool:
    name = "news_retrieval"
    description = "Retrieve relevant news articles before the query timestamp."
    cost = 1.5

    def run(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        limit = int(kwargs.get("limit", 5))
        keywords = kwargs.get("keywords")
        if keywords is None:
            keywords = [context.query.subject, context.query.object]
        if isinstance(keywords, str):
            keywords = [keywords]

        keyword_tokens = set()
        for keyword in keywords:
            keyword_tokens.update(_tokens(str(keyword)))
        keyword_tokens.update(_tokens(context.query.subject))
        keyword_tokens.update(_tokens(context.query.object))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for article in context.news:
            if article.timestamp >= context.query.timestamp:
                continue
            article_tokens = _tokens(article.title + " " + article.text + " " + " ".join(article.entities))
            lexical_overlap = len(keyword_tokens & article_tokens)
            entity_bonus = 0
            if context.query.subject in article.entities:
                entity_bonus += 2
            if context.query.object in article.entities:
                entity_bonus += 2
            relation_bonus = len(set(context.query.candidate_relations) & set(article.relations))
            score = lexical_overlap + entity_bonus + relation_bonus
            if score <= 0:
                continue
            payload = article.to_dict()
            payload["score"] = score
            scored.append((score, payload))

        top = [payload for _, payload in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]
        return ToolResult(
            ok=True,
            data={"articles": top, "count": len(top), "keywords": list(keywords)},
            message=f"retrieved {len(top)} news articles",
            cost=self.cost,
        )
