"""Run a local end-to-end demo of the agentic forecasting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_event_forecasting.data import EventDataset, load_events_csv, load_news_csv
from agentic_event_forecasting.env import build_default_environment
from agentic_event_forecasting.inference import EventForecaster
from agentic_event_forecasting.reward import RewardScorer
from agentic_event_forecasting.rollout import RolloutGenerator
from agentic_event_forecasting.trainer import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agentic event forecasting demo.")
    parser.add_argument("--events", default=str(ROOT / "examples" / "events.csv"))
    parser.add_argument("--news", default=str(ROOT / "examples" / "news.csv"))
    parser.add_argument("--query-index", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events_csv(args.events)
    news = load_news_csv(args.news)
    dataset = EventDataset(events=events, news=news)
    queries = dataset.build_queries()
    query = queries[args.query_index]

    def env_factory():
        return build_default_environment(events=events, news=news, max_steps=6)

    forecaster = EventForecaster(env_factory=env_factory)
    trajectory = forecaster.predict(query)
    scorer = RewardScorer()
    reward = scorer.score(trajectory)

    rollout_generator = RolloutGenerator(env_factory=env_factory)
    trainer = GRPOTrainer(
        rollout_generator=rollout_generator,
        reward_scorer=scorer,
        config=GRPOConfig(group_size=args.group_size),
    )
    stats, records = trainer.train_step([query])

    payload = {
        "query": query.to_dict(),
        "prediction": None
        if trajectory.final_prediction is None
        else trajectory.final_prediction.to_dict(),
        "trajectory": trajectory.to_dict(),
        "reward": reward.to_dict(),
        "grpo_stats": stats.to_dict(),
        "advantages": [record.to_dict() for record in records],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
