from __future__ import annotations

import unittest
from datetime import date

from agentic_event_forecasting.agent import HeuristicReActPolicy
from agentic_event_forecasting.env import build_default_environment
from agentic_event_forecasting.evaluation import classification_metrics, trajectory_metrics
from agentic_event_forecasting.reward import RewardScorer
from agentic_event_forecasting.rollout import RolloutGenerator
from agentic_event_forecasting.schema import Event, ForecastQuery, NewsArticle
from agentic_event_forecasting.trainer import GRPOConfig, GRPOTrainer, compute_group_advantages


class CorePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            Event(date(2024, 1, 1), "A", "cooperate", "B", "e1"),
            Event(date(2024, 2, 1), "A", "consult", "B", "e2"),
            Event(date(2024, 3, 1), "A", "cooperate", "B", "e3"),
            Event(date(2024, 4, 1), "A", "conflict", "C", "e4"),
        ]
        self.news = [
            NewsArticle(
                date(2024, 3, 15),
                title="A and B expand cooperation",
                text="A and B discussed a cooperation program.",
                source_id="n1",
                entities=("A", "B"),
                relations=("cooperate",),
            )
        ]
        self.query = ForecastQuery(
            subject="A",
            object="B",
            timestamp=date(2024, 5, 1),
            query_id="q1",
            candidate_relations=("cooperate", "consult", "conflict"),
            ground_truth="cooperate",
        )

    def env_factory(self):
        return build_default_environment(self.events, self.news)

    def test_single_rollout_and_reward(self) -> None:
        env = self.env_factory()
        trajectory = env.run(self.query, HeuristicReActPolicy())
        self.assertTrue(trajectory.terminated)
        self.assertIsNotNone(trajectory.final_prediction)
        self.assertEqual(trajectory.final_prediction.label, "cooperate")

        reward = RewardScorer().score(trajectory)
        self.assertGreater(reward.total, 1.0)
        self.assertEqual(reward.outcome, 1.0)

    def test_group_advantage_and_grpo_stats(self) -> None:
        generator = RolloutGenerator(self.env_factory)
        trajectories = generator.generate_for_query(self.query, group_size=3)
        rewards = RewardScorer().score_group(trajectories)
        records = compute_group_advantages(trajectories, rewards)
        self.assertEqual(len(records), 3)

        trainer = GRPOTrainer(
            rollout_generator=generator,
            reward_scorer=RewardScorer(),
            config=GRPOConfig(group_size=3),
        )
        stats, records = trainer.train_step([self.query])
        self.assertEqual(stats.num_trajectories, 3)
        self.assertEqual(len(records), 3)

    def test_metrics(self) -> None:
        metrics = classification_metrics(["a", "b", "a"], ["a", "a", "a"])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        agent_metrics = trajectory_metrics([])
        self.assertEqual(agent_metrics["avg_tool_calls"], 0.0)


if __name__ == "__main__":
    unittest.main()
