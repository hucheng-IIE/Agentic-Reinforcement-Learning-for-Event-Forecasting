"""Evaluation metrics for predictions and agent behavior."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, List, Sequence

from agentic_event_forecasting.schema import Trajectory


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    per_label_f1: List[float] = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label_f1.append(f1)

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "accuracy": correct / len(y_true),
        "macro_f1": mean(per_label_f1) if per_label_f1 else 0.0,
        "micro_f1": micro_f1,
        "num_examples": len(y_true),
        "label_distribution": dict(Counter(y_true)),
    }


def trajectory_metrics(trajectories: Iterable[Trajectory]) -> dict:
    items = list(trajectories)
    if not items:
        return {
            "avg_tool_calls": 0.0,
            "invalid_call_rate": 0.0,
            "avg_total_cost": 0.0,
            "termination_reasons": {},
        }
    total_tool_calls = sum(item.tool_call_count for item in items)
    invalid_calls = sum(item.invalid_call_count for item in items)
    return {
        "avg_tool_calls": total_tool_calls / len(items),
        "invalid_call_rate": invalid_calls / max(1, total_tool_calls),
        "avg_total_cost": mean(item.total_cost for item in items),
        "termination_reasons": dict(Counter(item.termination_reason for item in items)),
    }
