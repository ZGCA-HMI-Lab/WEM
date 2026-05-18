#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.constants import DIAGNOSTIC_METRICS, FORMAL_METRICS


def parse_metric_selection(metric_arg: str) -> List[str]:
    normalized = metric_arg.strip().lower()
    if normalized == "formal":
        return FORMAL_METRICS
    if normalized == "diagnostic":
        return DIAGNOSTIC_METRICS
    if normalized == "all":
        return FORMAL_METRICS + DIAGNOSTIC_METRICS

    lookup = {metric.lower(): metric for metric in FORMAL_METRICS + DIAGNOSTIC_METRICS}
    metrics = []
    for item in metric_arg.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in lookup:
            raise ValueError(f"Unknown metric: {item}")
        metrics.append(lookup[key])
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run HTEWorld evaluation")
    parser.add_argument("--output-root", required=True, help="Model output root that contains task_xx/*.mp4")
    parser.add_argument("--benchmark-root", default="benchmark", help="GT benchmark root")
    parser.add_argument(
        "--save-dir",
        default="results/hteworld_eval",
        help="Base directory for JSON/CSV reports; outputs are saved under <save-dir>/<model-name>/",
    )
    parser.add_argument("--config", default="", help="Optional YAML config that overrides hteworld_eval/config/default.yaml")
    parser.add_argument("--metrics", default="all", help="formal | diagnostic | all | comma-separated metric ids")
    parser.add_argument("--model-name", default="", help="Optional name used for result file prefixes")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of task ids, e.g. task_0 task_1")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    return parser.parse_args()


def main():
    args = parse_args()
    from eval.benchmark import HTEWorldEvaluator, load_hteworld_config

    config = load_hteworld_config(Path(args.config).expanduser() if args.config else None)
    selected_metrics = parse_metric_selection(args.metrics)

    evaluator = HTEWorldEvaluator(config)
    evaluator.evaluate(
        benchmark_root=Path(args.benchmark_root),
        output_root=Path(args.output_root),
        save_dir=Path(args.save_dir),
        selected_metrics=selected_metrics,
        task_ids=args.tasks,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        model_name=args.model_name or None,
    )


if __name__ == "__main__":
    main()
