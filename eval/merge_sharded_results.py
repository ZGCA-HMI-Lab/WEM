#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.constants import DIAGNOSTIC_METRICS, FORMAL_METRICS
from eval.reporting import aggregate_metric_entries, build_summary, save_json, save_summary_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded HTEWorld result files")
    parser.add_argument("result_files", nargs="+", help="Per-shard *_results.json files")
    parser.add_argument("--output", required=True, help="Merged result JSON path")
    return parser.parse_args()


def summary_stem_for(output_path: Path) -> str:
    if output_path.stem.endswith("_results"):
        return output_path.stem[: -len("_results")] + "_summary"
    return output_path.stem + "_summary"


def main():
    args = parse_args()
    merged = None

    for result_file in args.result_files:
        payload = json.loads(Path(result_file).read_text(encoding="utf-8"))
        if merged is None:
            merged = {
                "meta": payload.get("meta", {}),
                "metrics": {metric: {"entries": []} for metric in payload.get("metrics", {})},
            }
            merged["meta"]["merged_from"] = args.result_files
        for metric_name, metric_payload in payload.get("metrics", {}).items():
            merged["metrics"].setdefault(metric_name, {"entries": []})
            merged["metrics"][metric_name]["entries"].extend(metric_payload.get("entries", []))

    if merged is None:
        raise ValueError("No result files were provided")

    merged["metrics"] = {
        metric_name: aggregate_metric_entries(metric_payload["entries"])
        for metric_name, metric_payload in merged["metrics"].items()
    }
    merged["summary"] = build_summary(
        merged,
        formal_metrics=[metric for metric in FORMAL_METRICS if metric in merged["metrics"]],
        diagnostic_metrics=[metric for metric in DIAGNOSTIC_METRICS if metric in merged["metrics"]],
    )

    output_path = Path(args.output).expanduser().resolve()
    summary_stem = summary_stem_for(output_path)
    save_json(output_path, merged)
    save_json(output_path.with_name(summary_stem + ".json"), merged["summary"])
    save_summary_csv(output_path.with_name(summary_stem + ".csv"), merged["summary"])


if __name__ == "__main__":
    main()
