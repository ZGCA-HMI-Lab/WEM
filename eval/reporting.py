from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import csv
import json


def aggregate_metric_entries(entries: List[Dict]) -> Dict:
    valid_entries = [entry for entry in entries if entry.get("status") == "ok" and entry.get("value") is not None]
    skipped_entries = [entry for entry in entries if entry.get("status") != "ok" or entry.get("value") is None]

    per_task = {}
    for entry in valid_entries:
        per_task.setdefault(entry["task_id"], []).append(float(entry["value"]))

    per_task_average = {
        task_id: sum(values) / len(values)
        for task_id, values in per_task.items()
        if values
    }

    global_average = None
    if valid_entries:
        global_average = sum(float(entry["value"]) for entry in valid_entries) / len(valid_entries)

    return {
        "global_average": global_average,
        "count_valid": len(valid_entries),
        "count_skipped": len(skipped_entries),
        "per_task_average": per_task_average,
        "entries": entries,
    }


def build_summary(results: Dict, formal_metrics: Iterable[str], diagnostic_metrics: Iterable[str]) -> Dict:
    summary = {
        "formal": {},
        "diagnostic": {},
    }

    for metric_name in formal_metrics:
        metric_result = results["metrics"].get(metric_name, {})
        summary["formal"][metric_name] = metric_result.get("global_average")

    for metric_name in diagnostic_metrics:
        metric_result = results["metrics"].get(metric_name, {})
        summary["diagnostic"][metric_name] = metric_result.get("global_average")

    return summary


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary_csv(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "value"])
        for group in ("formal", "diagnostic"):
            for metric_name, value in summary.get(group, {}).items():
                writer.writerow([group, metric_name, value])

