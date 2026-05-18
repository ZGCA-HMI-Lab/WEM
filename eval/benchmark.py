from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import time

import yaml

from .constants import DIAGNOSTIC_METRICS, FORMAL_METRICS
from .dataset import (
    BenchmarkTask,
    VideoChunk,
    build_segmented_sample,
    list_generated_samples,
    list_ignored_generated_videos,
    load_benchmark_tasks,
    load_video_frames,
    resolve_generated_segmentation,
    segment_frames_by_fixed_blocks,
    select_tasks_for_shard,
)
from .metrics import (
    compute_cisr,
    compute_cpdm,
    compute_fphsc,
    compute_lpsa,
    compute_pmpa,
    compute_rcbd,
)
from .models import ModelRegistry
from .reporting import aggregate_metric_entries, build_summary, save_json, save_summary_csv


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def safe_path_component(value: str) -> str:
    cleaned = value.replace("/", "_").replace("\\", "_").strip()
    return cleaned or "model"


def load_hteworld_config(config_path: Optional[Path] = None) -> Dict:
    default_path = Path(__file__).resolve().parent / "config" / "default.yaml"
    config = _load_yaml(default_path)
    if config_path is not None:
        config = deep_merge(config, _load_yaml(config_path))

    checkpoints = config.setdefault("checkpoints", {})
    checkpoints.setdefault("clip_model", "")
    checkpoints.setdefault("raft_model", "")
    checkpoints.setdefault("musiq_model", "")
    checkpoints.setdefault("alexnet_model", "")

    return config


class HTEWorldEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.models = ModelRegistry(config)
        self.verbose = bool(config.get("verbose", True))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[HTEWorld] {message}", flush=True)

    def _load_gt_chunks(self, task: BenchmarkTask) -> List[VideoChunk]:
        gt_chunks: List[VideoChunk] = []
        if task.chunk_metas and all(meta.gt_video_path.stem.isdigit() for meta in task.chunk_metas):
            for meta in task.chunk_metas:
                frames = load_video_frames(meta.gt_video_path)
                gt_chunks.append(
                    VideoChunk(
                        index=meta.index,
                        prompt=meta.prompt,
                        phase=meta.phase,
                        gt_frame_count=len(frames),
                        frames=frames,
                        start_frame_idx=0,
                        end_frame_idx=max(len(frames) - 1, 0),
                    )
                )
            return gt_chunks

        if task.gt_video_path is None:
            return gt_chunks

        frames = load_video_frames(task.gt_video_path)
        initial_frame_count, chunk_frame_count = resolve_generated_segmentation(
            self.config.get("segmentation", {})
        )
        segments = segment_frames_by_fixed_blocks(
            frames,
            task.num_chunks,
            segmentation_config=self.config.get("segmentation", {}),
        )

        for meta, chunk_frames in zip(task.chunk_metas, segments):
            start_frame_idx = initial_frame_count + meta.index * chunk_frame_count
            gt_chunks.append(
                VideoChunk(
                    index=meta.index,
                    prompt=meta.prompt,
                    phase=meta.phase,
                    gt_frame_count=len(chunk_frames),
                    frames=chunk_frames,
                    start_frame_idx=start_frame_idx,
                    end_frame_idx=start_frame_idx + max(len(chunk_frames) - 1, 0),
                )
            )
        return gt_chunks

    def _empty_metric_entries(self, selected_metrics: Iterable[str]) -> Dict[str, List[Dict]]:
        return {metric_name: [] for metric_name in selected_metrics}

    def evaluate(
        self,
        benchmark_root: Path,
        output_root: Path,
        save_dir: Path,
        selected_metrics: Optional[Iterable[str]] = None,
        task_ids: Optional[Iterable[str]] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        model_name: Optional[str] = None,
    ) -> Dict:
        benchmark_root = benchmark_root.expanduser().resolve()
        output_root = output_root.expanduser().resolve()
        save_root = save_dir.expanduser().resolve()
        effective_model_name = model_name or output_root.name
        safe_model_name = safe_path_component(effective_model_name)
        save_dir = save_root / safe_model_name

        if selected_metrics is None:
            selected_metrics = FORMAL_METRICS + DIAGNOSTIC_METRICS
        selected_metrics = list(selected_metrics)

        tasks = load_benchmark_tasks(benchmark_root, task_ids=task_ids)
        tasks = select_tasks_for_shard(tasks, shard_id=shard_id, num_shards=num_shards)
        self._log(
            f"Starting evaluation: tasks={len(tasks)}, metrics={','.join(selected_metrics)}, "
            f"shard={shard_id}/{num_shards}, output_root={output_root}"
        )

        metric_entries = self._empty_metric_entries(selected_metrics)
        warnings = []
        ignored_generated_videos = {}
        segmentation_config = self.config.get("segmentation", {})
        generated_initial_frames, generated_chunk_frames = resolve_generated_segmentation(segmentation_config)

        for task_index, task in enumerate(tasks, start=1):
            task_start = time.perf_counter()
            self._log(f"Task {task_index}/{len(tasks)} {task.task_id}: loading generated samples")
            sample_specs = list_generated_samples(output_root, task.task_id)
            ignored_videos = list_ignored_generated_videos(output_root, task.task_id)
            if ignored_videos:
                ignored_generated_videos[task.task_id] = [str(path) for path in ignored_videos]
                warnings.append(
                    f"{task.task_id}: ignored {len(ignored_videos)} non-full-video mp4 files "
                    f"with non-numeric stems"
                )
                self._log(
                    f"Task {task.task_id}: ignored {len(ignored_videos)} non-full-video mp4 files "
                    f"({', '.join(path.name for path in ignored_videos[:5])}"
                    f"{'...' if len(ignored_videos) > 5 else ''})"
                )
            if not sample_specs:
                warnings.append(f"{task.task_id}: no generated full videos found under {output_root}")
                self._log(f"Task {task.task_id}: skipped, no generated full videos")
                continue

            segmented_samples = [
                build_segmented_sample(task, sample_spec, segmentation_config=segmentation_config)
                for sample_spec in sample_specs
            ]
            self._log(
                f"Task {task.task_id}: samples={len(segmented_samples)}, chunks={task.num_chunks}, "
                f"phases={'/'.join(task.phase_labels)}"
            )
            needs_gt_chunks = any(metric in metric_entries for metric in ("RCBD", "LPSA", "CISR", "PMPA", "CPDM", "FPHSC"))
            gt_chunks = self._load_gt_chunks(task) if needs_gt_chunks else []

            for sample_index, segmented_sample in enumerate(segmented_samples, start=1):
                sample_metrics = {}
                sample_start = time.perf_counter()
                self._log(
                    f"Task {task.task_id} sample {sample_index}/{len(segmented_samples)} "
                    f"{segmented_sample.sample_id}: evaluating"
                )

                if "RCBD" in metric_entries:
                    self._log(f"Task {task.task_id} sample {segmented_sample.sample_id}: RCBD")
                    sample_metrics["RCBD"] = compute_rcbd(segmented_sample, gt_chunks, self.models, self.config)
                if "LPSA" in metric_entries:
                    self._log(f"Task {task.task_id} sample {segmented_sample.sample_id}: LPSA")
                    sample_metrics["LPSA"] = compute_lpsa(segmented_sample, gt_chunks, self.models, self.config)
                if "CISR" in metric_entries:
                    self._log(f"Task {task.task_id} sample {segmented_sample.sample_id}: CISR")
                    sample_metrics["CISR"] = compute_cisr(segmented_sample, gt_chunks, self.models, self.config)
                if "PMPA" in metric_entries:
                    self._log(f"Task {task.task_id} sample {segmented_sample.sample_id}: PMPA")
                    sample_metrics["PMPA"] = compute_pmpa(segmented_sample, gt_chunks, self.models, self.config)
                if "CPDM" in metric_entries:
                    self._log(f"Task {task.task_id} sample {segmented_sample.sample_id}: CPDM")
                    sample_metrics["CPDM"] = compute_cpdm(segmented_sample, gt_chunks, self.models, self.config)
                if "FPHSC" in metric_entries:
                    self._log(f"Task {task.task_id} sample {segmented_sample.sample_id}: FPHSC")
                    sample_metrics["FPHSC"] = compute_fphsc(segmented_sample.chunks, gt_chunks, self.models, self.config)

                for metric_name, metric_result in sample_metrics.items():
                    metric_entries[metric_name].append(
                        {
                            "task_id": segmented_sample.task_id,
                            "sample_id": segmented_sample.sample_id,
                            "video_path": str(segmented_sample.video_path),
                            "chunk_count": len(segmented_sample.chunks),
                            "chunk_frame_counts": [len(chunk.frames) for chunk in segmented_sample.chunks],
                            "expected_chunk_frames": generated_chunk_frames,
                            **metric_result,
                        }
                    )
                self._log(
                    f"Task {task.task_id} sample {segmented_sample.sample_id}: done in "
                    f"{time.perf_counter() - sample_start:.1f}s"
                )
            self._log(f"Task {task.task_id}: done in {time.perf_counter() - task_start:.1f}s")

        aggregated_metrics = {
            metric_name: aggregate_metric_entries(entries)
            for metric_name, entries in metric_entries.items()
        }

        results = {
            "meta": {
                "model_name": effective_model_name,
                "model_dir_name": safe_model_name,
                "benchmark_root": str(benchmark_root),
                "output_root": str(output_root),
                "save_root": str(save_root),
                "save_dir": str(save_dir),
                "selected_metrics": selected_metrics,
                "task_count": len(tasks),
                "shard_id": shard_id,
                "num_shards": num_shards,
                "metric_backends": {
                    "RCBD": "LPIPS + RAFT boundary dynamics matched to GT",
                    "LPSA": "late-weighted CLIP prefix end-state alignment to GT",
                    "CISR": "same-task GT step retrieval with CLIP video features",
                    "PMPA": "RAFT temporal motion profile alignment to GT",
                    "CPDM": "same-task opposite-phase contrastive margin",
                    "FPHSC": "RAFT change-region cropped CLIP phase handoff similarity",
                },
                "generated_segmentation": {
                    "initial_frames": generated_initial_frames,
                    "chunk_frames": generated_chunk_frames,
                    "protocol": "require exact initial_frames + chunk_frames * instruction_count frames; skip initial_frames, then split fixed blocks",
                },
                "ignored_generated_videos": ignored_generated_videos,
                "warnings": warnings,
            },
            "metrics": aggregated_metrics,
        }

        summary = build_summary(
            results=results,
            formal_metrics=[metric for metric in FORMAL_METRICS if metric in aggregated_metrics],
            diagnostic_metrics=[metric for metric in DIAGNOSTIC_METRICS if metric in aggregated_metrics],
        )
        results["summary"] = summary

        file_stem = safe_model_name
        if num_shards > 1:
            file_stem = f"{safe_model_name}_shard_{shard_id:03d}_of_{num_shards:03d}"

        save_json(save_dir / f"{file_stem}_results.json", results)
        save_json(save_dir / f"{file_stem}_summary.json", summary)
        save_summary_csv(save_dir / f"{file_stem}_summary.csv", summary)
        return results
