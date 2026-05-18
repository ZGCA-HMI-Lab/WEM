from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


MANIP_PATTERNS = [
    r"\bgrasp(?:s|ing|ed)?\b",
    r"\bgrip(?:s|ping|ped)?\b",
    r"\bpick(?:s|ing|ed)?\s+up\b",
    r"\blift(?:s|ing|ed)?\b",
    r"\bhold(?:s|ing)?\b",
    r"\brelease(?:s|d|ing)?\b",
    r"\bdrop(?:s|ped|ping)?\b",
    r"\bplace(?:s|d|ing)?\b",
    r"\bput(?:s|ting)?\b",
    r"\bpush(?:es|ed|ing)?\b",
    r"\bpull(?:s|ed|ing)?\b",
    r"\bpress(?:es|ed|ing)?\b",
    r"\bopen(?:s|ed|ing)?\b",
    r"\bclose(?:s|d|ing)?\b",
    r"\bturning it\b",
    r"\bturns? .* (?:on|off)\b",
    r"\breach(?:es|ed|ing)? .*gripper\b",
    r"\bextend(?:s|ed|ing)? (?:its|his|her|both) (?:left |right )?(?:arm|gripper|hand)s?\b",
    r"\bretract(?:s|ed|ing)? (?:its|his|her|both) (?:left |right )?(?:arm|gripper|hand)s?\b",
    r"\blower(?:s|ed|ing)? (?:its|his|her|both) (?:left |right )?(?:arm|gripper|hand)s?\b",
    r"\braise(?:s|d|ing)? (?:its|his|her|both) (?:left |right )?(?:arm|gripper|hand)s?\b",
    r"\bmoves? (?:its|his|her|the|both) (?:left |right )?(?:arm|gripper|hand)s?\b",
    r"\buses? (?:its|his|her) (?:left |right )?(?:arm|gripper|hand)\b",
    r"\bwith (?:its|his|her) (?:left |right )?(?:arm|gripper|hand)\b",
]

NAV_PATTERNS = [
    r"\bnavigat(?:e|es|ed|ing)\b",
    r"\bmoves? forward\b",
    r"\bmoves? across\b",
    r"\bmoves? toward\b",
    r"\bmoves? through\b",
    r"\bcontinues? moving forward\b",
    r"\bapproach(?:es|ed|ing)?\b",
    r"\bturn(?:s|ed|ing)? (?:to|toward|away)\b",
    r"\bpositions? itself\b",
    r"\bcarries? .* while navigat(?:e|ing)\b",
]

DEFAULT_GENERATED_INITIAL_FRAMES = 1
DEFAULT_GENERATED_CHUNK_FRAMES = 37


@dataclass(frozen=True)
class ChunkMeta:
    index: int
    prompt: str
    phase: str
    gt_video_path: Path
    gt_frame_count: int


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_dir: Path
    first_frame_path: Path
    gt_video_path: Optional[Path]
    prompts: List[str]
    phase_labels: List[str]
    chunk_metas: List[ChunkMeta]

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_metas)

    @property
    def has_nav(self) -> bool:
        return any(label == "Nav" for label in self.phase_labels)

    @property
    def has_manip(self) -> bool:
        return any(label == "Manip" for label in self.phase_labels)


@dataclass(frozen=True)
class GeneratedSample:
    task_id: str
    sample_id: str
    video_path: Path


@dataclass
class VideoChunk:
    index: int
    prompt: str
    phase: str
    gt_frame_count: int
    frames: List[np.ndarray]
    start_frame_idx: int
    end_frame_idx: int


@dataclass
class SegmentedSample:
    task_id: str
    sample_id: str
    video_path: Path
    total_frame_count: int
    chunks: List[VideoChunk]


def classify_prompt(prompt: str) -> str:
    text = " ".join(prompt.lower().split())
    for pattern in MANIP_PATTERNS:
        if re.search(pattern, text):
            return "Manip"
    for pattern in NAV_PATTERNS:
        if re.search(pattern, text):
            return "Nav"
    return "Nav"


def load_text_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def numeric_stem_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 10**9, path.stem


def find_task_dirs(benchmark_root: Path) -> List[Path]:
    task_dirs = []
    for candidate in benchmark_root.iterdir():
        if candidate.is_dir() and re.fullmatch(r"task_\d+", candidate.name):
            task_dirs.append(candidate)
    task_dirs.sort(key=lambda path: int(path.name.split("_", 1)[1]))
    return task_dirs


def count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            return count
    finally:
        cap.release()

    frames = load_video_frames(video_path)
    return len(frames)


def load_video_frames(video_path: Path) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _allocate_segment_counts(total_frames: int, weights: List[int]) -> List[int]:
    if total_frames <= 0:
        return [0 for _ in weights]
    if not weights:
        return []

    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = np.clip(weights_arr, 1.0, None)
    raw = weights_arr / weights_arr.sum() * float(total_frames)
    counts = np.floor(raw).astype(int)
    remainder = raw - counts

    missing = total_frames - int(counts.sum())
    if missing > 0:
        order = np.argsort(-remainder)
        for idx in order[:missing]:
            counts[idx] += 1

    if total_frames >= len(weights):
        zero_indices = np.where(counts == 0)[0].tolist()
        while zero_indices:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                break
            recv = zero_indices.pop(0)
            counts[donor] -= 1
            counts[recv] += 1

    return counts.astype(int).tolist()


def segment_frames_by_gt_counts(frames: List[np.ndarray], gt_counts: List[int]) -> List[List[np.ndarray]]:
    counts = _allocate_segment_counts(len(frames), gt_counts)
    segments: List[List[np.ndarray]] = []
    cursor = 0
    for count in counts:
        segments.append(frames[cursor : cursor + count])
        cursor += count
    return segments


def resolve_generated_segmentation(segmentation_config: Optional[Dict] = None) -> Tuple[int, int]:
    config = segmentation_config or {}
    initial_frame_count = int(config.get("generated_initial_frames", DEFAULT_GENERATED_INITIAL_FRAMES))
    chunk_frame_count = int(config.get("generated_chunk_frames", DEFAULT_GENERATED_CHUNK_FRAMES))

    if initial_frame_count < 0:
        raise ValueError(f"generated_initial_frames must be >= 0, got {initial_frame_count}")
    if chunk_frame_count <= 0:
        raise ValueError(f"generated_chunk_frames must be > 0, got {chunk_frame_count}")

    return initial_frame_count, chunk_frame_count


def expected_generated_frame_count(num_chunks: int, segmentation_config: Optional[Dict] = None) -> int:
    initial_frame_count, chunk_frame_count = resolve_generated_segmentation(segmentation_config)
    return initial_frame_count + max(num_chunks, 0) * chunk_frame_count


def segment_frames_by_fixed_blocks(
    frames: List[np.ndarray],
    num_chunks: int,
    segmentation_config: Optional[Dict] = None,
) -> List[List[np.ndarray]]:
    initial_frame_count, chunk_frame_count = resolve_generated_segmentation(segmentation_config)
    segments: List[List[np.ndarray]] = []

    for index in range(num_chunks):
        start = initial_frame_count + index * chunk_frame_count
        end = start + chunk_frame_count
        if start >= len(frames):
            segments.append([])
        else:
            segments.append(frames[start : min(end, len(frames))])

    return segments


def build_segmented_sample(
    task: BenchmarkTask,
    sample: GeneratedSample,
    segmentation_config: Optional[Dict] = None,
) -> SegmentedSample:
    frames = load_video_frames(sample.video_path)
    initial_frame_count, chunk_frame_count = resolve_generated_segmentation(segmentation_config)
    expected_frame_count = expected_generated_frame_count(task.num_chunks, segmentation_config)
    if len(frames) != expected_frame_count:
        raise ValueError(
            f"{task.task_id}/{sample.sample_id}: generated video has {len(frames)} frames, "
            f"expected exactly {expected_frame_count} frames for "
            f"{initial_frame_count}+{chunk_frame_count}*{task.num_chunks} splitting"
        )

    segments = segment_frames_by_fixed_blocks(frames, task.num_chunks, segmentation_config)

    chunks: List[VideoChunk] = []
    for chunk_index, (meta, chunk_frames) in enumerate(zip(task.chunk_metas, segments)):
        start_frame_idx = initial_frame_count + chunk_index * chunk_frame_count
        end_frame_idx = start_frame_idx + max(len(chunk_frames) - 1, 0)
        chunks.append(
            VideoChunk(
                index=meta.index,
                prompt=meta.prompt,
                phase=meta.phase,
                gt_frame_count=meta.gt_frame_count,
                frames=chunk_frames,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
            )
        )

    return SegmentedSample(
        task_id=sample.task_id,
        sample_id=sample.sample_id,
        video_path=sample.video_path,
        total_frame_count=len(frames),
        chunks=chunks,
    )


def resolve_output_task_dir(output_root: Path, task_id: str) -> Optional[Path]:
    candidates = [
        output_root / task_id,
        output_root / "benchmark" / task_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def list_generated_samples(output_root: Path, task_id: str) -> List[GeneratedSample]:
    task_dir = resolve_output_task_dir(output_root, task_id)
    if task_dir is None:
        return []

    samples = []
    for video_path in sorted(task_dir.glob("*.mp4"), key=numeric_stem_key):
        if video_path.stem == "merged" or not video_path.stem.isdigit():
            continue
        samples.append(
            GeneratedSample(
                task_id=task_id,
                sample_id=video_path.stem,
                video_path=video_path,
            )
        )
    return samples


def list_ignored_generated_videos(output_root: Path, task_id: str) -> List[Path]:
    task_dir = resolve_output_task_dir(output_root, task_id)
    if task_dir is None:
        return []

    ignored = []
    for video_path in sorted(task_dir.glob("*.mp4"), key=numeric_stem_key):
        if video_path.stem == "merged":
            continue
        if not video_path.stem.isdigit():
            ignored.append(video_path)
    return ignored


def load_benchmark_tasks(benchmark_root: Path, task_ids: Optional[Iterable[str]] = None) -> List[BenchmarkTask]:
    selected = set(task_ids) if task_ids is not None else None
    tasks: List[BenchmarkTask] = []

    for task_dir in find_task_dirs(benchmark_root):
        if selected is not None and task_dir.name not in selected:
            continue

        prompts = load_text_lines(task_dir / "prompts.txt")
        if not prompts:
            continue

        phase_labels = load_text_lines(task_dir / "prompt_nav_manip.txt")
        if not phase_labels:
            phase_labels = [classify_prompt(prompt) for prompt in prompts]

        gt_chunk_paths = sorted(task_dir.glob("*.mp4"), key=numeric_stem_key)
        gt_chunk_paths = [path for path in gt_chunk_paths if path.stem.isdigit()]
        gt_video_path = task_dir / "video.mp4"

        if gt_chunk_paths:
            valid_count = min(len(prompts), len(phase_labels), len(gt_chunk_paths))
        elif gt_video_path.is_file():
            valid_count = min(len(prompts), len(phase_labels))
        else:
            continue

        prompts = prompts[:valid_count]
        phase_labels = phase_labels[:valid_count]

        chunk_metas = []
        if gt_chunk_paths:
            gt_chunk_paths = gt_chunk_paths[:valid_count]
            for idx, (prompt, phase, video_path) in enumerate(zip(prompts, phase_labels, gt_chunk_paths)):
                chunk_metas.append(
                    ChunkMeta(
                        index=idx,
                        prompt=prompt,
                        phase=phase,
                        gt_video_path=video_path,
                        gt_frame_count=count_video_frames(video_path),
                    )
                )
            full_gt_video_path: Optional[Path] = gt_video_path if gt_video_path.is_file() else None
        else:
            for idx, (prompt, phase) in enumerate(zip(prompts, phase_labels)):
                chunk_metas.append(
                    ChunkMeta(
                        index=idx,
                        prompt=prompt,
                        phase=phase,
                        gt_video_path=gt_video_path,
                        gt_frame_count=DEFAULT_GENERATED_CHUNK_FRAMES,
                    )
                )
            full_gt_video_path = gt_video_path

        tasks.append(
            BenchmarkTask(
                task_id=task_dir.name,
                task_dir=task_dir,
                first_frame_path=task_dir / "first_frame.jpg",
                gt_video_path=full_gt_video_path,
                prompts=prompts,
                phase_labels=phase_labels,
                chunk_metas=chunk_metas,
            )
        )

    return tasks


def select_tasks_for_shard(tasks: List[BenchmarkTask], shard_id: int, num_shards: int) -> List[BenchmarkTask]:
    if num_shards <= 1:
        return tasks
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard {shard_id} for num_shards={num_shards}")
    return [task for index, task in enumerate(tasks) if index % num_shards == shard_id]
