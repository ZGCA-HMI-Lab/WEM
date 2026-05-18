"""Extract fixed-length video clips from BEHAVIOR-1K episodes for WEM training.

Output layout:
    {output_dir}/{task}/{episode}/first_frame.jpg
    {output_dir}/{task}/{episode}/clip_{i}/video.mp4
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as io
from tqdm import tqdm

CHUNK_FRAMES = 37
TARGET_FPS = 16


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize [T, H, W, C] array to (height, width); no-op if already correct size."""
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    return np.stack([cv2.resize(f, (width, height)) for f in frames])


def process_episode(
    vid_path: Path,
    json_path: Path,
    out_dir: Path,
    speed_factor: float,
    height: int,
    width: int,
) -> str | None:
    """Segment one episode into CHUNK_FRAMES-length clips. Returns an error string or None."""
    try:
        vframes, _, info = io.read_video(str(vid_path), pts_unit="sec")
    except Exception as e:
        return str(e)
    if vframes.size(0) == 0:
        return "empty video"

    fps_ratio = TARGET_FPS / (info["video_fps"] or 30.0)
    anno = json.loads(json_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    first_frame = resize_frames(vframes[:1].numpy(), height, width)[0]
    ok = cv2.imwrite(
        str(out_dir / "first_frame.jpg"),
        cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR),
    )
    if not ok:
        return "failed to write first_frame.jpg"

    def _start(skill):
        fd = skill.get("frame_duration", [0, 0])
        return fd[0][0] if isinstance(fd[0], list) else fd[0]

    skills = sorted(anno.get("skill_annotation", []), key=_start)

    clip_idx = 0
    for skill in skills:
        fd = skill.get("frame_duration", [0, 0])
        start_f, end_f = (fd[0][0], fd[0][1]) if isinstance(fd[0], list) else (fd[0], fd[1])

        # Skip frame 0 for the first skill to avoid overlap with the condition frame.
        if skill["skill_idx"] == 0 and start_f == 0:
            start_f = 1
        end_f = min(end_f, vframes.size(0))
        if start_f >= end_f:
            continue

        n_chunks = max(1, int((end_f - start_f) / speed_factor * fps_ratio / CHUNK_FRAMES))
        indices = np.linspace(start_f, end_f - 1, n_chunks * CHUNK_FRAMES, dtype=int)
        frames = resize_frames(vframes[torch.tensor(indices)].numpy(), height, width)

        for i in range(n_chunks):
            chunk = torch.from_numpy(frames[i * CHUNK_FRAMES : (i + 1) * CHUNK_FRAMES])
            clip_dir = out_dir / f"clip_{clip_idx}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            io.write_video(
                str(clip_dir / "video.mp4"),
                chunk,
                fps=TARGET_FPS,
                video_codec="libx264",
                options={"crf": "18"},
            )
            clip_idx += 1

    return None


def _worker(args):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    vid_path, json_path, out_dir, speed_factor, height, width = args
    err = process_episode(vid_path, json_path, out_dir, speed_factor, height, width)
    return vid_path.stem, err


def main():
    parser = argparse.ArgumentParser(description="Preprocess BEHAVIOR-1K into WEM training clips")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="B1K dataset root (must contain videos/ and annotations/ subdirectories)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root directory")
    parser.add_argument("--task_name", type=str, default="all",
                        help="Task ID to process (e.g. task-0002), or 'all' to process every task")
    parser.add_argument("--max_videos", type=int, default=-1,
                        help="Max episodes per task (-1 = unlimited)")
    parser.add_argument("--speed_factor", type=float, default=2.0,
                        help="Temporal compression factor (2.0 = 2x speed-up)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    videos_root = Path(args.root_dir) / "videos"
    anno_root = Path(args.root_dir) / "annotations"
    out_root = Path(args.output_dir)

    if args.task_name == "all":
        tasks = sorted(p.name for p in videos_root.iterdir() if p.is_dir())
    else:
        tasks = [args.task_name]

    jobs = []
    for task in tasks:
        vid_dir = videos_root / task / "observation.images.rgb.head"
        if not vid_dir.exists():
            print(f"[skip] {task}: video directory not found")
            continue
        episodes = sorted(vid_dir.glob("episode_*.mp4"))
        if args.max_videos > 0:
            episodes = episodes[: args.max_videos]
        for vid_path in episodes:
            json_path = anno_root / task / f"{vid_path.stem}.json"
            if not json_path.exists():
                continue
            jobs.append((
                vid_path, json_path,
                out_root / task / vid_path.stem,
                args.speed_factor, args.height, args.width,
            ))

    print(f"Processing {len(jobs)} episodes across {len(tasks)} tasks ({args.num_workers} workers)")
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for name, err in tqdm(pool.map(_worker, jobs), total=len(jobs)):
            if err:
                errors.append(f"{name}: {err}")

    print(f"\nDone. {len(jobs) - len(errors)} succeeded, {len(errors)} failed.")
    for e in errors[:10]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
