#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TRAIN_TASKS = [f"task-{idx:04d}" for idx in range(9)] + ["task-0010"]


def episode_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 10**12, path.name


def clip_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 10**12, path.name


def ensure_clean_or_create(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_train(
    src_root: Path,
    dst_root: Path,
    skip_first_episodes: int,
    dry_run: bool,
) -> dict[str, int]:
    stats = {
        "tasks": 0,
        "episodes": 0,
        "clips": 0,
        "missing_caption": 0,
        "missing_mask": 0,
    }

    for task_name in TRAIN_TASKS:
        task_src = src_root / task_name
        if not task_src.is_dir():
            raise FileNotFoundError(f"Missing training task directory: {task_src}")

        task_clip_count = 0
        stats["tasks"] += 1
        episodes = sorted(
            [path for path in task_src.iterdir() if path.is_dir() and path.name.startswith("episode_")],
            key=episode_key,
        )
        episodes = episodes[skip_first_episodes:]

        for episode_src in episodes:
            clips = sorted(
                [path for path in episode_src.iterdir() if path.is_dir() and path.name.startswith("clip_")],
                key=clip_key,
            )
            copied_any_clip = False

            for clip_src in clips:
                caption_src = clip_src / "caption_api.txt"
                mask_src = clip_src / "camera_motion_mask.npz"

                if not caption_src.is_file():
                    stats["missing_caption"] += 1
                    continue
                if not mask_src.is_file():
                    stats["missing_mask"] += 1
                    continue

                rel_clip = clip_src.relative_to(src_root)
                clip_dst = dst_root / rel_clip
                copy_file(caption_src, clip_dst / "caption.txt", dry_run=dry_run)
                copy_file(mask_src, clip_dst / "mask.npz", dry_run=dry_run)

                copied_any_clip = True
                stats["clips"] += 1
                task_clip_count += 1

            if copied_any_clip:
                stats["episodes"] += 1

        print(f"Prepared train {task_name}: {task_clip_count} clips")

    return stats


def prepare_eval(src_root: Path, dst_root: Path, dry_run: bool) -> dict[str, int]:
    if not src_root.is_dir():
        raise FileNotFoundError(f"Missing eval source directory: {src_root}")

    stats = {"tasks": 0, "files": 0}
    task_dirs = sorted(
        [path for path in src_root.iterdir() if path.is_dir() and path.name.startswith("task_")],
        key=lambda path: path.name,
    )

    for task_src in task_dirs:
        required = ["first_frame.jpg", "video.mp4", "prompts.txt", "prompt_nav_manip.txt"]
        missing = [name for name in required if not (task_src / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{task_src} missing required eval files: {missing}")

        stats["tasks"] += 1
        for file_src in task_src.rglob("*"):
            if not file_src.is_file():
                continue
            rel_file = file_src.relative_to(src_root)
            copy_file(file_src, dst_root / rel_file, dry_run=dry_run)
            stats["files"] += 1

        if stats["tasks"] % 50 == 0:
            print(f"Prepared eval tasks: {stats['tasks']}/{len(task_dirs)}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HTEWorld dataset folder for Hugging Face upload")
    parser.add_argument("--train_src", type=Path, default=Path("/home/lzy/pan/data/b1k"))
    parser.add_argument("--eval_src", type=Path, default=Path("/home/lzy/pan/data/hteworld_eval"))
    parser.add_argument("--output_dir", type=Path, default=Path("/home/lzy/hteworld"))
    parser.add_argument("--skip_first_episodes", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    train_dst = output_dir / "train"
    eval_dst = output_dir / "eval"

    if not args.dry_run:
        ensure_clean_or_create(train_dst, overwrite=args.overwrite)
        ensure_clean_or_create(eval_dst, overwrite=args.overwrite)

    train_stats = prepare_train(
        src_root=args.train_src.expanduser().resolve(),
        dst_root=train_dst,
        skip_first_episodes=args.skip_first_episodes,
        dry_run=args.dry_run,
    )
    eval_stats = prepare_eval(
        src_root=args.eval_src.expanduser().resolve(),
        dst_root=eval_dst,
        dry_run=args.dry_run,
    )

    print(f"Output: {output_dir}")
    print(
        "Train: "
        f"{train_stats['tasks']} tasks, {train_stats['episodes']} episodes, "
        f"{train_stats['clips']} clips, "
        f"missing_caption={train_stats['missing_caption']}, "
        f"missing_mask={train_stats['missing_mask']}"
    )
    print(f"Eval: {eval_stats['tasks']} tasks, {eval_stats['files']} files")


if __name__ == "__main__":
    main()
