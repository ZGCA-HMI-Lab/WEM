"""VAE-encode first_frame.jpg (per episode) and video.mp4 (per clip) → first_frame_latents.pt / latents.pt.
Multi-GPU: one process per GPU via mp.Process.
"""
import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_party.wan.modules.vae2_2 import Wan2_2_VAE


def read_video_tensor(video_path: Path) -> torch.Tensor | None:
    """Read video.mp4 → float tensor [C, T, H, W] normalized to [-1, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    arr = np.stack(frames, axis=0)  # [T, H, W, C]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).float()  # [C, T, H, W]
    return t / 127.5 - 1.0


def worker(gpu_id: int, episode_dirs: list, vae_pth: str, overwrite: bool):
    device = f"cuda:{gpu_id}"
    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device)
    print(f"[GPU {gpu_id}] Loaded VAE, {len(episode_dirs)} episodes")

    skip = errors = 0
    for ep_dir in tqdm(episode_dirs, position=gpu_id, desc=f"GPU {gpu_id}", leave=True):
        # Encode first frame
        img_path = ep_dir / "first_frame.jpg"
        img_out = ep_dir / "first_frame_latents.pt"
        if img_path.exists() and (overwrite or not img_out.exists()):
            try:
                img = Image.open(img_path).convert("RGB")
                x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
                x = (x / 127.5 - 1.0).unsqueeze(1)  # [C, 1, H, W]
                latents = vae.encode([x.to(device)])[0]
                torch.save(latents.cpu(), img_out)
            except Exception as e:
                print(f"[GPU {gpu_id}] Error {img_path}: {e}")
                errors += 1

        # Encode each clip's video
        for clip_dir in sorted(ep_dir.iterdir()):
            if not clip_dir.is_dir() or not clip_dir.name.startswith("clip_"):
                continue
            vid_path = clip_dir / "video.mp4"
            vid_out = clip_dir / "latents.pt"
            if not vid_path.exists() or (not overwrite and vid_out.exists()):
                skip += 1
                continue
            try:
                x = read_video_tensor(vid_path)
                if x is None:
                    errors += 1
                    continue
                latents = vae.encode([x.to(device)])[0]
                torch.save(latents.cpu(), vid_out)
            except Exception as e:
                print(f"[GPU {gpu_id}] Error {vid_path}: {e}")
                errors += 1

    print(f"[GPU {gpu_id}] Done. Skipped: {skip} | Errors: {errors}")


def collect_episode_dirs(data_root: Path, task_filter: set | None) -> list:
    dirs = []
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        if task_filter and task_dir.name not in task_filter:
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if ep_dir.is_dir() and ep_dir.name.startswith("episode_"):
                dirs.append(ep_dir)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="VAE-encode first frames and video clips")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vae_pth", required=True, help="Path to Wan2.2_VAE.pth")
    parser.add_argument("--tasks", default=None, help="Comma-separated task filter, e.g. task-0000,task-0001")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_filter = set(args.tasks.split(",")) if args.tasks else None
    episode_dirs = collect_episode_dirs(data_root, task_filter)
    print(f"Found {len(episode_dirs)} episodes.")
    if not episode_dirs:
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available.")
    n = min(args.num_gpus or num_gpus, num_gpus, len(episode_dirs))

    size = math.ceil(len(episode_dirs) / n)
    chunks = [episode_dirs[i:i + size] for i in range(0, len(episode_dirs), size)]

    mp.set_start_method("spawn", force=True)
    processes = [
        mp.Process(target=worker, args=(i, chunks[i], args.vae_pth, args.overwrite))
        for i in range(len(chunks))
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done.")


if __name__ == "__main__":
    main()
