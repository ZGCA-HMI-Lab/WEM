"""Extract Qwen3-VL visual embeddings for clips (video.mp4 → visual_embeds.pt)
and episodes (first_frame.jpg → first_frame_embeds.pt).
Multi-GPU: one process per GPU via mp.Process.
"""
import argparse
import math
import multiprocessing as mp
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def read_video_frames(video_path: Path) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _run_visual(model, pv, gt):
    out = model.visual(pv, grid_thw=gt)
    emb = out[0] if isinstance(out, tuple) else out
    return emb.squeeze(0).cpu()


def encode_video(model, processor, video_path: Path) -> torch.Tensor | None:
    frames = read_video_frames(video_path)
    if not frames:
        return None
    inputs = processor.video_processor(
        videos=frames,
        return_tensors="pt",
        video_metadata=[{"fps": 16.0, "total_num_frames": len(frames)}],
    )
    pv = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
    gt = inputs.get("video_grid_thw", inputs.get("grid_thw"))
    if pv is None:
        return None
    pv = pv.to(model.device, dtype=model.dtype)
    if gt is not None:
        gt = gt.to(model.device)
    return _run_visual(model, pv, gt)


def encode_image(model, processor, img_path: Path) -> torch.Tensor | None:
    image = Image.open(img_path).convert("RGB")
    inputs = processor.image_processor(images=image, return_tensors="pt")
    pv = inputs.get("pixel_values_images", inputs.get("pixel_values"))
    gt = inputs.get("image_grid_thw", inputs.get("grid_thw"))
    if pv is None:
        return None
    pv = pv.to(model.device, dtype=model.dtype)
    if gt is not None:
        gt = gt.to(model.device)
    return _run_visual(model, pv, gt)


def worker(gpu_id: int, episode_dirs: list, model_path: str, overwrite: bool):
    device = f"cuda:{gpu_id}"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    print(f"[GPU {gpu_id}] Loaded model, {len(episode_dirs)} episodes")

    skip = errors = 0
    with torch.no_grad():
        for ep_dir in tqdm(episode_dirs, position=gpu_id, desc=f"GPU {gpu_id}", leave=True):
            # Encode first frame
            img_path = ep_dir / "first_frame.jpg"
            img_out = ep_dir / "first_frame_embeds.pt"
            if img_path.exists() and (overwrite or not img_out.exists()):
                try:
                    emb = encode_image(model, processor, img_path)
                    if emb is not None:
                        torch.save(emb, img_out)
                    else:
                        errors += 1
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error {img_path}: {e}")
                    errors += 1

            # Encode each clip's video
            for clip_dir in sorted(ep_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith("clip_"):
                    continue
                vid_path = clip_dir / "video.mp4"
                vid_out = clip_dir / "visual_embeds.pt"
                if not vid_path.exists() or (not overwrite and vid_out.exists()):
                    skip += 1
                    continue
                try:
                    emb = encode_video(model, processor, vid_path)
                    if emb is not None:
                        torch.save(emb, vid_out)
                    else:
                        errors += 1
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
    parser = argparse.ArgumentParser(description="Extract Qwen3-VL visual embeddings")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True, help="Path to Qwen3-VL model directory")
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
        mp.Process(target=worker, args=(i, chunks[i], args.model_path, args.overwrite))
        for i in range(len(chunks))
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done.")


if __name__ == "__main__":
    main()
