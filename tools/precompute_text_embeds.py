"""T5-encode caption.txt for every clip → text_embeds.pt + text_embeds_null.pt.
Multi-GPU: one process per GPU. Each process encodes sequentially on its GPU,
then flushes results to disk via a thread pool.
"""
import argparse
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_task_filter(raw: str) -> list:
    tasks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.startswith("task-"):
            try:
                item = f"task-{int(item):04d}"
            except ValueError:
                item = f"task-{item}"
        tasks.append(item)
    return tasks


def find_clip_dirs(data_root: Path, task_filter: list | None) -> list:
    clip_dirs = []
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        if task_filter and task_dir.name not in task_filter:
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            for clip_dir in sorted(ep_dir.iterdir()):
                if clip_dir.is_dir() and clip_dir.name.startswith("clip_"):
                    clip_dirs.append(clip_dir)
    return clip_dirs


def distribute(clip_dirs: list, n: int) -> list:
    """Round-robin by episode for cache locality."""
    groups: dict = {}
    for cd in clip_dirs:
        groups.setdefault(cd.parent, []).append(cd)
    buckets = [[] for _ in range(n)]
    for i, clips in enumerate(groups.values()):
        buckets[i % n].extend(clips)
    return buckets


def gpu_worker(rank: int, gpu_id: int, clip_dirs: list, overwrite: bool,
               t5_pth: str, tokenizer_path: str, io_workers: int):
    device = f"cuda:{gpu_id}"
    from third_party.wan.modules.t5 import T5EncoderModel

    t0 = time.time()
    encoder = T5EncoderModel(text_len=512, checkpoint_path=t5_pth,
                             tokenizer_path=tokenizer_path, device=device)
    print(f"[GPU {gpu_id}] Loaded in {time.time() - t0:.1f}s, {len(clip_dirs)} clips")

    with torch.no_grad():
        raw = encoder([""], device)
        null_embed = (raw[0] if isinstance(raw, list) else raw).squeeze(0).cpu()

    results = []
    skip = errors = 0
    for clip_dir in tqdm(clip_dirs, desc=f"[GPU {gpu_id}]", position=rank, leave=True):
        out = clip_dir / "text_embeds.pt"
        null_out = clip_dir / "text_embeds_null.pt"
        if not overwrite and out.exists() and null_out.exists():
            skip += 1
            continue
        caption_file = clip_dir / "caption.txt"
        if not caption_file.exists():
            errors += 1
            continue
        try:
            prompt = caption_file.read_text(encoding="utf-8").strip()
            with torch.no_grad():
                if not prompt:
                    text_embed = null_embed.clone()
                else:
                    raw = encoder([prompt], device)
                    text_embed = (raw[0] if isinstance(raw, list) else raw).squeeze(0).cpu()
            results.append((clip_dir, text_embed, null_embed))
        except Exception as e:
            print(f"[GPU {gpu_id}] Error {clip_dir.name}: {e}")
            errors += 1

    def save_one(args):
        clip_dir, te, ne = args
        try:
            torch.save(te, clip_dir / "text_embeds.pt")
            torch.save(ne, clip_dir / "text_embeds_null.pt")
            return True
        except Exception:
            return False

    success = sum(ThreadPoolExecutor(io_workers).map(save_one, results))
    print(f"[GPU {gpu_id}] Done. Success: {success} | Skipped: {skip} | Errors: {errors}")


def resolve_gpus(num_gpus: int | None, gpu_ids_str: str | None) -> list:
    total = torch.cuda.device_count()
    if gpu_ids_str:
        ids = [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]
        for gid in ids:
            if gid >= total:
                raise ValueError(f"GPU {gid} not available (total: {total})")
        return ids
    return list(range(min(num_gpus or total, total)))


def main():
    parser = argparse.ArgumentParser(description="T5-encode clip captions")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--t5_pth", required=True, help="Path to models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--tokenizer_path", required=True, help="Path to google/umt5-xxl directory")
    parser.add_argument("--tasks", default=None, help="Task filter, e.g. task-0000,task-0001")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--gpu_ids", default=None, help="Specific GPU ids, e.g. 0,1,4,5")
    parser.add_argument("--io_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_filter = parse_task_filter(args.tasks) if args.tasks else None
    clip_dirs = find_clip_dirs(data_root, task_filter)
    print(f"Found {len(clip_dirs)} clips.")
    if not clip_dirs:
        return

    gpu_list = resolve_gpus(args.num_gpus, args.gpu_ids)
    n = min(len(gpu_list), len(clip_dirs))
    buckets = distribute(clip_dirs, n)

    mp.set_start_method("spawn", force=True)
    processes = []
    for rank, (gid, bucket) in enumerate(zip(gpu_list[:n], buckets)):
        if not bucket:
            continue
        p = mp.Process(target=gpu_worker,
                       args=(rank, gid, bucket, args.overwrite,
                             args.t5_pth, args.tokenizer_path, args.io_workers))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All workers finished.")


if __name__ == "__main__":
    main()
