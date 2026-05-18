"""Tokenize caption.txt for every clip → text_ids.pt using Qwen3-VL tokenizer.
CPU-parallel: one worker process per CPU core subset.
"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

_tokenizer = None


def init_worker(model_path: str):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def process_caption(caption_file: Path) -> tuple[str, bool, str]:
    output_file = caption_file.parent / "text_ids.pt"
    if output_file.exists():
        return (str(caption_file), True, "skip")
    try:
        text = caption_file.read_text(encoding="utf-8").strip()
        if not text:
            torch.save(torch.tensor([], dtype=torch.long), output_file)
            return (str(caption_file), True, "empty")
        enc = _tokenizer(text, return_tensors="pt", add_special_tokens=False)
        torch.save(enc.input_ids[0], output_file)
        return (str(caption_file), True, "")
    except Exception as e:
        return (str(caption_file), False, str(e))


def find_caption_files(data_root: Path, task_filter: set | None) -> list:
    files = []
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        if task_filter and task_dir.name not in task_filter:
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            for clip_dir in sorted(ep_dir.iterdir()):
                if not clip_dir.is_dir() or not clip_dir.name.startswith("clip_"):
                    continue
                cap = clip_dir / "caption.txt"
                if cap.exists():
                    files.append(cap)
    return files


def main():
    parser = argparse.ArgumentParser(description="Tokenize clip captions using Qwen3-VL tokenizer")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True, help="Path to Qwen3-VL model directory")
    parser.add_argument("--tasks", default=None, help="Comma-separated task filter, e.g. task-0000,task-0001")
    parser.add_argument("--num_workers", type=int,
                        default=max(1, multiprocessing.cpu_count() // 2))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    task_filter = set(args.tasks.split(",")) if args.tasks else None
    caption_files = find_caption_files(data_root, task_filter)
    print(f"Found {len(caption_files)} caption files.")
    if not caption_files:
        return

    success = skip = empty = errors = 0
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args.model_path,),
    ) as executor:
        futures = {executor.submit(process_caption, f): f for f in caption_files}
        with tqdm(total=len(caption_files), desc="Tokenizing") as pbar:
            for future in as_completed(futures):
                _, ok, msg = future.result()
                if ok:
                    if msg == "skip":
                        skip += 1
                    elif msg == "empty":
                        empty += 1
                    else:
                        success += 1
                else:
                    errors += 1
                    print(f"Error: {msg}")
                pbar.update(1)

    print(f"Done. Success: {success} | Skipped: {skip} | Empty: {empty} | Errors: {errors}")


if __name__ == "__main__":
    main()
