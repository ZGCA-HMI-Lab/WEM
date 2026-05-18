import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from third_party.wan.utils.utils import save_video
from wem.configs.wem import wem_cfg
from wem.models.generator import WEMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WEM video generation")

    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--wan_ckpt_dir",
                        default=os.environ.get("WAN_CKPT_DIR", "checkpoints/Wan2.2-TI2V-5B"))
    parser.add_argument("--qwen_ckpt_dir",
                        default=os.environ.get("QWEN_CKPT_DIR", "checkpoints/Qwen3-VL-2B-Instruct"))

    parser.add_argument("--image")
    parser.add_argument("--instructions", nargs="+")
    parser.add_argument("--output", default="output.mp4")

    parser.add_argument("--batch")
    parser.add_argument("--benchmark_root")
    parser.add_argument("--output_dir", default="results")

    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--state_guidance_scale", type=float, default=2.5)
    parser.add_argument("--ego_guidance_scale", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.batch is None and args.benchmark_root is None and (args.image is None or args.instructions is None):
        parser.error("Provide --benchmark_root, --batch, or both --image and --instructions.")

    return args


def init_distributed() -> tuple[int, int]:
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def apply_config_overrides(args: argparse.Namespace) -> None:
    wan = args.wan_ckpt_dir
    wem_cfg.t5_checkpoint = os.path.join(wan, "models_t5_umt5-xxl-enc-bf16.pth")
    wem_cfg.t5_tokenizer = os.path.join(wan, "google/umt5-xxl")
    wem_cfg.vae_checkpoint = os.path.join(wan, "Wan2.2_VAE.pth")
    wem_cfg.world_model.model_name = args.qwen_ckpt_dir


def fit_prompts(instructions: list[str], num_chunks: int | None) -> list[str]:
    """Optionally truncate or pad instruction list to exactly num_chunks entries."""
    if not instructions:
        raise ValueError("At least one instruction is required.")
    if num_chunks is None:
        return list(instructions)
    prompts = instructions[:num_chunks]
    if len(prompts) < num_chunks:
        prompts += [prompts[-1]] * (num_chunks - len(prompts))
    return prompts


def load_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_benchmark_tasks(benchmark_root: Path) -> list[dict]:
    tasks = []
    for task_dir in sorted(benchmark_root.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        image_path = task_dir / "first_frame.jpg"
        prompts_path = task_dir / "prompts.txt"
        if not image_path.is_file() or not prompts_path.is_file():
            continue
        tasks.append({
            "name": f"{task_dir.name}/0",
            "image": str(image_path),
            "instructions": load_text_lines(prompts_path),
        })
    return tasks


def run_one(
    generator: WEMGenerator,
    image: Image.Image,
    prompts: list[str],
    out_path: Path,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if out_path.exists() and not args.overwrite:
        print(f"[rank {rank}] skip {out_path.name} (exists)")
        return

    video = generator.generate(
        input_prompts=prompts,
        img=image,
        sampling_steps=args.num_steps,
        action_guide_scale=args.guidance_scale,
        state_guide_scale=args.state_guidance_scale,
        ego_state_guide_scale=args.ego_guidance_scale,
        seed=args.seed,
    )

    if video is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=str(out_path),
            fps=args.fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"[rank {rank}] saved → {out_path}")


def main() -> None:
    args = parse_args()
    rank, world_size = init_distributed()
    device_id = rank % torch.cuda.device_count()

    apply_config_overrides(args)

    generator = WEMGenerator(
        config=wem_cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=rank,
    )

    if args.batch or args.benchmark_root:
        if args.benchmark_root:
            tasks = load_benchmark_tasks(Path(args.benchmark_root))
        else:
            tasks: list = json.loads(Path(args.batch).read_text())
        output_dir = Path(args.output_dir)
        local_tasks = list(tasks)[rank::world_size]

        for task in local_tasks:
            image = Image.open(task["image"]).convert("RGB")
            prompts = fit_prompts(task["instructions"], args.num_chunks)
            out_path = output_dir / f"{task['name']}.mp4"
            print(f"[rank {rank}] generating {task['name']} ({len(prompts)} chunks)")
            run_one(generator, image, prompts, out_path, args, rank)
    else:
        image = Image.open(args.image).convert("RGB")
        prompts = fit_prompts(args.instructions, args.num_chunks)
        print(f"Generating {len(prompts)} chunks → {args.output}")
        run_one(generator, image, prompts, Path(args.output), args, rank)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
