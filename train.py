import argparse
import os
import random
import json
from functools import partial
import wandb
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from safetensors.torch import load_file

from wem.training.trainer import WanARTrainer, WEMTrainer
from wem.training import distributed, fsdp
from wem.data.datasets import WanARDataset, B1KDataset, RobotwinDataset, wan_collate_fn, b1k_collate_fn
from wem.models.wan_ar import WanARModel
from wem.models.wem import WEMModel
from wem.configs.wem import wem_cfg


def apply_wem_path_overrides(args: argparse.Namespace) -> None:
    if hasattr(args, "qwen_model_path") and args.qwen_model_path:
        wem_cfg.world_model.model_name = args.qwen_model_path
        wem_cfg.world_model.ckpt_path = args.qwen_model_path
    if hasattr(args, "t5_checkpoint") and args.t5_checkpoint:
        wem_cfg.t5_checkpoint = args.t5_checkpoint
    if hasattr(args, "t5_tokenizer") and args.t5_tokenizer:
        wem_cfg.t5_tokenizer = args.t5_tokenizer
    if hasattr(args, "vae_checkpoint") and args.vae_checkpoint:
        wem_cfg.vae_checkpoint = args.vae_checkpoint
    if hasattr(args, "cache_dir"):
        wem_cfg.cache_dir = args.cache_dir or None
    if hasattr(args, "device") and args.device:
        wem_cfg.device = args.device
        wem_cfg.world_model.device = args.device


def build_wem_model(args: argparse.Namespace) -> torch.nn.Module:
    apply_wem_path_overrides(args)
    if hasattr(args, "sink_size") and args.sink_size is not None:
        wem_cfg.wan_decoder.num_sink_frames = args.sink_size
    if hasattr(args, "cond_size"):
        wem_cfg.wan_decoder.num_cond_frames = args.cond_size
    if hasattr(args, "latent_chunk_size"):
        wem_cfg.wan_decoder.num_chunk_frames = args.latent_chunk_size

    def resolve_wan_io_dims(ckpt_path: str, default_in: int, default_out: int):
        if not ckpt_path:
            return default_in, default_out
        ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else os.path.dirname(ckpt_path)
        if not ckpt_dir:
            return default_in, default_out
        config_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.exists(config_path):
            return default_in, default_out
        try:
            with open(config_path, "r") as f:
                ckpt_cfg = json.load(f)
            return int(ckpt_cfg.get("in_dim", default_in)), int(ckpt_cfg.get("out_dim", default_out))
        except Exception:
            return default_in, default_out

    def normalize_decoder_state_dict(
        raw_state_dict: dict[str, torch.Tensor],
        decoder_model: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """
        Map checkpoint keys (possibly with legacy prefixes or old block numbering)
        to the current WanARModel parameter names.

        Handles:
        - Prefixes: wan_decoder., module.wan_decoder., _orig_mod.wan_decoder.,
                    model.wan_decoder., module., _orig_mod., model.
        - Old flat blocks.<i>. layout remapped to enc_blocks / world_dec_blocks /
          ego_dec_blocks.
        """
        decoder_state_dict = decoder_model.state_dict()
        normalized_state_dict = {}
        num_enc = len(getattr(decoder_model, "enc_blocks", []))
        num_dec = len(getattr(decoder_model, "world_dec_blocks", []))

        prefix_candidates = (
            "wan_decoder.",
            "module.wan_decoder.",
            "_orig_mod.wan_decoder.",
            "model.wan_decoder.",
            "module.",
            "_orig_mod.",
            "model.",
        )

        def remap_ddt_block_keys(key_wo_prefix: str) -> list[str]:
            candidates: list[str] = [key_wo_prefix]

            parts = key_wo_prefix.split(".")
            if len(parts) < 3 or not parts[1].isdigit():
                return candidates
            block_type = parts[0]
            idx = int(parts[1])
            suffix = ".".join(parts[2:])

            # Old flat blocks.<i> layout (legacy Wan checkpoint)
            if block_type in ("blocks", "layers"):
                if idx < num_enc:
                    candidates.append(f"enc_blocks.{idx}.{suffix}")
                else:
                    dec_idx = idx - num_enc
                    if 0 <= dec_idx < num_dec:
                        candidates.append(f"world_dec_blocks.{dec_idx}.{suffix}")
                        candidates.append(f"ego_dec_blocks.{dec_idx}.{suffix}")

            # Stage-1 single dec_blocks → both stage-2 decoder stacks
            elif block_type == "dec_blocks":
                candidates.append(f"world_dec_blocks.{idx}.{suffix}")
                candidates.append(f"ego_dec_blocks.{idx}.{suffix}")

            return candidates

        for key, value in raw_state_dict.items():
            candidate_keys = [key]
            for prefix in prefix_candidates:
                if key.startswith(prefix):
                    candidate_keys.append(key[len(prefix):])

            expanded_candidates: list[str] = []
            for cand in candidate_keys:
                expanded_candidates.extend(remap_ddt_block_keys(cand))

            matched = False
            for target_key in expanded_candidates:
                if target_key not in decoder_state_dict:
                    continue
                if decoder_state_dict[target_key].shape != value.shape:
                    print(
                        f"Skipping {key} -> {target_key} due to shape mismatch: "
                        f"{tuple(value.shape)} vs {tuple(decoder_state_dict[target_key].shape)}"
                    )
                    continue
                normalized_state_dict[target_key] = value
                matched = True

            if not matched:
                continue

        return normalized_state_dict

    def load_decoder_weights(model: WEMModel, ckpt_path: str, *, required: bool = False) -> int:
        def fail_or_warn(message: str) -> int:
            if required:
                raise FileNotFoundError(message)
            print(f"Warning: {message}")
            return 0

        if not ckpt_path:
            return fail_or_warn("empty Wan decoder checkpoint path")

        decoder = model.wan_decoder
        load_candidates = []

        if os.path.isfile(ckpt_path):
            load_candidates.append(ckpt_path)
        elif os.path.isdir(ckpt_path):
            index_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors.index.json")
            if os.path.exists(index_path):
                print(f"Loading Wan decoder weights from sharded checkpoint {ckpt_path}")
                with open(index_path, "r") as f:
                    index = json.load(f)

                weight_map = index.get("weight_map", {})
                loaded_shards = set()
                merged_state_dict = {}
                for shard_name in weight_map.values():
                    if shard_name in loaded_shards:
                        continue
                    shard_path = os.path.join(ckpt_path, shard_name)
                    if not os.path.exists(shard_path):
                        print(f"Warning: shard not found: {shard_path}")
                        continue
                    print(f"Loading shard: {shard_name}")
                    merged_state_dict.update(load_file(shard_path))
                    loaded_shards.add(shard_name)

                final_dict = normalize_decoder_state_dict(merged_state_dict, decoder)
                if not final_dict:
                    return fail_or_warn(f"no Wan decoder tensors matched checkpoint {ckpt_path}")
                missing, unexpected = decoder.load_state_dict(final_dict, strict=False)
                print(f"WanDecoder loaded tensors: {len(final_dict)}")
                print(f"WanDecoder: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                print(missing)
                return len(final_dict)

            load_candidates.extend([
                os.path.join(ckpt_path, "model.safetensors"),
                os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors"),
            ])
        else:
            return fail_or_warn(f"Wan decoder checkpoint path does not exist: {ckpt_path}")

        for candidate in load_candidates:
            if not os.path.exists(candidate):
                continue
            print(f"Loading Wan decoder weights from {candidate}")
            state_dict = load_file(candidate)
            final_dict = normalize_decoder_state_dict(state_dict, decoder)
            if not final_dict:
                return fail_or_warn(f"no Wan decoder tensors matched checkpoint {candidate}")
            missing, unexpected = decoder.load_state_dict(final_dict, strict=False)
            print(f"WanDecoder loaded tensors: {len(final_dict)}")
            print(f"WanDecoder: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            print(missing)
            return len(final_dict)

        return fail_or_warn(f"no supported checkpoint found under {ckpt_path}")

    inferred_in_dim, inferred_out_dim = resolve_wan_io_dims(
        args.ckpt_path,
        wem_cfg.wan_decoder.in_dim,
        wem_cfg.wan_decoder.out_dim,
    )
    wem_cfg.wan_decoder.in_dim = inferred_in_dim
    wem_cfg.wan_decoder.out_dim = inferred_out_dim
    wem_cfg.wan_decoder.use_state_conditioning = True

    model = WEMModel(wem_cfg)
    if args.finetune:
        if getattr(args, "decoder_ckpt_path", None):
            print(f"==================================================")
            print(f"Skipping official Wan weights. Loading Wan decoder weights directly from: {args.decoder_ckpt_path}")
            print(f"==================================================")
            load_decoder_weights(model, args.decoder_ckpt_path, required=True)
        elif getattr(args, "ckpt_path", None):
            load_decoder_weights(model, args.ckpt_path, required=True)

    return model


def build_wan_ar_model(args: argparse.Namespace) -> torch.nn.Module:
    enc_layers = int(getattr(wem_cfg.wan_decoder, "enc_layers", 24))
    model = WanARModel.init_from_wan(
        checkpoint_dir=args.ckpt_path,
        num_cond_frames=args.cond_size,
        num_chunk_frames=args.latent_chunk_size,
        num_chunks=args.num_chunks,
        enc_layers=enc_layers,
        dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16,
        use_dual_branch=False,
    )
    return model


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WEM training entrypoint")

    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)

    parser.add_argument("--cond_size", type=int, default=1)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--latent_chunk_size", type=int, default=10)
    parser.add_argument("--num_chunks", type=int, default=2)

    parser.add_argument(
        "--ckpt_path",
        "--ckpt-path",
        dest="ckpt_path",
        type=str,
        default=os.environ.get("WAN_CKPT_DIR", "checkpoints/Wan2.2-TI2V-5B"),
    )
    parser.add_argument(
        "--decoder_ckpt_path",
        "--decoder-ckpt-path",
        dest="decoder_ckpt_path",
        type=str,
        default=None,
        help="Overwrites wan_decoder weights with a custom trained checkpoint",
    )
    parser.add_argument("--finetune", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--data_root",
        "--data-root",
        dest="data_root",
        type=str,
        default=None,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["b1k", "robotwin"],
        default="b1k",
        help="Training dataset adapter to use.",
    )
    parser.add_argument(
        "--dataset_task_prefix",
        "--dataset-task-prefix",
        dest="dataset_task_prefix",
        type=str,
        default=None,
        help="Optional task directory prefix for dataset adapters that support task filtering.",
    )
    parser.add_argument(
        "--qwen_model_path",
        "--qwen-model",
        dest="qwen_model_path",
        type=str,
        default=os.environ.get("QWEN_CKPT_DIR", "checkpoints/Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument(
        "--t5_checkpoint",
        "--t5-checkpoint",
        dest="t5_checkpoint",
        type=str,
        default=wem_cfg.t5_checkpoint,
    )
    parser.add_argument(
        "--t5_tokenizer",
        "--t5-tokenizer",
        dest="t5_tokenizer",
        type=str,
        default=wem_cfg.t5_tokenizer,
    )
    parser.add_argument(
        "--vae_checkpoint",
        "--vae-checkpoint",
        dest="vae_checkpoint",
        type=str,
        default=wem_cfg.vae_checkpoint,
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        dest="cache_dir",
        type=str,
        default=wem_cfg.cache_dir,
    )

    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp_sharding", type=str, choices=["full", "hybrid"], default="full")
    parser.add_argument("--fsdp_mixed_precision", type=str, choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--fsdp_activation_checkpointing", action="store_true")
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument(
        "--fsdp_limit_all_gathers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fsdp_sync_module_states",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fsdp_forward_prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fsdp_auto_wrap_policy", type=str, choices=["transformer", "size"], default="transformer")
    parser.add_argument("--fsdp_min_num_params", type=int, default=int(1e7))
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--mask_loss_weight", type=float, default=0.3)
    parser.add_argument("--mask_loss_weight_floor_ratio", type=float, default=0.2)
    parser.add_argument("--wandb_project", type=str, default="WEM Training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_root is None:
        raise ValueError(
            "--data_root must be specified. "
            "Set it to the path of your dataset directory."
        )
    set_seed(args.seed)

    if args.num_warmup_steps is None:
        args.num_warmup_steps = max(1, int(args.max_steps * 0.05))
        if distributed.is_main_process():
            print(f"Auto-adjusting warmup steps to {args.num_warmup_steps} (5% of {args.max_steps})")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_dist = args.distributed or world_size > 1 or args.fsdp
    if use_dist:
        backend = "nccl" if torch.cuda.is_available() and "cuda" in args.device else "gloo"
        distributed.init_distributed(backend=backend, force=args.fsdp and world_size <= 1)
        device = distributed.setup_device(args.device)
    else:
        device = args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu"

    if args.stage == 1:
        model = build_wan_ar_model(args)
    elif args.stage == 2:
        model = build_wem_model(args)
    else:
        raise ValueError(f"Unknown stage {args.stage}, only stage 1 or 2 is supported.")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    fsdp_enabled = args.fsdp
    is_main = distributed.is_main_process()

    if fsdp_enabled and distributed.is_distributed():
        rank = distributed.get_rank()
        print(f"[Rank {rank}] Waiting for all ranks before FSDP initialization...")
        distributed.barrier()
        print(f"[Rank {rank}] All ranks ready, initializing FSDP...")

    if fsdp_enabled:
        model = fsdp.build_fsdp_model(model, args)
        if distributed.is_distributed():
            rank = distributed.get_rank()
            distributed.barrier()
            print(f"[Rank {rank}] FSDP initialization complete")
    else:
        model.to(device)

    if distributed.is_distributed():
        distributed.barrier()
        if is_main:
            print("All ranks synced, creating dataset...")

    dataset_cls = RobotwinDataset if args.dataset == "robotwin" else B1KDataset
    dataset_extra_kwargs = {}
    if args.dataset == "robotwin" and args.dataset_task_prefix:
        dataset_extra_kwargs["task_prefix"] = args.dataset_task_prefix

    if args.stage == 1:
        dataset = dataset_cls(
            data_root=args.data_root,
            stage1_only=True,
            **dataset_extra_kwargs,
        )

        trainer = WanARTrainer(
            model=model,
            cond_size=args.cond_size,
            chunk_size=args.latent_chunk_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            dtype=torch.float32,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            log_memory=args.log_memory,
            max_steps=args.max_steps,
            num_warmup_steps=args.num_warmup_steps,
        )
        collate_fn = partial(b1k_collate_fn, stage=1)

    elif args.stage == 2:
        dataset = dataset_cls(
            data_root=args.data_root,
            use_sink=getattr(wem_cfg.wan_decoder, "num_sink_frames", 0) > 0,
            num_sink_frames=getattr(wem_cfg.wan_decoder, "num_sink_frames", 0),
            **dataset_extra_kwargs,
        )

        trainer = WEMTrainer(
            model=model,
            chunk_size=args.latent_chunk_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            dtype=torch.float32,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            max_steps=args.max_steps,
            num_warmup_steps=args.num_warmup_steps,
            log_memory=args.log_memory,
            mask_loss_weight=args.mask_loss_weight,
            mask_loss_weight_floor_ratio=args.mask_loss_weight_floor_ratio,
        )
        collate_fn = partial(b1k_collate_fn, stage=2)

    sampler = None
    if distributed.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=True)

    if args.use_wandb and args.wandb_run_name:
        run_name = args.wandb_run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stage == 2:
        run_name = "Stage2_" + run_name

    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory="cuda" in str(device),
        collate_fn=collate_fn,
        drop_last=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    epoch = 0
    if is_main and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={k: str(v) for k, v in vars(args).items()},
            dir=run_dir,
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
    distributed.barrier()

    if is_main:
        print(f"Dataset size: {len(dataset)} samples")
        print("Starting training loop...")

    pbar = tqdm(
        total=args.max_steps,
        disable=not is_main,
        dynamic_ncols=True,
        mininterval=10.0,
    )
    try:
        while trainer.global_step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for _, batch in enumerate(dataloader):
                metrics = trainer.train_one_step(batch)
                if metrics["did_step"]:
                    if is_main:
                        pbar.set_postfix(
                            loss=f"{metrics['loss']:.4f}",
                            loss_flow=f"{metrics.get('loss_flow', 0.0):.4f}",
                            loss_mask=f"{metrics.get('loss_mask', 0.0):.4f}",
                            grad_norm=f"{metrics['grad_norm']:.4f}",
                            avg_timestep=f"{metrics['avg_timestep']:.2f}",
                            lr=f"{metrics['lr']:.2e}",
                            refresh=False,
                        )
                        pbar.update(1)
                        if args.use_wandb:
                            log: dict[str, float | int] = {"train/step": trainer.global_step}
                            for k, v in metrics.items():
                                if k == "did_step":
                                    continue
                                if isinstance(v, (int, float)):
                                    log[f"train/{k}"] = float(v)

                            if args.log_memory and torch.cuda.is_available():
                                mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                                log["train/max_mem_gb"] = mem
                            wandb.log(log, step=trainer.global_step)
                    if args.save_every > 0 and trainer.global_step % args.save_every == 0:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        step_dir = os.path.join(
                            run_dir, f"{trainer.global_step:06d}"
                        )
                        trainer.save_checkpoint(step_dir)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if is_main:
                            print(f"Saved checkpoint: {step_dir}")
                if trainer.global_step >= args.max_steps:
                    break
            epoch += 1
    finally:
        if is_main:
            pbar.close()
            if args.use_wandb:
                wandb.finish()


if __name__ == "__main__":
    main()
