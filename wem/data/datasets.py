from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from wem.utils.masks import resize_soft_mask
from wem.utils.chat_template import QWEN_CHAT_TOKENS


class WanARDataset(Dataset):
    def __init__(self, data_root: str, seed: Optional[int] = 0, dtype: torch.dtype = torch.float32):
        self.seed = seed
        self.dtype = dtype
        self.data_root = Path(data_root)
        self.samples = []
        self._index_dataset()

    def __len__(self) -> int:
        return len(self.samples)

    def _index_dataset(self):
        for video_dir in sorted(d for d in self.data_root.iterdir() if d.is_dir()):
            self.samples.append(video_dir)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_dir = self.samples[idx]
        latents = torch.load(video_dir / "latents.pt", map_location="cpu")
        text_embeds = torch.load(video_dir / "text_embeds.pt", map_location="cpu")
        return {"latents": latents, "context": text_embeds}


class B1KDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        tokenizer: Any = None,
        max_context_length: int = 12,
        first_frame_token_id: int = QWEN_CHAT_TOKENS.vision_start_token_id,
        end_frame_token_id: int = QWEN_CHAT_TOKENS.vision_end_token_id,
        video_token_id: int = QWEN_CHAT_TOKENS.video_token_id,
        pad_token_id: int = QWEN_CHAT_TOKENS.pad_token_id,
        ramp_window: int = 2,
        stage1_only: bool = False,
        use_sink: bool = False,
        num_sink_frames: int = 0,
        latent_spatial_align: str = "none",
        latent_spatial_multiple: int = 1,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.first_frame_token_id = first_frame_token_id
        self.end_frame_token_id = end_frame_token_id
        self.video_token_id = video_token_id
        self.pad_token_id = pad_token_id
        self.ramp_window = ramp_window
        self.stage1_only = stage1_only
        self.use_sink = bool(use_sink and num_sink_frames > 0)
        self.num_sink_frames = int(max(0, num_sink_frames))
        self.latent_spatial_align = latent_spatial_align
        self.latent_spatial_multiple = int(max(1, latent_spatial_multiple))
        if self.latent_spatial_align not in ("none", "crop"):
            raise ValueError(
                f"Unknown latent_spatial_align={self.latent_spatial_align!r}; expected 'none' or 'crop'."
            )

        self.im_start_token_id = QWEN_CHAT_TOKENS.im_start_token_id
        self.im_end_token_id = QWEN_CHAT_TOKENS.im_end_token_id
        self.user_token_id = QWEN_CHAT_TOKENS.user_token_id
        self.assistant_token_id = QWEN_CHAT_TOKENS.assistant_token_id
        self.newline_token_id = QWEN_CHAT_TOKENS.newline_token_id

        self.samples = []
        self._index_dataset()
        mode = "stage1-only" if self.stage1_only else "full-stage2"
        print(f"{self.__class__.__name__}({mode}): {len(self.samples)} samples from {self.data_root}")

    @staticmethod
    def _normalize_first_frame_latents(first_frame: torch.Tensor, video_dir: Path) -> torch.Tensor:
        if first_frame.dim() == 5:
            first_frame = first_frame.squeeze(0)
        if first_frame.dim() == 4 and first_frame.shape[0] == 1:
            first_frame = first_frame.permute(1, 0, 2, 3)
        if first_frame.dim() != 4:
            raise ValueError(
                f"Expected first_frame_latents.pt under {video_dir} to have shape [C, T, H, W], "
                f"got {tuple(first_frame.shape)}"
            )
        return first_frame

    def _load_first_frame_latents(self, video_dir: Path, num_frames: int) -> torch.Tensor:
        first_frame = self._normalize_first_frame_latents(
            torch.load(video_dir / "first_frame_latents.pt", map_location="cpu"),
            video_dir,
        )
        if first_frame.shape[1] != num_frames:
            raise ValueError(
                f"Requested {num_frames} sink/cond frames, got {tuple(first_frame.shape)} from {video_dir}"
            )
        return first_frame

    def _aligned_latent_spatial_size(self, height: int, width: int) -> Tuple[int, int]:
        if self.latent_spatial_align == "none":
            return int(height), int(width)
        multiple = self.latent_spatial_multiple
        target_h = int(height) - (int(height) % multiple)
        target_w = int(width) - (int(width) % multiple)
        if target_h <= 0 or target_w <= 0:
            raise ValueError(
                f"Cannot crop latent spatial size {(height, width)} to a positive multiple of {multiple}."
            )
        return target_h, target_w

    @staticmethod
    def _crop_latents_to_size(latents: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        return latents[:, :, :target_h, :target_w]

    @staticmethod
    def _crop_mask_to_size(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        return mask[:, :target_h, :target_w]

    def _collect_episode_dirs(self):
        task_dirs = sorted(d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("task-"))
        collected = []
        for tdir in task_dirs:
            collected.extend(d for d in tdir.iterdir() if d.is_dir() and d.name.startswith("episode_"))
        return sorted(collected)

    def _index_dataset(self):
        for v_dir in self._collect_episode_dirs():
            required_episode_file = "first_frame_latents.pt" if self.stage1_only else "first_frame_embeds.pt"
            if not (v_dir / required_episode_file).exists():
                print(f"Skipping {v_dir}: Missing {required_episode_file}")
                continue

            clip_dirs = sorted(d for d in v_dir.iterdir() if d.is_dir() and d.name.startswith("clip_"))
            try:
                clip_dirs.sort(key=lambda x: int(x.name.split("_")[-1]))
            except ValueError:
                continue
            if not clip_dirs:
                continue

            stage_tag = "stage1" if self.stage1_only else "stage2"
            required_clip_files = (
                ["latents.pt", "text_embeds.pt", "text_embeds_null.pt"]
                if self.stage1_only
                else ["latents.pt", "visual_embeds.pt", "text_ids.pt", "text_embeds.pt", "text_embeds_null.pt"]
            )

            valid_clips = []
            for d in clip_dirs:
                if all((d / f).exists() for f in required_clip_files):
                    valid_clips.append(d)
                else:
                    print(f"Skipping clip {d.name} in {v_dir}: Missing required files for {stage_tag}")
                    break

            clip_names = [d.name for d in valid_clips]
            for i in range(len(valid_clips)):
                self.samples.append({"video_dir": v_dir, "curr_idx": i, "clip_names": clip_names})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_dir = sample["video_dir"]
        curr_idx = sample["curr_idx"]
        clip_names = sample["clip_names"]

        curr_clip_path = video_dir / clip_names[curr_idx]

        curr_latents = torch.load(curr_clip_path / "latents.pt", map_location="cpu")
        if curr_latents.dim() == 5:
            curr_latents = curr_latents.squeeze(0)
        _, T_lat, H_lat_raw, W_lat_raw = curr_latents.shape
        H_lat, W_lat = self._aligned_latent_spatial_size(H_lat_raw, W_lat_raw)
        curr_latents = self._crop_latents_to_size(curr_latents, H_lat, W_lat)

        if curr_idx < len(clip_names) - 1:
            next_latents = torch.load(video_dir / clip_names[curr_idx + 1] / "latents.pt", map_location="cpu")
            if next_latents.dim() == 5:
                next_latents = next_latents.squeeze(0)
            next_latents = self._crop_latents_to_size(next_latents, H_lat, W_lat)
        else:
            next_latents = curr_latents.clone()

        if curr_idx == 0:
            cond_latents = self._load_first_frame_latents(video_dir, num_frames=1)
            cond_latents = self._crop_latents_to_size(cond_latents, H_lat, W_lat)
        else:
            latents_prev = torch.load(video_dir / clip_names[curr_idx - 1] / "latents.pt", map_location="cpu")
            if latents_prev.dim() == 5:
                latents_prev = latents_prev.squeeze(0)
            latents_prev = self._crop_latents_to_size(latents_prev, H_lat, W_lat)
            cond_latents = latents_prev[:, -1:, :, :]

        if self.use_sink:
            sink_latents = self._load_first_frame_latents(video_dir, num_frames=self.num_sink_frames)
            sink_latents = self._crop_latents_to_size(sink_latents, H_lat, W_lat)
            target_latents = torch.cat([sink_latents, cond_latents, curr_latents, next_latents], dim=1)
        else:
            target_latents = torch.cat([cond_latents, curr_latents, next_latents], dim=1)

        def _load_mask(clip_path):
            mask_file = clip_path / "mask.npz"
            if mask_file.exists():
                raw = np.load(str(mask_file))["mask"].astype(np.float32)
                world_mask = torch.from_numpy(raw)
                world_mask = resize_soft_mask(world_mask, (T_lat, H_lat_raw, W_lat_raw))
                return self._crop_mask_to_size(world_mask, H_lat, W_lat)
            return torch.ones(T_lat, H_lat, W_lat)

        curr_mask = _load_mask(curr_clip_path)
        next_mask = _load_mask(video_dir / clip_names[curr_idx + 1]) if curr_idx < len(clip_names) - 1 else curr_mask
        cond_mask = torch.ones(cond_latents.shape[1], H_lat, W_lat)
        if self.use_sink:
            sink_mask = torch.ones(self.num_sink_frames, H_lat, W_lat)
            world_mask = torch.cat([sink_mask, cond_mask, curr_mask, next_mask], dim=0)
        else:
            world_mask = torch.cat([cond_mask, curr_mask, next_mask], dim=0)

        curr_context = torch.load(curr_clip_path / "text_embeds.pt", map_location="cpu")
        if curr_context.dim() == 3:
            curr_context = curr_context.squeeze(0)

        null_context = torch.load(curr_clip_path / "text_embeds_null.pt", map_location="cpu")
        if null_context.dim() == 3:
            null_context = null_context.squeeze(0)

        if curr_idx < len(clip_names) - 1:
            next_context = torch.load(video_dir / clip_names[curr_idx + 1] / "text_embeds.pt", map_location="cpu")
            if next_context.dim() == 3:
                next_context = next_context.squeeze(0)
        else:
            next_context = curr_context

        if self.stage1_only:
            return {
                "latents": target_latents,
                "curr_context": curr_context,
                "next_context": next_context,
                "null_context": null_context,
                "video_dir": str(video_dir),
                "clip_idx": curr_idx,
                "is_last": curr_idx == len(clip_names) - 1,
            }

        curr_text_ids = torch.load(curr_clip_path / "text_ids.pt", map_location="cpu")
        if curr_text_ids.dim() == 2:
            curr_text_ids = curr_text_ids.squeeze(0)

        ff_embeds = torch.load(video_dir / "first_frame_embeds.pt", map_location="cpu")
        if ff_embeds.dim() == 3:
            ff_embeds = ff_embeds.squeeze(0)

        input_ids_list = []
        token_turn_ids_list = []
        visual_embeds_list = []

        start_idx = max(0, curr_idx - self.max_context_length)
        input_ids_list.append(torch.tensor([
            self.im_start_token_id, self.user_token_id, self.newline_token_id
        ], dtype=torch.long))
        token_turn_ids_list.append(torch.full((3,), start_idx, dtype=torch.long))

        num_ff = ff_embeds.shape[0]
        input_ids_list.append(torch.tensor([self.first_frame_token_id], dtype=torch.long))
        token_turn_ids_list.append(torch.full((1,), start_idx, dtype=torch.long))
        input_ids_list.append(torch.full((num_ff,), self.video_token_id, dtype=torch.long))
        token_turn_ids_list.append(torch.full((num_ff,), start_idx, dtype=torch.long))
        input_ids_list.append(torch.tensor([self.end_frame_token_id], dtype=torch.long))
        token_turn_ids_list.append(torch.full((1,), start_idx, dtype=torch.long))
        visual_embeds_list.append(ff_embeds)

        for i in range(start_idx, curr_idx):
            c_path = video_dir / clip_names[i]
            c_text = torch.load(c_path / "text_ids.pt", map_location="cpu")
            if c_text.dim() == 2:
                c_text = c_text.squeeze(0)
            c_vis = torch.load(c_path / "visual_embeds.pt", map_location="cpu")
            if c_vis.dim() == 3:
                c_vis = c_vis.squeeze(0)

            input_ids_list.append(c_text)
            token_turn_ids_list.append(torch.full((c_text.shape[0],), i, dtype=torch.long))
            num_tokens = c_vis.shape[0]
            input_ids_list.append(torch.tensor([self.first_frame_token_id], dtype=torch.long))
            token_turn_ids_list.append(torch.full((1,), i + 1, dtype=torch.long))
            input_ids_list.append(torch.full((num_tokens,), self.video_token_id, dtype=torch.long))
            token_turn_ids_list.append(torch.full((num_tokens,), i + 1, dtype=torch.long))
            input_ids_list.append(torch.tensor([self.end_frame_token_id], dtype=torch.long))
            token_turn_ids_list.append(torch.full((1,), i + 1, dtype=torch.long))
            visual_embeds_list.append(c_vis)

        input_ids_list.append(curr_text_ids)
        token_turn_ids_list.append(torch.full((curr_text_ids.shape[0],), curr_idx, dtype=torch.long))
        input_ids_list.append(torch.tensor([self.im_end_token_id, self.newline_token_id], dtype=torch.long))
        token_turn_ids_list.append(torch.full((2,), curr_idx, dtype=torch.long))
        input_ids_list.append(torch.tensor([
            self.im_start_token_id, self.assistant_token_id, self.newline_token_id
        ], dtype=torch.long))
        token_turn_ids_list.append(torch.full((3,), curr_idx, dtype=torch.long))

        input_ids = torch.cat(input_ids_list, dim=0)
        token_turn_ids = torch.cat(token_turn_ids_list, dim=0)
        visual_embeds = torch.cat(visual_embeds_list, dim=0)

        return {
            "input_ids": input_ids,
            "token_turn_ids": token_turn_ids,
            "attention_mask": torch.ones_like(input_ids),
            "visual_embeds": visual_embeds,
            "latents": target_latents,
            "curr_context": curr_context,
            "next_context": next_context,
            "null_context": null_context,
            "world_mask": world_mask,
            "video_dir": str(video_dir),
            "clip_idx": curr_idx,
            "is_last": curr_idx == len(clip_names) - 1,
        }


class RobotwinDataset(B1KDataset):
    def __init__(self, *args, task_prefix: str = "task-", **kwargs):
        self.task_prefix = task_prefix
        kwargs.setdefault("latent_spatial_align", "crop")
        kwargs.setdefault("latent_spatial_multiple", 2)
        super().__init__(*args, **kwargs)

    def _collect_episode_dirs(self):
        task_dirs = sorted(
            d for d in self.data_root.iterdir()
            if d.is_dir() and d.name.startswith(self.task_prefix)
        )
        collected = []
        for tdir in task_dirs:
            collected.extend(d for d in tdir.iterdir() if d.is_dir() and d.name.startswith("episode_"))
        return sorted(collected)


def pad_text_batch(text_list):
    B = len(text_list)
    lens = torch.tensor([t.shape[0] for t in text_list], dtype=torch.long)
    L_max = lens.max().item()
    D = text_list[0].shape[1]
    padded = text_list[0].new_zeros((B, L_max, D))
    for i, t in enumerate(text_list):
        padded[i, : t.shape[0]] = t
    return padded, lens


def pad_sequence_batch(sequences, padding_value=0):
    B = len(sequences)
    max_len = max(s.shape[0] for s in sequences)
    if sequences[0].dim() == 1:
        padded = sequences[0].new_full((B, max_len), padding_value)
        for i, s in enumerate(sequences):
            padded[i, : s.shape[0]] = s
    elif sequences[0].dim() == 2:
        D = sequences[0].shape[1]
        padded = sequences[0].new_full((B, max_len, D), padding_value)
        for i, s in enumerate(sequences):
            padded[i, : s.shape[0]] = s
    else:
        raise ValueError("Only 1D or 2D sequences supported")
    return padded


def wan_collate_fn(batch):
    return {
        "latents": torch.stack([b["latents"] for b in batch]),
        "context": [b["context"] for b in batch],
    }


def b1k_collate_fn(batch, stage: int = 2):
    if stage == 1:
        return {
            "latents": torch.stack([b["latents"] for b in batch]),
            "context_curr": [b["curr_context"] for b in batch],
            "context_next": [b["next_context"] for b in batch],
            "context_null": [b["null_context"] for b in batch],
            "video_dir": [b.get("video_dir") for b in batch],
            "clip_idx": [b.get("clip_idx") for b in batch],
            "is_last": torch.tensor([b["is_last"] for b in batch], dtype=torch.bool),
        }

    return {
        "input_ids": pad_sequence_batch([b["input_ids"] for b in batch], padding_value=0),
        "token_turn_ids": pad_sequence_batch(
            [b.get("token_turn_ids", torch.zeros_like(b["input_ids"])) for b in batch]
        ),
        "attention_mask": pad_sequence_batch([b["attention_mask"] for b in batch], padding_value=0),
        "visual_embeds": pad_sequence_batch([b["visual_embeds"] for b in batch], padding_value=0),
        "latents": torch.stack([b["latents"] for b in batch]),
        "context_curr": [b["curr_context"] for b in batch],
        "context_next": [b["next_context"] for b in batch],
        "context_null": [b["null_context"] for b in batch],
        "world_masks": torch.stack([b["world_mask"] for b in batch]),
        "is_last": torch.tensor([b["is_last"] for b in batch], dtype=torch.bool),
        "video_dir": [b["video_dir"] for b in batch],
        "clip_idx": [b["clip_idx"] for b in batch],
    }
