from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_QUALITY_ROOT = PROJECT_ROOT / "video_quality"
if str(VIDEO_QUALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_QUALITY_ROOT))


@dataclass
class TrackFrame:
    left_box: Optional[np.ndarray]
    right_box: Optional[np.ndarray]
    left_center: Optional[np.ndarray]
    right_center: Optional[np.ndarray]


def frame_diagonal(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    return float(np.hypot(width, height))


def uniform_sample(items: Sequence, limit: int) -> List:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=limit).astype(int).tolist()
    return [items[index] for index in indices]


def pairwise(items: Sequence):
    for idx in range(len(items) - 1):
        yield items[idx], items[idx + 1]


def interpolate_polyline(points: np.ndarray, target_len: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if len(points) == target_len:
        return points.astype(np.float32)
    if len(points) == 1:
        return np.repeat(points.astype(np.float32), target_len, axis=0)

    old_t = np.linspace(0.0, 1.0, num=len(points))
    new_t = np.linspace(0.0, 1.0, num=target_len)
    x = np.interp(new_t, old_t, points[:, 0])
    y = np.interp(new_t, old_t, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


class AttributeDict(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value):
        self[key] = value


class ModelRegistry:
    def __init__(self, config: Dict):
        device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.config = config

    @cached_property
    def lpips_metric(self):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        alexnet_path_value = self.config.get("checkpoints", {}).get("alexnet_model", "")
        if alexnet_path_value:
            alexnet_path = Path(alexnet_path_value).expanduser()
            if not alexnet_path.is_file():
                raise FileNotFoundError(f"Configured AlexNet checkpoint does not exist: {alexnet_path}")
            if alexnet_path.parent.name != "checkpoints":
                raise ValueError(
                    "alexnet_model must point to a file under a torch hub checkpoints directory, "
                    f"got: {alexnet_path}"
                )
            torch.hub.set_dir(str(alexnet_path.parent.parent))

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)
        metric.eval()
        return metric

    @cached_property
    def clip_bundle(self):
        import clip

        model_path = self.config["checkpoints"].get("clip_model") or "ViT-B/32"
        model, preprocess = clip.load(model_path, device=str(self.device))
        model.eval()
        return clip, model, preprocess

    @cached_property
    def raft_bundle(self):
        raft_path = self.config["checkpoints"].get("raft_model", "")
        if not raft_path:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=True).to(self.device)
            model.eval()
            return "torchvision", model, weights.transforms()

        from WorldArena.third_party.RAFT.core.raft import RAFT
        from WorldArena.third_party.RAFT.core.utils_core.utils import InputPadder

        args = AttributeDict(
            model=raft_path,
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0.0,
        )
        model = RAFT(args)
        ckpt = torch.load(args.model, map_location="cpu")
        state = {key.replace("module.", ""): value for key, value in ckpt.items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return "worldarena", model, InputPadder

    @cached_property
    def musiq_model(self):
        musiq_path = self.config["checkpoints"].get("musiq_model", "")
        if not musiq_path:
            import pyiqa

            metric = pyiqa.create_metric("musiq", device=self.device, as_loss=False)
            if hasattr(metric, "eval"):
                metric.eval()
            return metric

        from pyiqa.archs.musiq_arch import MUSIQ

        model = MUSIQ(pretrained_model_path=musiq_path)
        model.to(self.device)
        model.eval()
        return model

    @cached_property
    def sam3_detector(self):
        from processing.detection_tracking import GripperDetector

        return GripperDetector(model_path=self.config["checkpoints"]["sam3_model"])

    def _frame_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def lpips(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        with torch.no_grad():
            value = self.lpips_metric(self._frame_tensor(frame_a), self._frame_tensor(frame_b))
        return float(value.detach().cpu().item())

    def flow_map(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        backend, model, aux = self.raft_bundle
        image1 = torch.from_numpy(frame_a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        image2 = torch.from_numpy(frame_b).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if backend == "torchvision":
                transforms = aux
                image1, image2 = transforms(image1, image2)
                height, width = image1.shape[-2:]
                pad_h = (8 - height % 8) % 8
                pad_w = (8 - width % 8) % 8
                if pad_h or pad_w:
                    image1 = F.pad(image1, (0, pad_w, 0, pad_h), mode="replicate")
                    image2 = F.pad(image2, (0, pad_w, 0, pad_h), mode="replicate")
                flow_up = model(image1, image2)[-1]
                flow = flow_up[0, :, :height, :width].permute(1, 2, 0).detach().cpu().numpy()
            else:
                input_padder = aux
                image1 = image1.float()
                image2 = image2.float()
                padder = input_padder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow = padder.unpad(flow_up[0]).permute(1, 2, 0).detach().cpu().numpy()
        return flow

    def mean_flow_magnitude(self, frames: Sequence[np.ndarray], max_pairs: int) -> float:
        sampled = uniform_sample(list(frames), max_pairs + 1)
        magnitudes = []
        for first, second in pairwise(sampled):
            flow = self.flow_map(first, second)
            magnitudes.append(float(np.linalg.norm(flow, axis=2).mean()))
        if not magnitudes:
            return 0.0
        return float(np.mean(magnitudes))

    def local_flow_magnitude(self, frames: Sequence[np.ndarray], max_pairs: int) -> float:
        sampled = uniform_sample(list(frames), max_pairs + 1)
        tracks = self.detect_gripper_tracks(sampled)
        values = []
        for idx, (first, second) in enumerate(pairwise(sampled)):
            if idx >= len(tracks):
                break
            flow = self.flow_map(first, second)
            boxes = [tracks[idx].left_box, tracks[idx].right_box]
            for box in boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = box.astype(int).tolist()
                x1 = max(0, min(flow.shape[1] - 1, x1))
                y1 = max(0, min(flow.shape[0] - 1, y1))
                x2 = max(x1 + 1, min(flow.shape[1], x2))
                y2 = max(y1 + 1, min(flow.shape[0], y2))
                crop = flow[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                values.append(float(np.linalg.norm(crop, axis=2).mean()))
        if not values:
            return 0.0
        return float(np.mean(values))

    def flow_motion_statistics(self, frames: Sequence[np.ndarray], max_pairs: int, top_q_percent: float) -> Dict[str, float]:
        sampled = uniform_sample(list(frames), max_pairs + 1)
        broad_values = []
        local_values = []

        top_q_percent = float(np.clip(top_q_percent, 0.0, 100.0))
        for first, second in pairwise(sampled):
            flow = self.flow_map(first, second)
            magnitude = np.linalg.norm(flow, axis=2).astype(np.float32)
            broad_values.append(float(np.median(magnitude)))

            flat = magnitude.reshape(-1)
            if flat.size == 0:
                continue
            if top_q_percent <= 0.0:
                top_count = 1
            else:
                top_count = int(np.ceil(flat.size * top_q_percent / 100.0))
            top_count = max(1, min(flat.size, top_count))
            local_values.append(float(np.partition(flat, -top_count)[-top_count:].mean()))

        if not broad_values:
            return {"broad": 0.0, "local": 0.0, "pair_count": 0}

        return {
            "broad": float(np.mean(broad_values)),
            "local": float(np.mean(local_values)) if local_values else 0.0,
            "pair_count": len(broad_values),
        }

    def flow_boundary_gap(
        self,
        prev_prev: np.ndarray,
        prev_last: np.ndarray,
        next_first: np.ndarray,
        next_second: np.ndarray,
    ) -> float:
        flow_prev = self.flow_map(prev_prev, prev_last)
        flow_next = self.flow_map(next_first, next_second)
        return float(np.linalg.norm(flow_prev - flow_next, axis=2).mean())

    def continuation_score(
        self,
        prev_chunk_frames: Sequence[np.ndarray],
        next_chunk_frames: Sequence[np.ndarray],
        lpips_scale: float,
        flow_scale_ratio: float,
    ) -> Optional[float]:
        if len(prev_chunk_frames) < 2 or len(next_chunk_frames) < 2:
            return None
        appearance = self.lpips(prev_chunk_frames[-1], next_chunk_frames[0])
        motion_gap = self.flow_boundary_gap(
            prev_chunk_frames[-2],
            prev_chunk_frames[-1],
            next_chunk_frames[0],
            next_chunk_frames[1],
        )
        motion_scale = max(flow_scale_ratio * frame_diagonal(prev_chunk_frames[-1]), 1e-6)
        appearance_norm = appearance / (appearance + lpips_scale)
        motion_norm = motion_gap / (motion_gap + motion_scale)
        return float(1.0 - 0.5 * (appearance_norm + motion_norm))

    def _encode_images(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        _, model, preprocess = self.clip_bundle
        images = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(self.device)
        with torch.no_grad():
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_image(self, frame: np.ndarray) -> torch.Tensor:
        return self._encode_images([frame])[0]

    def encode_video(self, frames: Sequence[np.ndarray], max_frames: int) -> torch.Tensor:
        sampled = uniform_sample(list(frames), max_frames)
        if not sampled:
            raise ValueError("Cannot encode an empty frame list")
        features = self._encode_images(sampled)
        pooled = features.mean(dim=0)
        return pooled / pooled.norm(dim=-1, keepdim=True)

    def encode_text(self, text: str) -> torch.Tensor:
        clip_mod, model, _ = self.clip_bundle
        tokens = clip_mod.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0]

    def video_text_similarity(self, frames: Sequence[np.ndarray], text: str, max_frames: int) -> float:
        video_feat = self.encode_video(frames, max_frames)
        text_feat = self.encode_text(text)
        return float(torch.dot(video_feat, text_feat).detach().cpu().item())

    def directional_state_score(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        prompt: str,
        reference_prompt: str,
    ) -> float:
        start_feat = self.encode_image(start_frame)
        end_feat = self.encode_image(end_frame)
        prompt_feat = self.encode_text(prompt)
        ref_feat = self.encode_text(reference_prompt)

        delta_phi = end_feat - start_feat
        delta_psi = prompt_feat - ref_feat

        norm_phi = float(delta_phi.norm().detach().cpu().item())
        norm_psi = float(delta_psi.norm().detach().cpu().item())
        if norm_phi < 1e-6 or norm_psi < 1e-6:
            return 0.0

        score = torch.dot(delta_phi / delta_phi.norm(), delta_psi / delta_psi.norm())
        return float(score.detach().cpu().item())

    def directional_video_state_score(
        self,
        start_frames: Sequence[np.ndarray],
        end_frames: Sequence[np.ndarray],
        prompt: str,
        reference_prompt: str,
        max_frames: int,
    ) -> float:
        if not start_frames or not end_frames:
            return 0.0

        start_feat = self.encode_video(start_frames, max_frames)
        end_feat = self.encode_video(end_frames, max_frames)
        prompt_feat = self.encode_text(prompt)
        ref_feat = self.encode_text(reference_prompt)

        delta_phi = end_feat - start_feat
        delta_psi = prompt_feat - ref_feat

        norm_phi = float(delta_phi.norm().detach().cpu().item())
        norm_psi = float(delta_psi.norm().detach().cpu().item())
        if norm_phi < 1e-6 or norm_psi < 1e-6:
            return 0.0

        score = torch.dot(delta_phi / delta_phi.norm(), delta_psi / delta_psi.norm())
        return float(score.detach().cpu().item())

    def video_feature_similarity(
        self,
        frames_a: Sequence[np.ndarray],
        frames_b: Sequence[np.ndarray],
        max_frames: int,
    ) -> float:
        if not frames_a or not frames_b:
            return 0.0
        feat_a = self.encode_video(frames_a, max_frames)
        feat_b = self.encode_video(frames_b, max_frames)
        return float(torch.dot(feat_a, feat_b).detach().cpu().item())

    def musiq_score(self, frames: Sequence[np.ndarray], max_frames: int) -> float:
        sampled = uniform_sample(list(frames), max_frames)
        if not sampled:
            return 0.0

        scores = []
        with torch.no_grad():
            for frame in sampled:
                tensor = self._frame_tensor(frame)
                score = self.musiq_model(tensor)
                scores.append(float(score.detach().cpu().item()) / 100.0)
        return float(np.mean(scores)) if scores else 0.0

    def estimate_global_2d_trajectory(self, frames: Sequence[np.ndarray], max_pairs: int) -> np.ndarray:
        sampled = uniform_sample(list(frames), max_pairs + 1)
        if len(sampled) < 2:
            return np.zeros((1, 2), dtype=np.float32)

        positions = [np.zeros(2, dtype=np.float32)]
        for first, second in pairwise(sampled):
            flow = self.flow_map(first, second)
            magnitude = np.linalg.norm(flow, axis=2)
            threshold = np.quantile(magnitude, 0.8)
            mask = magnitude <= threshold
            if mask.sum() == 0:
                mask = np.ones_like(magnitude, dtype=bool)
            dx = float(np.median(flow[..., 0][mask]))
            dy = float(np.median(flow[..., 1][mask]))
            positions.append(positions[-1] + np.array([dx, dy], dtype=np.float32))
        return np.stack(positions, axis=0).astype(np.float32)

    def normalized_trajectory_error(self, pred_points: np.ndarray, gt_points: np.ndarray) -> float:
        from scipy.linalg import orthogonal_procrustes

        target_len = max(len(pred_points), len(gt_points), 2)
        pred = interpolate_polyline(pred_points, target_len)
        gt = interpolate_polyline(gt_points, target_len)

        pred_center = pred - pred.mean(axis=0, keepdims=True)
        gt_center = gt - gt.mean(axis=0, keepdims=True)

        pred_norm = float(np.linalg.norm(pred_center))
        gt_norm = float(np.linalg.norm(gt_center))
        if pred_norm < 1e-6 or gt_norm < 1e-6:
            return 1.0

        rotation, _ = orthogonal_procrustes(pred_center, gt_center)
        pred_aligned = pred_center @ rotation * (gt_norm / pred_norm)
        rmse = float(np.sqrt(np.mean(np.sum((pred_aligned - gt_center) ** 2, axis=1))))

        gt_extent = float(np.max(np.linalg.norm(gt_center, axis=1)))
        if gt_extent < 1e-6:
            gt_extent = float(np.linalg.norm(gt.max(axis=0) - gt.min(axis=0)))
        if gt_extent < 1e-6:
            gt_extent = 1.0
        return rmse / gt_extent

    def detect_gripper_tracks(self, frames: Sequence[np.ndarray]) -> List[TrackFrame]:
        results: List[TrackFrame] = []
        for frame in frames:
            boxes, scores = self.sam3_detector.detect_frame(frame, prompts=["robot arm", "end effector", "gripper"])
            left_box = None
            right_box = None
            left_center = None
            right_center = None

            detections = []
            for box, score in zip(boxes, scores):
                if float(score) < 0.3:
                    continue
                box_arr = np.asarray(box, dtype=np.float32)
                center = np.array([(box_arr[0] + box_arr[2]) / 2.0, (box_arr[1] + box_arr[3]) / 2.0], dtype=np.float32)
                detections.append((box_arr, center, float(score)))

            left_detections = [det for det in detections if det[1][0] < frame.shape[1] / 2.0]
            right_detections = [det for det in detections if det[1][0] >= frame.shape[1] / 2.0]

            if left_detections:
                left_box, left_center, _ = max(left_detections, key=lambda item: item[2])
            if right_detections:
                right_box, right_center, _ = max(right_detections, key=lambda item: item[2])

            results.append(
                TrackFrame(
                    left_box=left_box,
                    right_box=right_box,
                    left_center=left_center,
                    right_center=right_center,
                )
            )
        return results


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return inter / union
