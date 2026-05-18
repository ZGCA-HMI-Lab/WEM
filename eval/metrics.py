from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .dataset import SegmentedSample, VideoChunk
from .models import ModelRegistry, TrackFrame, box_iou, frame_diagonal, pairwise, uniform_sample


def metric_ok(value: float, **details) -> Dict:
    return {
        "status": "ok",
        "value": float(value),
        "details": details,
    }


def metric_unsupported(reason: str, **details) -> Dict:
    return {
        "status": "unsupported",
        "value": None,
        "details": {"reason": reason, **details},
    }


def ols_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def collect_phase_frames(chunks: Sequence[VideoChunk], phase: str) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for chunk in chunks:
        if chunk.phase == phase:
            frames.extend(chunk.frames)
    return frames


def ssim_score(first_gray: np.ndarray, last_gray: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(first_gray, last_gray, data_range=255))
    except Exception:
        if hasattr(cv2, "quality"):
            return float(cv2.quality.QualitySSIM_compute(first_gray, last_gray)[0][0])
        first = first_gray.astype(np.float32)
        last = last_gray.astype(np.float32)
        mse = float(np.mean((first - last) ** 2))
        return max(0.0, 1.0 - mse / (255.0**2))


def compute_icbc(sample: SegmentedSample, models: ModelRegistry, config: Dict) -> Dict:
    lpips_scale = float(config["metrics"].get("icbc_lpips_scale", 0.5))
    flow_scale_ratio = float(config["metrics"].get("icbc_flow_scale_ratio", 0.05))

    appearance_values = []
    motion_values = []
    continuity_scores = []
    trivial_copy_hits = 0

    for prev_chunk, next_chunk in pairwise(sample.chunks):
        if len(prev_chunk.frames) < 2 or len(next_chunk.frames) < 2:
            continue

        end_frame = prev_chunk.frames[-1]
        start_frame = next_chunk.frames[0]

        appearance = models.lpips(end_frame, start_frame)
        motion_gap = models.flow_boundary_gap(
            prev_chunk.frames[-2],
            prev_chunk.frames[-1],
            next_chunk.frames[0],
            next_chunk.frames[1],
        )
        motion_scale = max(flow_scale_ratio * frame_diagonal(end_frame), 1e-6)
        appearance_norm = appearance / (appearance + lpips_scale)
        motion_norm = motion_gap / (motion_gap + motion_scale)
        continuity = 1.0 - 0.5 * (appearance_norm + motion_norm)

        psnr = cv2.PSNR(cv2.cvtColor(end_frame, cv2.COLOR_RGB2BGR), cv2.cvtColor(start_frame, cv2.COLOR_RGB2BGR))
        if psnr > 35.0:
            trivial_copy_hits += 1

        appearance_values.append(float(appearance))
        motion_values.append(float(motion_gap))
        continuity_scores.append(float(continuity))

    if not continuity_scores:
        return metric_unsupported("insufficient_boundary_frames")

    return metric_ok(
        float(np.mean(continuity_scores)),
        appearance_mean=float(np.mean(appearance_values)),
        motion_gap_mean=float(np.mean(motion_values)),
        boundary_count=len(continuity_scores),
        copy_ratio=float(trivial_copy_hits / max(len(continuity_scores), 1)),
    )


def compute_crsd(sample: SegmentedSample, models: ModelRegistry, config: Dict) -> Dict:
    clip_max_frames = int(config["metrics"].get("clip_max_frames", 8))

    distances = []
    similarities = []
    for chunk in sample.chunks:
        if not chunk.frames:
            continue
        similarity = models.video_text_similarity(chunk.frames, chunk.prompt, clip_max_frames)
        similarities.append(float(similarity))
        distances.append(float(1.0 - similarity))

    if len(distances) < 2:
        return metric_unsupported("insufficient_chunks")

    slope = ols_slope(distances)
    return metric_ok(
        max(0.0, slope),
        raw_slope=float(slope),
        similarities=similarities,
        distances=distances,
    )


def compute_dstc(sample: SegmentedSample, models: ModelRegistry, config: Dict) -> Dict:
    reference_prompt = config["metrics"].get("neutral_reference_prompt", "a static scene with no motion")
    scores = []

    for chunk in sample.chunks:
        if len(chunk.frames) < 2:
            continue
        scores.append(
            models.directional_state_score(
                start_frame=chunk.frames[0],
                end_frame=chunk.frames[-1],
                prompt=chunk.prompt,
                reference_prompt=reference_prompt,
            )
        )

    if not scores:
        return metric_unsupported("insufficient_chunks")

    return metric_ok(
        float(np.mean(scores)),
        variance=float(np.var(scores)),
        per_chunk=scores,
    )


def compute_ccsp(task_id: str, samples: Sequence[SegmentedSample], models: ModelRegistry, config: Dict) -> Dict:
    if len(samples) < 2:
        return metric_unsupported("requires_at_least_two_samples", task_id=task_id)

    lpips_scale = float(config["metrics"].get("icbc_lpips_scale", 0.5))
    flow_scale_ratio = float(config["metrics"].get("icbc_flow_scale_ratio", 0.05))

    total = 0
    correct = 0
    per_task_scores = []

    for sample_idx, sample in enumerate(samples):
        for chunk_idx in range(len(sample.chunks) - 1):
            query_chunk = sample.chunks[chunk_idx]
            candidate_scores = []
            for candidate_sample in samples:
                candidate_chunk = candidate_sample.chunks[chunk_idx + 1]
                score = models.continuation_score(
                    query_chunk.frames,
                    candidate_chunk.frames,
                    lpips_scale=lpips_scale,
                    flow_scale_ratio=flow_scale_ratio,
                )
                if score is None:
                    score = float("-inf")
                candidate_scores.append(score)

            if all(score == float("-inf") for score in candidate_scores):
                continue

            predicted = int(np.argmax(candidate_scores))
            hit = int(predicted == sample_idx)
            total += 1
            correct += hit
            per_task_scores.append(hit)

    if total == 0:
        return metric_unsupported("insufficient_boundary_frames", task_id=task_id)

    raw_accuracy = correct / total
    num_candidates = len(samples)
    normalized = (raw_accuracy - 1.0 / num_candidates) / (1.0 - 1.0 / num_candidates)

    return metric_ok(
        float(normalized),
        raw_accuracy=float(raw_accuracy),
        num_candidates=num_candidates,
        decision_count=total,
        backend="continuation_proxy",
    )


def compute_crvqd(sample: SegmentedSample, models: ModelRegistry, config: Dict) -> Dict:
    musiq_max_frames = int(config["metrics"].get("musiq_max_frames", 16))
    qualities = []
    for chunk in sample.chunks:
        qualities.append(models.musiq_score(chunk.frames, musiq_max_frames))

    if len(qualities) < 2:
        return metric_unsupported("insufficient_chunks")

    slope = ols_slope(qualities)
    qdr = float(qualities[-1] / max(qualities[0], 1e-6))
    return metric_ok(
        float(-slope),
        raw_slope=float(slope),
        qdr=qdr,
        qualities=qualities,
    )


def compute_npc(sample: SegmentedSample, gt_nav_frames: Sequence[np.ndarray], models: ModelRegistry, config: Dict) -> Dict:
    gen_nav_frames = collect_phase_frames(sample.chunks, "Nav")
    if not gen_nav_frames:
        return metric_unsupported("task_has_no_nav_phase")
    if len(gt_nav_frames) < 2:
        return metric_unsupported("gt_has_no_nav_frames")

    max_pairs = int(config["metrics"].get("npc_max_pairs", 16))
    pred_traj = models.estimate_global_2d_trajectory(gen_nav_frames, max_pairs=max_pairs)
    gt_traj = models.estimate_global_2d_trajectory(gt_nav_frames, max_pairs=max_pairs)
    error = models.normalized_trajectory_error(pred_traj, gt_traj)
    score = 1.0 - min(1.0, error)

    return metric_ok(
        float(score),
        error=float(error),
        backend="proxy_flow_2d_trajectory",
        pred_points=int(len(pred_traj)),
        gt_points=int(len(gt_traj)),
    )


def _last_valid_center(track_frames: Sequence[TrackFrame], side: str) -> Optional[np.ndarray]:
    for frame in reversed(track_frames):
        center = frame.left_center if side == "left" else frame.right_center
        if center is not None:
            return center
    return None


def compute_mep(sample: SegmentedSample, gt_manip_frames: Sequence[np.ndarray], models: ModelRegistry, config: Dict) -> Dict:
    gen_manip_frames = collect_phase_frames(sample.chunks, "Manip")
    if not gen_manip_frames:
        return metric_unsupported("task_has_no_manip_phase")
    if not gt_manip_frames:
        return metric_unsupported("gt_has_no_manip_frames")

    max_frames = int(config["metrics"].get("mep_max_frames", 16))
    alpha = float(config["metrics"].get("mep_alpha", 0.6))

    gen_frames = uniform_sample(gen_manip_frames, max_frames)
    gt_frames = uniform_sample(gt_manip_frames, len(gen_frames))
    target_len = min(len(gen_frames), len(gt_frames))
    if target_len == 0:
        return metric_unsupported("insufficient_manip_frames")
    gen_frames = gen_frames[:target_len]
    gt_frames = gt_frames[:target_len]

    gen_tracks = models.detect_gripper_tracks(gen_frames)
    gt_tracks = models.detect_gripper_tracks(gt_frames)

    ious = []
    for gen_track, gt_track in zip(gen_tracks, gt_tracks):
        side_values = []
        if gen_track.left_box is not None and gt_track.left_box is not None:
            side_values.append(box_iou(gen_track.left_box, gt_track.left_box))
        if gen_track.right_box is not None and gt_track.right_box is not None:
            side_values.append(box_iou(gen_track.right_box, gt_track.right_box))
        if side_values:
            ious.append(float(np.mean(side_values)))

    endpoint_errors = []
    diag = frame_diagonal(gen_frames[-1])
    for side in ("left", "right"):
        gen_center = _last_valid_center(gen_tracks, side)
        gt_center = _last_valid_center(gt_tracks, side)
        if gen_center is None or gt_center is None:
            continue
        endpoint_errors.append(float(np.linalg.norm(gen_center - gt_center) / max(diag, 1e-6)))

    if not ious and not endpoint_errors:
        return metric_unsupported("sam3_failed_to_track_end_effector")

    iou_score = float(np.mean(ious)) if ious else 0.0
    contact_error = float(np.mean(endpoint_errors)) if endpoint_errors else 1.0
    value = alpha * iou_score + (1.0 - alpha) * (1.0 - min(1.0, contact_error))

    return metric_ok(
        float(value),
        iou_score=iou_score,
        endpoint_error=contact_error,
        backend="sam3_box_track_proxy",
    )


def compute_ptej(sample: SegmentedSample, models: ModelRegistry) -> Dict:
    transition_scores = []
    intra_scores = []

    for prev_chunk, next_chunk in pairwise(sample.chunks):
        if not prev_chunk.frames or not next_chunk.frames:
            continue

        score = models.lpips(prev_chunk.frames[-1], next_chunk.frames[0])
        if prev_chunk.phase == "Nav" and next_chunk.phase == "Manip":
            transition_scores.append(score)
        elif prev_chunk.phase == next_chunk.phase:
            intra_scores.append(score)

    if not transition_scores:
        return metric_unsupported("task_has_no_nav_to_manip_transition")
    if not intra_scores:
        return metric_unsupported("task_has_no_intra_phase_boundary")

    transition_mean = float(np.mean(transition_scores))
    intra_mean = float(np.mean(intra_scores))
    value = max(0.0, transition_mean / max(intra_mean, 1e-6) - 1.0)

    return metric_ok(
        float(value),
        transition_lpips=transition_mean,
        intra_lpips=intra_mean,
    )


def compute_psa(sample: SegmentedSample, models: ModelRegistry, config: Dict) -> Dict:
    tau_b = float(config["metrics"].get("psa_ssim_threshold", 0.92))
    tau_f = float(config["metrics"].get("psa_global_flow_threshold", 2.0))
    tau_l = float(config["metrics"].get("psa_local_flow_threshold", 2.0))
    max_pairs = int(config["metrics"].get("psa_max_pairs", 4))

    predicted = []
    lengths = []

    for chunk in sample.chunks:
        if len(chunk.frames) < 2:
            predicted.append("Unknown")
            lengths.append(max(chunk.gt_frame_count, 1))
            continue

        first = cv2.cvtColor(chunk.frames[0], cv2.COLOR_RGB2GRAY)
        last = cv2.cvtColor(chunk.frames[-1], cv2.COLOR_RGB2GRAY)
        bg_ssim = ssim_score(first, last)
        global_flow = models.mean_flow_magnitude(chunk.frames, max_pairs=max_pairs)
        local_flow = models.local_flow_magnitude(chunk.frames, max_pairs=max_pairs)

        if bg_ssim < tau_b and global_flow > tau_f:
            pred = "Nav"
        elif bg_ssim >= tau_b and local_flow > tau_l:
            pred = "Manip"
        else:
            pred = "Nav" if global_flow >= local_flow else "Manip"

        predicted.append(pred)
        lengths.append(max(chunk.gt_frame_count, len(chunk.frames), 1))

    def weighted_iou(label: str) -> float:
        inter = 0
        union = 0
        for pred, chunk, weight in zip(predicted, sample.chunks, lengths):
            gt = chunk.phase
            if pred == label and gt == label:
                inter += weight
            if pred == label or gt == label:
                union += weight
        if union == 0:
            return 1.0
        return inter / union

    nav_iou = weighted_iou("Nav")
    manip_iou = weighted_iou("Manip")
    return metric_ok(
        0.5 * (nav_iou + manip_iou),
        nav_iou=float(nav_iou),
        manip_iou=float(manip_iou),
        predicted_labels=predicted,
    )


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def symmetric_match(a_gen: float, a_gt: float, eps: float = 1e-6) -> float:
    return float(np.exp(-abs(np.log((float(a_gen) + eps) / (float(a_gt) + eps)))))


def compute_rds(
    sample: SegmentedSample,
    models: ModelRegistry,
    config: Dict,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict:
    clip_max_frames = int(config["metrics"].get("clip_max_frames", 8))
    musiq_max_frames = int(config["metrics"].get("musiq_max_frames", 16))
    semantic_weight = float(config["metrics"].get("rds_semantic_weight", 0.5))
    quality_weight = float(config["metrics"].get("rds_quality_weight", 0.5))
    weight_sum = max(semantic_weight + quality_weight, 1e-6)
    semantic_weight /= weight_sum
    quality_weight /= weight_sum

    semantic_raw = []
    semantic_scores = []
    quality_scores = []
    round_scores = []

    for chunk_idx, chunk in enumerate(sample.chunks, start=1):
        if not chunk.frames:
            continue
        if progress is not None:
            progress(f"RDS chunk {chunk_idx}/{len(sample.chunks)}: CLIP+MUSIQ scoring")
        semantic = models.video_text_similarity(chunk.frames, chunk.prompt, clip_max_frames)
        quality = models.musiq_score(chunk.frames, musiq_max_frames)
        semantic_norm = _clip01((float(semantic) + 1.0) * 0.5)
        quality_norm = _clip01(float(quality))

        semantic_raw.append(float(semantic))
        semantic_scores.append(semantic_norm)
        quality_scores.append(quality_norm)
        round_scores.append(semantic_weight * semantic_norm + quality_weight * quality_norm)

    if len(round_scores) < 2:
        return metric_unsupported("insufficient_chunks")

    score_slope = ols_slope(round_scores)
    semantic_slope = ols_slope(semantic_scores)
    quality_slope = ols_slope(quality_scores)
    return metric_ok(
        max(0.0, -score_slope),
        raw_slope=float(score_slope),
        semantic_raw=semantic_raw,
        semantic_scores=semantic_scores,
        quality_scores=quality_scores,
        round_scores=round_scores,
        semantic_degradation_slope=max(0.0, -semantic_slope),
        quality_degradation_slope=max(0.0, -quality_slope),
        semantic_slope=float(semantic_slope),
        quality_slope=float(quality_slope),
        semantic_weight=float(semantic_weight),
        quality_weight=float(quality_weight),
    )


def compute_dtc(sample: SegmentedSample, models: ModelRegistry, config: Dict) -> Dict:
    reference_prompt = config["metrics"].get("neutral_reference_prompt", "a static scene with no motion")
    window_frames = int(config["metrics"].get("dtc_window_frames", 4))
    max_frames = int(config["metrics"].get("dtc_max_frames", window_frames))
    scores = []
    per_chunk = []

    for chunk in sample.chunks:
        if len(chunk.frames) < 2:
            continue
        start_window = chunk.frames[: max(1, window_frames)]
        end_window = chunk.frames[-max(1, window_frames) :]
        score = models.directional_video_state_score(
            start_frames=start_window,
            end_frames=end_window,
            prompt=chunk.prompt,
            reference_prompt=reference_prompt,
            max_frames=max_frames,
        )
        scores.append(float(score))
        per_chunk.append(
            {
                "chunk_index": chunk.index,
                "phase": chunk.phase,
                "score": float(score),
            }
        )

    if not scores:
        return metric_unsupported("insufficient_chunks")

    return metric_ok(
        float(np.mean(scores)),
        variance=float(np.var(scores)),
        per_chunk=per_chunk,
    )


def compute_motion_stats_for_chunks(chunks: Sequence[VideoChunk], models: ModelRegistry, config: Dict) -> List[Dict]:
    max_pairs = int(config["metrics"].get("motion_max_pairs", 8))
    top_q_percent = float(config["metrics"].get("local_top_q_percent", 20.0))
    stats = []

    for chunk in chunks:
        motion = models.flow_motion_statistics(
            chunk.frames,
            max_pairs=max_pairs,
            top_q_percent=top_q_percent,
        )
        stats.append(
            {
                "chunk_index": int(chunk.index),
                "phase": chunk.phase,
                "broad": float(motion["broad"]),
                "local": float(motion["local"]),
                "pair_count": int(motion["pair_count"]),
            }
        )
    return stats


def compute_pcmf_from_stats(gen_stats: Sequence[Dict], gt_stats: Sequence[Dict], config: Dict) -> Dict:
    eps = float(config["metrics"].get("match_epsilon", 1e-6))
    nav_scores = []
    manip_scores = []
    skipped = 0

    for gen, gt in zip(gen_stats, gt_stats):
        if int(gt.get("pair_count", 0)) <= 0:
            skipped += 1
            continue
        phase = gen.get("phase")
        if phase == "Nav":
            nav_scores.append(symmetric_match(gen["broad"], gt["broad"], eps))
        elif phase == "Manip":
            manip_scores.append(symmetric_match(gen["local"], gt["local"], eps))

    nav_score = _mean_or_none(nav_scores)
    manip_score = _mean_or_none(manip_scores)
    if nav_score is not None and manip_score is not None:
        value = float(np.sqrt(nav_score * manip_score))
        phase_mode = "mixed"
    elif nav_score is not None:
        value = nav_score
        phase_mode = "nav_only"
    elif manip_score is not None:
        value = manip_score
        phase_mode = "manip_only"
    else:
        return metric_unsupported("no_valid_phase_motion_stats", skipped_chunks=skipped)

    return metric_ok(
        value,
        nav_score=nav_score,
        manip_score=manip_score,
        nav_count=len(nav_scores),
        manip_count=len(manip_scores),
        phase_mode=phase_mode,
        skipped_chunks=skipped,
        gen_motion_stats=list(gen_stats),
        gt_motion_stats=list(gt_stats),
    )


def compute_cpi_from_stats(gen_stats: Sequence[Dict], gt_stats: Sequence[Dict], config: Dict) -> Dict:
    eps = float(config["metrics"].get("match_epsilon", 1e-6))
    nav_excess = []
    manip_excess = []
    skipped = 0

    for gen, gt in zip(gen_stats, gt_stats):
        if int(gt.get("pair_count", 0)) <= 0:
            skipped += 1
            continue
        phase = gen.get("phase")
        if phase == "Nav":
            gen_ratio = (float(gen["local"]) + eps) / (float(gen["broad"]) + eps)
            gt_ratio = (float(gt["local"]) + eps) / (float(gt["broad"]) + eps)
            nav_excess.append(max(0.0, float(np.log((gen_ratio + eps) / (gt_ratio + eps)))))
        elif phase == "Manip":
            manip_excess.append(
                max(
                    0.0,
                    float(np.log((float(gen["broad"]) + eps) / (float(gt["broad"]) + eps))),
                )
            )

    nav_score = _mean_or_none(nav_excess)
    manip_score = _mean_or_none(manip_excess)
    available = [value for value in (nav_score, manip_score) if value is not None]
    if not available:
        return metric_unsupported("no_valid_phase_motion_stats", skipped_chunks=skipped)

    return metric_ok(
        float(np.mean(available)),
        nav_interference=nav_score,
        manip_interference=manip_score,
        nav_count=len(nav_excess),
        manip_count=len(manip_excess),
        skipped_chunks=skipped,
        gen_motion_stats=list(gen_stats),
        gt_motion_stats=list(gt_stats),
    )


def _boundary_window(prev_chunk: VideoChunk, next_chunk: VideoChunk, radius: int) -> List[np.ndarray]:
    radius = max(1, int(radius))
    frames = []
    if prev_chunk.frames:
        frames.extend(prev_chunk.frames[-radius:])
    if next_chunk.frames:
        frames.extend(next_chunk.frames[:radius])
    return frames


def compute_phsc(
    gen_chunks: Sequence[VideoChunk],
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    radius = int(config["metrics"].get("phsc_window_frames", 4))
    max_frames = int(config["metrics"].get("phsc_max_frames", max(2, radius * 2)))
    scores = []
    per_boundary = []
    switch_count = 0

    max_boundary = min(len(gen_chunks), len(gt_chunks)) - 1
    for idx in range(max_boundary):
        gen_prev, gen_next = gen_chunks[idx], gen_chunks[idx + 1]
        gt_prev, gt_next = gt_chunks[idx], gt_chunks[idx + 1]
        if gen_prev.phase == gen_next.phase:
            continue
        switch_count += 1

        gen_window = _boundary_window(gen_prev, gen_next, radius)
        gt_window = _boundary_window(gt_prev, gt_next, radius)
        if not gen_window or not gt_window:
            continue

        score = models.video_feature_similarity(gen_window, gt_window, max_frames=max_frames)
        scores.append(float(score))
        per_boundary.append(
            {
                "boundary_index": idx,
                "from_phase": gen_prev.phase,
                "to_phase": gen_next.phase,
                "score": float(score),
            }
        )

    if switch_count == 0:
        return metric_unsupported("task_has_no_phase_switch")
    if not scores:
        return metric_unsupported("no_valid_phase_switch_windows", switch_count=switch_count)

    return metric_ok(
        float(np.mean(scores)),
        switch_count=switch_count,
        evaluated_switch_count=len(scores),
        per_boundary=per_boundary,
    )


def _chunk_end_window(chunk: VideoChunk, window_frames: int) -> List[np.ndarray]:
    if not chunk.frames:
        return []
    return chunk.frames[-max(1, int(window_frames)) :]


def _encode_chunk_features(
    chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    max_frames: int,
) -> List[Optional[np.ndarray]]:
    features: List[Optional[np.ndarray]] = []
    for chunk in chunks:
        if not chunk.frames:
            features.append(None)
            continue
        feat = models.encode_video(chunk.frames, max_frames)
        features.append(feat.detach().cpu().numpy().astype(np.float32))
    return features


def _cosine_from_features(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> Optional[float]:
    if first is None or second is None:
        return None
    return float(np.dot(first, second))


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def compute_rcbd(
    sample: SegmentedSample,
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    eps = float(config["metrics"].get("match_epsilon", 1e-6))
    boundary_scores = []
    per_boundary = []

    max_boundary = min(len(sample.chunks), len(gt_chunks)) - 1
    for idx in range(max_boundary):
        gen_prev, gen_next = sample.chunks[idx], sample.chunks[idx + 1]
        gt_prev, gt_next = gt_chunks[idx], gt_chunks[idx + 1]
        if min(len(gen_prev.frames), len(gen_next.frames), len(gt_prev.frames), len(gt_next.frames)) < 2:
            continue

        gen_app = models.lpips(gen_prev.frames[-1], gen_next.frames[0])
        gt_app = models.lpips(gt_prev.frames[-1], gt_next.frames[0])
        gen_motion = models.flow_boundary_gap(
            gen_prev.frames[-2],
            gen_prev.frames[-1],
            gen_next.frames[0],
            gen_next.frames[1],
        )
        gt_motion = models.flow_boundary_gap(
            gt_prev.frames[-2],
            gt_prev.frames[-1],
            gt_next.frames[0],
            gt_next.frames[1],
        )

        app_score = symmetric_match(gen_app, gt_app, eps)
        motion_score = symmetric_match(gen_motion, gt_motion, eps)
        score = float(np.sqrt(app_score * motion_score))
        boundary_scores.append(score)
        per_boundary.append(
            {
                "boundary_index": idx,
                "appearance_gen": float(gen_app),
                "appearance_gt": float(gt_app),
                "motion_gen": float(gen_motion),
                "motion_gt": float(gt_motion),
                "appearance_score": float(app_score),
                "motion_score": float(motion_score),
                "score": score,
            }
        )

    if not boundary_scores:
        return metric_unsupported("insufficient_boundary_frames")

    return metric_ok(
        float(np.mean(boundary_scores)),
        boundary_count=len(boundary_scores),
        per_boundary=per_boundary,
    )


def compute_lpsa(
    sample: SegmentedSample,
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    window_frames = int(config["metrics"].get("lpsa_window_frames", 4))
    max_frames = int(config["metrics"].get("lpsa_max_frames", window_frames))
    scores = []
    weights = []
    per_chunk = []

    for idx, (gen_chunk, gt_chunk) in enumerate(zip(sample.chunks, gt_chunks), start=1):
        gen_window = _chunk_end_window(gen_chunk, window_frames)
        gt_window = _chunk_end_window(gt_chunk, window_frames)
        if not gen_window or not gt_window:
            continue
        score = models.video_feature_similarity(gen_window, gt_window, max_frames=max_frames)
        weight = float(idx)
        scores.append(float(score))
        weights.append(weight)
        per_chunk.append(
            {
                "chunk_index": gen_chunk.index,
                "phase": gen_chunk.phase,
                "weight": weight,
                "score": float(score),
            }
        )

    if not scores:
        return metric_unsupported("insufficient_chunk_windows")

    weights_arr = np.asarray(weights, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    value = float(np.sum(weights_arr * scores_arr) / max(float(weights_arr.sum()), 1e-6))
    return metric_ok(
        value,
        chunk_count=len(scores),
        per_chunk=per_chunk,
    )


def compute_cisr(
    sample: SegmentedSample,
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    max_frames = int(config["metrics"].get("cisr_max_frames", 8))
    gen_features = _encode_chunk_features(sample.chunks, models, max_frames)
    gt_features = _encode_chunk_features(gt_chunks, models, max_frames)

    reciprocal_ranks = []
    top1_hits = []
    per_chunk = []
    for idx, gen_feat in enumerate(gen_features):
        if gen_feat is None or idx >= len(gt_features) or gt_features[idx] is None:
            continue
        sims = []
        for gt_feat in gt_features:
            sims.append(_cosine_from_features(gen_feat, gt_feat) if gt_feat is not None else float("-inf"))
        correct_sim = sims[idx]
        rank = 1 + sum(1 for sim in sims if sim > correct_sim)
        reciprocal = 1.0 / float(rank)
        top1 = int(rank == 1)
        reciprocal_ranks.append(reciprocal)
        top1_hits.append(top1)
        per_chunk.append(
            {
                "chunk_index": sample.chunks[idx].index,
                "phase": sample.chunks[idx].phase,
                "rank": int(rank),
                "reciprocal_rank": float(reciprocal),
                "top1": top1,
                "correct_similarity": float(correct_sim),
                "best_similarity": float(max(sims)),
            }
        )

    if not reciprocal_ranks:
        return metric_unsupported("insufficient_chunk_features")

    return metric_ok(
        float(np.mean(reciprocal_ranks)),
        top1_accuracy=float(np.mean(top1_hits)),
        chunk_count=len(reciprocal_ranks),
        per_chunk=per_chunk,
    )


def _motion_entropy(magnitude: np.ndarray, bins: int) -> float:
    flat = magnitude.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return 0.0
    max_value = float(flat.max())
    if max_value <= 1e-6:
        return 0.0
    hist, _ = np.histogram(flat, bins=max(2, bins), range=(0.0, max_value))
    prob = hist.astype(np.float64)
    total = float(prob.sum())
    if total <= 0.0:
        return 0.0
    prob = prob[prob > 0] / total
    return float(-np.sum(prob * np.log(prob)) / np.log(max(2, bins)))


def _motion_profile(
    frames: Sequence[np.ndarray],
    models: ModelRegistry,
    config: Dict,
) -> np.ndarray:
    max_pairs = int(config["metrics"].get("pmpa_max_pairs", 16))
    top_q_percent = float(config["metrics"].get("local_top_q_percent", 20.0))
    entropy_bins = int(config["metrics"].get("pmpa_entropy_bins", 16))
    sampled = uniform_sample(list(frames), max_pairs + 1)
    values = []

    top_q_percent = float(np.clip(top_q_percent, 0.0, 100.0))
    for first, second in pairwise(sampled):
        flow = models.flow_map(first, second)
        magnitude = np.linalg.norm(flow, axis=2).astype(np.float32)
        flat = magnitude.reshape(-1)
        if flat.size == 0:
            continue

        median_value = float(np.median(flat))
        top_count = max(1, min(flat.size, int(np.ceil(flat.size * max(top_q_percent, 1e-6) / 100.0))))
        top_value = float(np.partition(flat, -top_count)[-top_count:].mean())
        diag = max(frame_diagonal(first), 1e-6)
        ratio = float(np.log1p(top_value / max(median_value, 1e-6)))
        entropy = _motion_entropy(magnitude, entropy_bins)
        values.append([median_value / diag, top_value / diag, ratio, entropy])

    if not values:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _resample_profile(profile: np.ndarray, target_len: int) -> np.ndarray:
    target_len = max(1, int(target_len))
    if profile.shape[0] == 0:
        return np.zeros((target_len, 4), dtype=np.float32)
    if profile.shape[0] == target_len:
        return profile.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, num=profile.shape[0])
    new_x = np.linspace(0.0, 1.0, num=target_len)
    columns = [np.interp(new_x, old_x, profile[:, col]) for col in range(profile.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def compute_pmpa(
    sample: SegmentedSample,
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    profile_len = int(config["metrics"].get("pmpa_profile_len", 16))
    temperature = float(config["metrics"].get("pmpa_temperature", 1.0))
    scores = []
    nav_scores = []
    manip_scores = []
    per_chunk = []

    for gen_chunk, gt_chunk in zip(sample.chunks, gt_chunks):
        gen_profile = _motion_profile(gen_chunk.frames, models, config)
        gt_profile = _motion_profile(gt_chunk.frames, models, config)
        if gen_profile.shape[0] == 0 or gt_profile.shape[0] == 0:
            continue
        gen_profile = _resample_profile(gen_profile, profile_len)
        gt_profile = _resample_profile(gt_profile, profile_len)
        distance = float(np.mean(np.linalg.norm(gen_profile - gt_profile, axis=1)))
        score = float(np.exp(-distance / max(temperature, 1e-6)))
        scores.append(score)
        if gen_chunk.phase == "Nav":
            nav_scores.append(score)
        elif gen_chunk.phase == "Manip":
            manip_scores.append(score)
        per_chunk.append(
            {
                "chunk_index": gen_chunk.index,
                "phase": gen_chunk.phase,
                "distance": distance,
                "score": score,
            }
        )

    if not scores:
        return metric_unsupported("insufficient_motion_profiles")

    return metric_ok(
        float(np.mean(scores)),
        nav_score=_mean_or_none(nav_scores),
        manip_score=_mean_or_none(manip_scores),
        chunk_count=len(scores),
        per_chunk=per_chunk,
    )


def compute_cpdm(
    sample: SegmentedSample,
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    max_frames = int(config["metrics"].get("cpdm_max_frames", 8))
    temperature = float(config["metrics"].get("cpdm_temperature", 0.05))
    gen_features = _encode_chunk_features(sample.chunks, models, max_frames)
    gt_features = _encode_chunk_features(gt_chunks, models, max_frames)

    scores = []
    margins = []
    per_chunk = []
    for idx, gen_feat in enumerate(gen_features):
        if gen_feat is None or idx >= len(gt_features) or gt_features[idx] is None:
            continue
        phase = sample.chunks[idx].phase
        opposite = []
        for gt_idx, gt_chunk in enumerate(gt_chunks):
            if gt_idx >= len(gt_features) or gt_features[gt_idx] is None:
                continue
            if gt_chunk.phase != phase:
                opposite.append((gt_idx, _cosine_from_features(gen_feat, gt_features[gt_idx])))
        if not opposite:
            continue

        pos = _cosine_from_features(gen_feat, gt_features[idx])
        neg_idx, neg = max(opposite, key=lambda item: item[1])
        margin = float(pos - neg)
        score = _sigmoid(margin / max(temperature, 1e-6))
        scores.append(score)
        margins.append(margin)
        per_chunk.append(
            {
                "chunk_index": sample.chunks[idx].index,
                "phase": phase,
                "positive_similarity": float(pos),
                "hard_negative_index": int(neg_idx),
                "hard_negative_phase": gt_chunks[neg_idx].phase,
                "hard_negative_similarity": float(neg),
                "margin": margin,
                "score": score,
            }
        )

    if not scores:
        return metric_unsupported("task_has_no_opposite_phase_candidates")

    return metric_ok(
        float(np.mean(scores)),
        mean_margin=float(np.mean(margins)),
        chunk_count=len(scores),
        per_chunk=per_chunk,
    )


def _change_bbox_from_gt_window(
    gt_window: Sequence[np.ndarray],
    models: ModelRegistry,
    top_q_percent: float,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    if not gt_window:
        return (0, 0, 1, 1)
    height, width = gt_window[0].shape[:2]
    if len(gt_window) < 2:
        return (0, 0, width, height)

    accum = np.zeros((height, width), dtype=np.float32)
    for first, second in pairwise(gt_window):
        flow = models.flow_map(first, second)
        magnitude = np.linalg.norm(flow, axis=2).astype(np.float32)
        if magnitude.shape != accum.shape:
            magnitude = cv2.resize(magnitude, (width, height), interpolation=cv2.INTER_LINEAR)
        accum = np.maximum(accum, magnitude)

    if float(accum.max()) <= 1e-6:
        return (0, 0, width, height)

    q = float(np.clip(top_q_percent, 0.0, 100.0))
    threshold = float(np.quantile(accum, max(0.0, 1.0 - q / 100.0)))
    mask = accum >= threshold
    if not np.any(mask):
        return (0, 0, width, height)

    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(round((x2 - x1) * padding_ratio))
    pad_y = int(round((y2 - y1) * padding_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, width, height)
    return (x1, y1, x2, y2)


def _crop_to_scaled_bbox(frame: np.ndarray, bbox: Tuple[int, int, int, int], base_shape: Tuple[int, int]) -> np.ndarray:
    base_h, base_w = base_shape
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    sx = width / max(float(base_w), 1.0)
    sy = height / max(float(base_h), 1.0)
    cx1 = max(0, min(width - 1, int(round(x1 * sx))))
    cy1 = max(0, min(height - 1, int(round(y1 * sy))))
    cx2 = max(cx1 + 1, min(width, int(round(x2 * sx))))
    cy2 = max(cy1 + 1, min(height, int(round(y2 * sy))))
    return frame[cy1:cy2, cx1:cx2]


def compute_fphsc(
    gen_chunks: Sequence[VideoChunk],
    gt_chunks: Sequence[VideoChunk],
    models: ModelRegistry,
    config: Dict,
) -> Dict:
    radius = int(config["metrics"].get("fphsc_window_frames", 4))
    max_frames = int(config["metrics"].get("fphsc_max_frames", max(2, radius * 2)))
    top_q = float(config["metrics"].get("fphsc_change_top_q_percent", 20.0))
    padding_ratio = float(config["metrics"].get("fphsc_crop_padding_ratio", 0.10))
    scores = []
    per_boundary = []
    switch_count = 0

    max_boundary = min(len(gen_chunks), len(gt_chunks)) - 1
    for idx in range(max_boundary):
        gen_prev, gen_next = gen_chunks[idx], gen_chunks[idx + 1]
        gt_prev, gt_next = gt_chunks[idx], gt_chunks[idx + 1]
        if gen_prev.phase == gen_next.phase:
            continue
        switch_count += 1

        gen_window = _boundary_window(gen_prev, gen_next, radius)
        gt_window = _boundary_window(gt_prev, gt_next, radius)
        if not gen_window or not gt_window:
            continue

        bbox = _change_bbox_from_gt_window(gt_window, models, top_q, padding_ratio)
        base_shape = gt_window[0].shape[:2]
        gen_crops = [_crop_to_scaled_bbox(frame, bbox, base_shape) for frame in gen_window]
        gt_crops = [_crop_to_scaled_bbox(frame, bbox, base_shape) for frame in gt_window]
        score = models.video_feature_similarity(gen_crops, gt_crops, max_frames=max_frames)
        scores.append(float(score))
        per_boundary.append(
            {
                "boundary_index": idx,
                "from_phase": gen_prev.phase,
                "to_phase": gen_next.phase,
                "bbox_xyxy": [int(v) for v in bbox],
                "score": float(score),
            }
        )

    if switch_count == 0:
        return metric_unsupported("task_has_no_phase_switch")
    if not scores:
        return metric_unsupported("no_valid_phase_switch_windows", switch_count=switch_count)

    return metric_ok(
        float(np.mean(scores)),
        switch_count=switch_count,
        evaluated_switch_count=len(scores),
        per_boundary=per_boundary,
    )
