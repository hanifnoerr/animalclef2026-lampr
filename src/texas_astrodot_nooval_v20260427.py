
import argparse
from pathlib import Path

import cv2
import numpy as np

import texas_astrodot_2025reuse_v20260426 as base


VERSION = "texas_astrodot_nooval_v20260427"


def align_vertical_no_flip(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Canonicalize Texas crops vertically without head/tail flipping.

    Field photos for this species already have the head at the top. The base
    helper flipped images when lower mask bands looked wider, but that can turn
    correctly oriented belly-dot fingerprints upside down.
    """
    crop_rgb, crop_mask = base.crop_to_mask(rgb, mask, 0.08)
    angle = base.core.pca_angle_degrees(crop_mask)
    rotate_angle = 90.0 - angle
    if abs(rotate_angle) > 1.5:
        crop_rgb, crop_mask = base.core.rotate_bound(crop_rgb, crop_mask, rotate_angle)
        crop_rgb, crop_mask = base.crop_to_mask(crop_rgb, crop_mask, 0.04)
    h, _ = crop_mask.shape[:2]
    widths = []
    for yf in [0.18, 0.30, 0.70, 0.84]:
        y = int(np.clip(round(h * yf), 0, h - 1))
        xs = np.where(crop_mask[y] > 0)[0]
        widths.append(float(xs.max() - xs.min() + 1) if len(xs) else 0.0)
    top_width = max(widths[:2])
    bottom_width = max(widths[2:])
    return crop_rgb, crop_mask, {
        "pca_angle": float(angle),
        "rotate_angle": float(rotate_angle),
        "top_width": float(top_width),
        "bottom_width": float(bottom_width),
        "flipped_vertical": False,
        "orientation_rule": "head_already_top_no_flip",
    }


def texas_belly_template_no_oval(row: dict, current_cluster: str, args: argparse.Namespace) -> base.TexasDotItem:
    """Texas v2: use the full aligned SAM-clean crop, not an oval belly cut.

    The Texas photos are already mostly standardized ventral views. The older
    ellipse mask removed some real belly/peripheral dot evidence, especially on
    wide bellies and side-edge dot patterns. This version keeps the complete
    cleaned foreground mask after alignment and canvas resize.
    """
    rgb, mask, quality = base.read_rgb_mask(row, args.max_side)
    aligned_rgb, aligned_mask, debug = align_vertical_no_flip(rgb, mask)
    w = int(args.texas_canvas_w)
    h = int(args.texas_canvas_h)
    belly_rgb = cv2.resize(aligned_rgb, (w, h), interpolation=cv2.INTER_AREA)
    belly_mask = cv2.resize(aligned_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    belly_mask = base.clean_mask(belly_mask)

    coverage = float(belly_mask.mean() / 255.0)
    if coverage < 0.035:
        # If the mask is unexpectedly missing, use the full crop instead of an
        # artificial oval. This preserves the user's visual observation that the
        # SAM-clean crop itself is the useful standardized object.
        belly_mask = np.full((h, w), 255, dtype=np.uint8)
        quality *= 0.68
        coverage = 1.0
    elif coverage > 0.92:
        # Full masks are okay here; just reduce quality slightly because they
        # may contain a little background residue.
        quality *= 0.94

    debug = dict(debug)
    debug["mask_mode"] = "full_aligned_crop_no_oval"
    debug["mask_coverage"] = coverage

    heat = base.texas_dot_heat(belly_rgb, belly_mask)
    points = base.dot_points_from_heat(heat, belly_mask)
    small = cv2.resize(heat, (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    mask_small = cv2.resize((belly_mask > 0).astype(np.float32), (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    vector = base.normalize(np.concatenate([small * mask_small, mask_small * 0.18]).astype(np.float32))
    quality *= min(1.0, 0.72 + 0.28 * min(1.0, len(points) / 45.0))

    return base.TexasDotItem(
        image_id=int(row["image_id"]),
        current_cluster=str(current_cluster),
        source_path=str(row.get("source_path", "")),
        view_path=str(row.get("view_path", row.get("source_path", ""))),
        view_source=str(row.get("view_source", "")),
        belly_rgb=belly_rgb,
        belly_mask=belly_mask,
        dot_heat=heat,
        dot_points=points,
        vector=vector,
        quality=float(quality),
        debug=debug,
    )


def texas_pair_score_no_flip(a: base.TexasDotItem, b: base.TexasDotItem) -> dict:
    """Score Texas belly dots without vertical flip search."""
    desc_cos = float(np.dot(a.vector, b.vector))
    h, w = a.dot_heat.shape[:2]
    best = {
        "score": 0.0,
        "corr": 0.0,
        "overlap": 0.0,
        "stack_gain": 0.0,
        "point_score": 0.0,
        "descriptor_cosine": desc_cos,
        "dx": 0,
        "dy": 0,
        "transform": "identity",
    }
    common_mask = np.where((a.belly_mask > 0), 1.0, 0.0).astype(np.float32)
    base_dx, base_dy = base.phase_shift(a.dot_heat, b.dot_heat, common_mask)
    candidates = {(0, 0), (base_dx, base_dy)}
    for dx0, dy0 in [(base_dx, base_dy), (0, 0)]:
        for ddx in (-8, 0, 8):
            for ddy in (-8, 0, 8):
                candidates.add((int(np.clip(dx0 + ddx, -24, 24)), int(np.clip(dy0 + ddy, -24, 24))))
    for dx, dy in candidates:
        corr, overlap, stack_gain = base.masked_corr_and_stack(a.dot_heat, a.belly_mask, b.dot_heat, b.belly_mask, dx, dy)
        if overlap < 0.045:
            continue
        point_score = base.chamfer_dot_score(a.dot_points, b.dot_points, dx / max(1, w - 1), dy / max(1, h - 1))
        fused = (
            0.34 * corr
            + 0.31 * point_score
            + 0.23 * stack_gain
            + 0.12 * max(0.0, desc_cos)
        )
        fused *= min(a.quality, b.quality)
        if fused > best["score"]:
            best = {
                "score": float(fused),
                "corr": float(corr),
                "overlap": float(overlap),
                "stack_gain": float(stack_gain),
                "point_score": float(point_score),
                "descriptor_cosine": desc_cos,
                "dx": int(dx),
                "dy": int(dy),
                "transform": "identity",
            }
    best["points_a"] = int(len(a.dot_points))
    best["points_b"] = int(len(b.dot_points))
    best["quality_a"] = float(a.quality)
    best["quality_b"] = float(b.quality)
    best["same_current_cluster"] = bool(a.current_cluster == b.current_cluster)
    return best


def main() -> None:
    base.VERSION = VERSION
    base.align_vertical = align_vertical_no_flip
    base.texas_belly_template = texas_belly_template_no_oval
    base.texas_pair_score = texas_pair_score_no_flip
    base.main()


if __name__ == "__main__":
    main()
