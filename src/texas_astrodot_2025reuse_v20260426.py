
import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import species_fingerprint_final_swing_v20260426 as core


VERSION = "texas_astrodot_2025reuse_v20260426"
SEED = 20260426
TEXAS = "TexasHornedLizards"
REUSED_SPECIES = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
BACKGROUND = np.array([238, 238, 232], dtype=np.uint8)


@dataclass
class TexasDotItem:
    image_id: int
    current_cluster: str
    source_path: str
    view_path: str
    view_source: str
    belly_rgb: np.ndarray
    belly_mask: np.ndarray
    dot_heat: np.ndarray
    dot_points: np.ndarray
    vector: np.ndarray
    quality: float
    debug: dict


class UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}
        self.size = {int(v): 1 for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
            return value
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Texas astro-dot stacking branch plus 2025-winner-style train "
            "local-match validation for reused AnimalCLEF species. This script "
            "writes upload-ready CSVs only; it never submits to Kaggle."
        )
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sam-manifest", type=str, default=None)
    parser.add_argument("--current-best-submission", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-side", type=int, default=760)
    parser.add_argument("--texas-canvas-w", type=int, default=224)
    parser.add_argument("--texas-canvas-h", type=int, default=320)
    parser.add_argument("--texas-pair-budget", type=int, default=12000, help="0 means all Texas pairs.")
    parser.add_argument("--reused-train-pair-budget", type=int, default=1800)
    parser.add_argument("--reused-max-train-images", type=int, default=650)
    parser.add_argument("--reused-max-per-identity", type=int, default=8)
    parser.add_argument("--skip-reused-validation", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=18)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def find_data_root(user_value: str | None) -> Path:
    candidates: list[Path] = []
    if user_value:
        candidates.append(Path(user_value))
    candidates.extend(
        [
            Path("/kaggle/input/animal-clef-2026"),
            Path("/kaggle/input/competitions/animal-clef-2026"),
            Path.cwd() / "animal-clef-2026",
            Path.cwd().parent / "animal-clef-2026",
            Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\animal-clef-2026"),
        ]
    )
    for root in candidates:
        if (root / "metadata.csv").exists() and (root / "sample_submission.csv").exists():
            return root.resolve()
    raise FileNotFoundError("Could not locate AnimalCLEF2026 data root.")


def find_sam_manifest(user_value: str | None) -> Path | None:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    direct = [
        Path("/kaggle/working/animalclef_sam3_views_cache/reports/view_manifest_sam3_all_species.csv"),
        Path("/kaggle/input/animalclef2026-export-sam3-views-all-species-v2026/animalclef_sam3_views_cache/reports/view_manifest_sam3_all_species.csv"),
        Path("/kaggle/input/animalclef2026-export-sam3-views-all-species-v2026/reports/view_manifest_sam3_all_species.csv"),
        Path.cwd() / "animalclef_sam3_views_cache" / "reports" / "view_manifest_sam3_all_species.csv",
        Path.cwd().parent / "animalclef_sam3_views_cache" / "reports" / "view_manifest_sam3_all_species.csv",
    ]
    for p in direct:
        if p.exists():
            return p.resolve()
    for base in [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent]:
        if not base.exists():
            continue
        try:
            matches = list(base.rglob("view_manifest_sam3_all_species.csv"))
        except Exception:
            matches = []
        if matches:
            matches.sort(key=lambda x: len(str(x)))
            return matches[0].resolve()
    return None


def export_root_from_manifest(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "reports":
        return manifest_path.parent.parent
    return manifest_path.parent


def remap_export_path(path_value: object, export_root: Path | None) -> Path | None:
    if path_value is None or pd.isna(path_value):
        return None
    s = str(path_value).strip()
    if not s:
        return None
    p = Path(s)
    if p.exists():
        return p.resolve()
    if export_root is None:
        return None
    normalized = s.replace("\\", "/")
    markers = [
        "animalclef_sam3_views_cache/",
        "views/",
        "mask_loose_square/",
        "mask_full_square/",
    ]
    for marker in markers:
        if marker not in normalized:
            continue
        rel = normalized.split(marker, 1)[1]
        if marker != "animalclef_sam3_views_cache/":
            rel = marker + rel
        candidate = export_root / Path(rel)
        if candidate.exists():
            return candidate.resolve()
    return None


def prepare_metadata(data_root: Path, sam_manifest: Path | None) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_csv(data_root / "metadata.csv").reset_index(drop=True)
    if "row_idx" not in metadata.columns:
        metadata["row_idx"] = np.arange(len(metadata), dtype=np.int64)
    if "species_id" not in metadata.columns:
        metadata["species_id"] = metadata["dataset"].astype(str)
    if "split" not in metadata.columns:
        metadata["split"] = np.where(metadata["path"].str.contains("/test/"), "test", "train")
    if "orientation" not in metadata.columns:
        metadata["orientation"] = "unknown"
    if "identity" not in metadata.columns:
        metadata["identity"] = ""
    metadata["source_path"] = metadata["path"].map(lambda p: str(data_root / str(p)))
    metadata["view_path"] = metadata["source_path"].astype(str)
    metadata["view_source"] = "original_fallback"
    metadata["sam_view_path"] = ""
    metadata["mask_path"] = ""
    metadata["mask_ok"] = False

    info = {
        "manifest_path": str(sam_manifest) if sam_manifest else None,
        "manifest_rows": 0,
        "resolved_sam_views": 0,
        "resolved_masks": 0,
    }
    if sam_manifest is None:
        return metadata, info

    manifest = pd.read_csv(sam_manifest)
    info["manifest_rows"] = int(len(manifest))
    export_root = export_root_from_manifest(sam_manifest)
    merge_key = "row_idx" if "row_idx" in manifest.columns else "image_id" if "image_id" in manifest.columns else None
    if merge_key is None:
        return metadata, info
    merged = metadata.merge(manifest, on=merge_key, how="left", suffixes=("", "_sam"))

    view_paths: list[str] = []
    sam_view_paths: list[str] = []
    mask_paths: list[str] = []
    view_sources: list[str] = []
    mask_ok: list[bool] = []
    sam_count = 0
    mask_count = 0
    for row in merged.to_dict("records"):
        resolved_sam_view = None
        for col in ["loose_path", "mask_loose_path", "mask_full_path", "view_path"]:
            if col in row:
                resolved_sam_view = remap_export_path(row.get(col), export_root)
                if resolved_sam_view is not None:
                    break
        resolved_mask = None
        for col in ["mask_path", "binary_mask_path"]:
            if col in row:
                resolved_mask = remap_export_path(row.get(col), export_root)
                if resolved_mask is not None:
                    break
        if resolved_sam_view is not None:
            view_paths.append(str(resolved_sam_view))
            sam_view_paths.append(str(resolved_sam_view))
            view_sources.append("sam_clean_primary")
            sam_count += 1
        else:
            view_paths.append(str(row["source_path"]))
            sam_view_paths.append("")
            view_sources.append("original_fallback")
        if resolved_mask is not None:
            mask_paths.append(str(resolved_mask))
            mask_ok.append(True)
            mask_count += 1
        else:
            mask_paths.append("")
            mask_ok.append(False)
    merged["view_path"] = view_paths
    merged["sam_view_path"] = sam_view_paths
    merged["view_source"] = view_sources
    merged["mask_path"] = mask_paths
    merged["mask_ok"] = mask_ok
    info["resolved_sam_views"] = int(sam_count)
    info["resolved_masks"] = int(mask_count)
    return merged, info


def find_current_best(user_value: str | None, data_root: Path) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    filename = core.CURRENT_BEST_FILENAME
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
        Path.cwd().parent,
        data_root.parent,
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\kaggle_upload_atlas_inputs_v20260426"),
    ]
    found = core.find_file_everywhere(filename, roots)
    if found is None:
        raise FileNotFoundError(f"Could not find {filename}.")
    return found


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def clean_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    if shape is not None and m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        m = np.where(m > 0, 255, 0).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas):
            biggest = 1 + int(np.argmax(areas))
            m = np.where(labels == biggest, 255, 0).astype(np.uint8)
    if float(m.mean() / 255.0) < 0.01:
        m = np.full(m.shape[:2], 255, dtype=np.uint8)
    return m


def read_rgb_mask(row: dict, max_side: int) -> tuple[np.ndarray, np.ndarray, float]:
    rgb, mask = core.read_rgb_with_optional_mask(row.get("view_path", row.get("source_path")), row.get("mask_path"), max_side)
    quality = 1.0
    if mask is None:
        mask = core.estimate_foreground_mask(rgb)
        quality = 0.86
    mask = clean_mask(mask, rgb.shape[:2])
    coverage = float(mask.mean() / 255.0)
    if coverage < 0.015 or coverage > 0.98:
        mask = core.estimate_foreground_mask(rgb)
        mask = clean_mask(mask, rgb.shape[:2])
        quality *= 0.78
    return rgb, mask, quality


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = core.bbox_from_mask(mask, pad)
    return rgb[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def align_vertical(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    crop_rgb, crop_mask = crop_to_mask(rgb, mask, 0.08)
    angle = core.pca_angle_degrees(crop_mask)
    rotate_angle = 90.0 - angle
    if abs(rotate_angle) > 1.5:
        crop_rgb, crop_mask = core.rotate_bound(crop_rgb, crop_mask, rotate_angle)
        crop_rgb, crop_mask = crop_to_mask(crop_rgb, crop_mask, 0.04)
    h, w = crop_mask.shape[:2]
    widths = []
    for yf in [0.18, 0.30, 0.70, 0.84]:
        y = int(np.clip(round(h * yf), 0, h - 1))
        xs = np.where(crop_mask[y] > 0)[0]
        widths.append(float(xs.max() - xs.min() + 1) if len(xs) else 0.0)
    top_width = max(widths[:2])
    bottom_width = max(widths[2:])
    flipped = False
    # Most Texas belly photos have the head/shoulder end above the tail end.
    # If the bottom bands are clearly wider, canonicalize by vertical flip.
    if bottom_width > top_width * 1.18:
        crop_rgb = crop_rgb[::-1, :, :].copy()
        crop_mask = crop_mask[::-1, :].copy()
        flipped = True
    return crop_rgb, crop_mask, {
        "pca_angle": float(angle),
        "rotate_angle": float(rotate_angle),
        "top_width": float(top_width),
        "bottom_width": float(bottom_width),
        "flipped_vertical": bool(flipped),
    }


def clahe_u8(channel: np.ndarray) -> np.ndarray:
    channel = np.clip(channel, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(channel)


def texas_dot_heat(belly_rgb: np.ndarray, belly_mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(belly_rgb, cv2.COLOR_RGB2LAB)
    l_eq = clahe_u8(lab[:, :, 0])
    dark = (255.0 - l_eq.astype(np.float32)) / 255.0
    blackhat = cv2.morphologyEx(l_eq, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    spot = 0.58 * (blackhat.astype(np.float32) / 255.0) + 0.42 * dark
    valid = belly_mask > 0
    if valid.sum() > 20:
        vals = spot[valid]
        lo = float(np.percentile(vals, 12))
        hi = float(np.percentile(vals, 98))
        spot = (spot - lo) / max(1e-6, hi - lo)
    spot = np.clip(spot, 0, 1)
    spot[~valid] = 0.0
    spot = cv2.GaussianBlur(spot, (3, 3), 0)
    return spot.astype(np.float32)


def texas_belly_template(row: dict, current_cluster: str, args: argparse.Namespace) -> TexasDotItem:
    rgb, mask, quality = read_rgb_mask(row, args.max_side)
    aligned_rgb, aligned_mask, debug = align_vertical(rgb, mask)
    w = int(args.texas_canvas_w)
    h = int(args.texas_canvas_h)
    belly_rgb = cv2.resize(aligned_rgb, (w, h), interpolation=cv2.INTER_AREA)
    belly_mask = cv2.resize(aligned_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    belly_mask = clean_mask(belly_mask)
    ellipse = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        ellipse,
        (w // 2, int(h * 0.49)),
        (int(w * 0.38), int(h * 0.35)),
        0,
        0,
        360,
        255,
        -1,
    )
    belly_mask = cv2.bitwise_and(belly_mask, ellipse)
    if float(belly_mask.mean() / 255.0) < 0.035:
        belly_mask = ellipse
        quality *= 0.72
    heat = texas_dot_heat(belly_rgb, belly_mask)
    points = dot_points_from_heat(heat, belly_mask)
    small = cv2.resize(heat, (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    mask_small = cv2.resize((belly_mask > 0).astype(np.float32), (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    vector = normalize(np.concatenate([small * mask_small, mask_small * 0.18]).astype(np.float32))
    quality *= min(1.0, 0.72 + 0.28 * min(1.0, len(points) / 35.0))
    return TexasDotItem(
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


def dot_points_from_heat(heat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask > 0
    if int(valid.sum()) < 40:
        return np.zeros((0, 4), dtype=np.float32)
    vals = heat[valid]
    thr = max(float(np.percentile(vals, 86)), float(vals.mean() + 0.60 * vals.std()))
    binary = np.zeros(heat.shape, dtype=np.uint8)
    binary[(heat >= thr) & valid] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    h, w = heat.shape[:2]
    total = float(h * w)
    pts: list[list[float]] = []
    min_area = max(2.0, total * 0.00005)
    max_area = total * 0.012
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = max(1.0, float(stats[idx, cv2.CC_STAT_WIDTH]))
        bh = max(1.0, float(stats[idx, cv2.CC_STAT_HEIGHT]))
        if max(bw / bh, bh / bw) > 5.5:
            continue
        cx, cy = centroids[idx]
        strength = float(heat[labels == idx].mean())
        pts.append([float(cx) / max(1, w - 1), float(cy) / max(1, h - 1), area / total, strength])
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    pts_arr = np.asarray(pts, dtype=np.float32)
    order = np.argsort(-pts_arr[:, 3])
    return pts_arr[order[:260]]


def shifted_slices(shape: tuple[int, int], dx: int, dy: int):
    h, w = shape
    xa1 = max(0, dx)
    xb1 = max(0, -dx)
    ya1 = max(0, dy)
    yb1 = max(0, -dy)
    ww = w - abs(dx)
    hh = h - abs(dy)
    if ww <= 8 or hh <= 8:
        return None
    return (slice(ya1, ya1 + hh), slice(xa1, xa1 + ww)), (slice(yb1, yb1 + hh), slice(xb1, xb1 + ww))


def dot_map_sharpness(heat: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if int(valid.sum()) < 20:
        return 0.0
    vals = heat[valid].astype(np.float32)
    p95 = float(np.percentile(vals, 95))
    p50 = float(np.percentile(vals, 50))
    lap = cv2.Laplacian(heat, cv2.CV_32F, ksize=3)
    lap_energy = float(np.mean(np.abs(lap[valid])))
    return float(max(0.0, p95 - p50) + 0.35 * lap_energy)


def masked_corr_and_stack(
    a_heat: np.ndarray,
    a_mask: np.ndarray,
    b_heat: np.ndarray,
    b_mask: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[float, float, float]:
    slices = shifted_slices(a_mask.shape[:2], dx, dy)
    if slices is None:
        return 0.0, 0.0, 0.0
    sa, sb = slices
    ma = a_mask[sa] > 0
    mb = b_mask[sb] > 0
    overlap_mask = ma & mb
    overlap = float(overlap_mask.mean()) if overlap_mask.size else 0.0
    if overlap < 0.04 or int(overlap_mask.sum()) < 40:
        return 0.0, overlap, 0.0
    va = a_heat[sa][overlap_mask].astype(np.float32)
    vb = b_heat[sb][overlap_mask].astype(np.float32)
    am = va - float(va.mean())
    bm = vb - float(vb.mean())
    denom = float(np.linalg.norm(am) * np.linalg.norm(bm))
    corr = float(np.dot(am, bm) / denom) if denom > 1e-6 else 0.0
    corr01 = float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))
    stack = np.zeros_like(a_heat, dtype=np.float32)
    stack_mask = np.zeros_like(a_mask, dtype=np.uint8)
    stack[sa] = 0.5 * (a_heat[sa] + b_heat[sb])
    stack_mask[sa] = np.where(overlap_mask, 255, 0).astype(np.uint8)
    sharp_stack = dot_map_sharpness(stack, stack_mask)
    sharp_a = dot_map_sharpness(a_heat[sa], np.where(overlap_mask, 255, 0).astype(np.uint8))
    sharp_b = dot_map_sharpness(b_heat[sb], np.where(overlap_mask, 255, 0).astype(np.uint8))
    baseline = 0.5 * (sharp_a + sharp_b)
    # Same individuals should retain or improve peakiness after stacking;
    # different dot fields smear and reduce the normalized sharpness.
    stack_gain = float(np.clip(sharp_stack / max(1e-6, baseline), 0.0, 1.35) / 1.35)
    return corr01, overlap, stack_gain


def chamfer_dot_score(points_a: np.ndarray, points_b: np.ndarray, dx_norm: float, dy_norm: float) -> float:
    if len(points_a) < 5 or len(points_b) < 5:
        return 0.0
    a = points_a[:, :2].astype(np.float32)
    b = points_b[:, :2].astype(np.float32).copy()
    b[:, 0] += dx_norm
    b[:, 1] += dy_norm
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.maximum(0.0, (diff * diff).sum(axis=2)))
    da = dist.min(axis=1)
    db = dist.min(axis=0)
    ka = np.argsort(da)[: min(len(da), 120)]
    kb = np.argsort(db)[: min(len(db), 120)]
    mean_d = 0.5 * (float(da[ka].mean()) + float(db[kb].mean()))
    return float(np.exp(-mean_d / 0.050))


def phase_shift(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    try:
        aw = (a * mask).astype(np.float32)
        bw = (b * mask).astype(np.float32)
        shift, _ = cv2.phaseCorrelate(aw, bw)
        dx = int(np.clip(round(shift[0]), -18, 18))
        dy = int(np.clip(round(shift[1]), -18, 18))
        return dx, dy
    except Exception:
        return 0, 0


def transform_heat_mask_points(item: TexasDotItem, transform: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat = item.dot_heat
    mask = item.belly_mask
    pts = item.dot_points.copy()
    if transform in {"flip_x", "flip_xy"}:
        heat = heat[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        if len(pts):
            pts[:, 0] = 1.0 - pts[:, 0]
    if transform in {"flip_y", "flip_xy"}:
        heat = heat[::-1, :].copy()
        mask = mask[::-1, :].copy()
        if len(pts):
            pts[:, 1] = 1.0 - pts[:, 1]
    return heat, mask, pts


def texas_pair_score(a: TexasDotItem, b: TexasDotItem) -> dict:
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
    transforms = ["identity", "flip_y"]
    common_mask = np.where((a.belly_mask > 0), 1.0, 0.0).astype(np.float32)
    for transform in transforms:
        b_heat, b_mask, b_pts = transform_heat_mask_points(b, transform)
        base_dx, base_dy = phase_shift(a.dot_heat, b_heat, common_mask)
        candidates = {(0, 0), (base_dx, base_dy)}
        for dx0, dy0 in [(base_dx, base_dy), (0, 0)]:
            for ddx in (-8, 0, 8):
                for ddy in (-8, 0, 8):
                    candidates.add((int(np.clip(dx0 + ddx, -24, 24)), int(np.clip(dy0 + ddy, -24, 24))))
        for dx, dy in candidates:
            corr, overlap, stack_gain = masked_corr_and_stack(a.dot_heat, a.belly_mask, b_heat, b_mask, dx, dy)
            if overlap < 0.045:
                continue
            point_score = chamfer_dot_score(a.dot_points, b_pts, dx / max(1, w - 1), dy / max(1, h - 1))
            fused = (
                0.34 * corr
                + 0.31 * point_score
                + 0.23 * stack_gain
                + 0.12 * max(0.0, desc_cos)
            )
            if transform != "identity":
                fused *= 0.92
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
                    "transform": transform,
                }
    best["points_a"] = int(len(a.dot_points))
    best["points_b"] = int(len(b.dot_points))
    best["quality_a"] = float(a.quality)
    best["quality_b"] = float(b.quality)
    best["same_current_cluster"] = bool(a.current_cluster == b.current_cluster)
    return best


def score_all_texas_pairs(items: list[TexasDotItem], pair_budget: int = 0) -> pd.DataFrame:
    vectors = np.stack([it.vector for it in items]).astype(np.float32)
    sim = vectors @ vectors.T
    ids = [it.image_id for it in items]
    by_id = {it.image_id: it for it in items}
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    if pair_budget and len(pairs) > pair_budget:
        id_to_idx = {image_id: idx for idx, image_id in enumerate(ids)}
        pairs.sort(key=lambda p: float(sim[id_to_idx[p[0]], id_to_idx[p[1]]]), reverse=True)
        current_pairs = [p for p in pairs if by_id[p[0]].current_cluster == by_id[p[1]].current_cluster]
        selected = current_pairs + [p for p in pairs if by_id[p[0]].current_cluster != by_id[p[1]].current_cluster]
        pairs = selected[:pair_budget]
    rows = []
    for idx, (a_id, b_id) in enumerate(pairs, start=1):
        a = by_id[int(a_id)]
        b = by_id[int(b_id)]
        score = texas_pair_score(a, b)
        rows.append(
            {
                "species": TEXAS,
                "image_id_a": int(a_id),
                "image_id_b": int(b_id),
                "current_cluster_a": a.current_cluster,
                "current_cluster_b": b.current_cluster,
                **score,
            }
        )
        if idx % 5000 == 0:
            print(f"[Texas astro-dot] scored {idx}/{len(pairs)} pairs")
    return pd.DataFrame(rows)


def relabel_components(ids: list[int], uf: UnionFind, variant: str) -> dict[int, str]:
    comp_order: dict[int, int] = {}
    labels: dict[int, str] = {}
    for image_id in sorted(ids):
        comp = uf.find(image_id)
        if comp not in comp_order:
            comp_order[comp] = len(comp_order)
        labels[image_id] = f"cluster_TexasHornedLizards_astrodot_{variant}_{comp_order[comp]}"
    return labels


def texas_variant_labels(items: list[TexasDotItem], pair_scores: pd.DataFrame, variant: str) -> dict[int, str]:
    ids = [it.image_id for it in items]
    by_cluster: dict[str, list[int]] = {}
    current_by_id = {it.image_id: it.current_cluster for it in items}
    for it in items:
        by_cluster.setdefault(it.current_cluster, []).append(it.image_id)

    # Thresholds are deliberately strict. Texas has no train labels, so this
    # branch should produce surgical candidates, not a free-running clusterer.
    if variant == "split_strict":
        keep_thr, merge_thr, merge_rank = 0.430, 9.9, 0
    elif variant == "merge_ultra":
        keep_thr, merge_thr, merge_rank = 0.0, 0.650, 2
    elif variant == "splitmerge_guarded":
        keep_thr, merge_thr, merge_rank = 0.405, 0.620, 2
    else:
        keep_thr, merge_thr, merge_rank = 0.385, 0.590, 3

    uf = UnionFind(ids)
    if variant in {"merge_ultra"}:
        # Preserve current clusters, then add only extremely high-confidence
        # cross-cluster dot-stack merges.
        for members in by_cluster.values():
            anchor = members[0]
            for other in members[1:]:
                uf.union(anchor, other)
    else:
        # Split current clusters by verified intra-cluster dot support.
        for cluster, members in by_cluster.items():
            if len(members) <= 1:
                continue
            g = pair_scores[
                pair_scores["image_id_a"].isin(members)
                & pair_scores["image_id_b"].isin(members)
                & pair_scores["same_current_cluster"].astype(bool)
            ]
            for row in g[g["score"].astype(float) >= keep_thr].itertuples(index=False):
                uf.union(int(row.image_id_a), int(row.image_id_b))

    if merge_rank > 0:
        cross = pair_scores[~pair_scores["same_current_cluster"].astype(bool)].copy()
        if not cross.empty:
            cross = cross[
                (cross["score"].astype(float) >= merge_thr)
                & (cross["overlap"].astype(float) >= 0.13)
                & (cross["point_score"].astype(float) >= 0.36)
                & (cross["stack_gain"].astype(float) >= 0.43)
            ].copy()
            if not cross.empty:
                neighbors: dict[int, list[tuple[int, float]]] = {i: [] for i in ids}
                for row in cross.itertuples(index=False):
                    a = int(row.image_id_a)
                    b = int(row.image_id_b)
                    s = float(row.score)
                    neighbors[a].append((b, s))
                    neighbors[b].append((a, s))
                ranks: dict[tuple[int, int], int] = {}
                for node, vals in neighbors.items():
                    vals.sort(key=lambda x: -x[1])
                    for rank, (other, _) in enumerate(vals, start=1):
                        ranks[(node, other)] = rank
                for row in cross.sort_values("score", ascending=False).itertuples(index=False):
                    a = int(row.image_id_a)
                    b = int(row.image_id_b)
                    if current_by_id[a] == current_by_id[b]:
                        continue
                    if ranks.get((a, b), 999) <= merge_rank and ranks.get((b, a), 999) <= merge_rank:
                        uf.union(a, b)
    components: dict[int, list[int]] = {}
    for image_id in ids:
        components.setdefault(uf.find(image_id), []).append(image_id)
    current_sets = {cluster: set(members) for cluster, members in by_cluster.items()}
    comp_order: dict[int, int] = {}
    labels: dict[int, str] = {}
    for root, members in sorted(components.items(), key=lambda kv: min(kv[1])):
        member_set = set(members)
        member_current = {current_by_id[i] for i in members}
        if len(member_current) == 1:
            current_cluster = next(iter(member_current))
            if member_set == current_sets.get(current_cluster, set()):
                for image_id in members:
                    labels[image_id] = current_cluster
                continue
        comp_order[root] = len(comp_order)
        new_label = f"cluster_TexasHornedLizards_astrodot_{variant}_{comp_order[root]}"
        for image_id in members:
            labels[image_id] = new_label
    return labels


def summarize_texas_variant(labels: dict[int, str], pair_scores: pd.DataFrame, variant: str, current_labels: dict[int, str]) -> dict:
    counts = pd.Series(list(labels.values())).value_counts()
    changed = int(sum(1 for i, lab in labels.items() if lab != current_labels.get(i)))
    return {
        "variant": variant,
        "species": TEXAS,
        "n_images": int(len(labels)),
        "n_clusters": int(counts.shape[0]),
        "singletons": int((counts == 1).sum()),
        "max_cluster_size": int(counts.max()) if not counts.empty else 0,
        "rows_changed_vs_current": changed,
        "pair_score_mean": float(pair_scores["score"].mean()) if not pair_scores.empty else 0.0,
        "pair_score_p95": float(pair_scores["score"].quantile(0.95)) if not pair_scores.empty else 0.0,
        "same_current_mean": float(pair_scores.loc[pair_scores["same_current_cluster"].astype(bool), "score"].mean())
        if not pair_scores.empty and pair_scores["same_current_cluster"].astype(bool).any()
        else 0.0,
        "cross_current_p99": float(pair_scores.loc[~pair_scores["same_current_cluster"].astype(bool), "score"].quantile(0.99))
        if not pair_scores.empty and (~pair_scores["same_current_cluster"].astype(bool)).any()
        else 0.0,
    }


def root_sift(desc: np.ndarray | None) -> np.ndarray | None:
    if desc is None or len(desc) == 0:
        return desc
    desc = desc.astype(np.float32)
    desc /= np.maximum(1e-7, desc.sum(axis=1, keepdims=True))
    return np.sqrt(desc).astype(np.float32)


def extract_local_features(row: dict, species: str, max_side: int, detector) -> tuple[list, np.ndarray | None, np.ndarray]:
    cfg = core.SPECIES_CONFIGS[species]
    rgb, mask = core.read_rgb_with_optional_mask(row.get("view_path", row.get("source_path")), row.get("mask_path"), max_side)
    if mask is None:
        mask = core.estimate_foreground_mask(rgb)
    roi_rgb, roi_mask, _ = core.species_roi(
        rgb,
        species,
        str(row.get("orientation", "unknown")).lower(),
        cfg,
        mask_override=mask,
    )
    gray = core.pattern_gray(roi_rgb, species)
    kps, desc = detector.detectAndCompute(gray, roi_mask)
    if kps is None:
        kps = []
    desc = root_sift(desc)
    return kps, desc, roi_mask


def local_pair_score(a_feat, b_feat, species: str) -> dict:
    kps_a, desc_a, mask_a = a_feat
    kps_b, desc_b, mask_b = b_feat
    if desc_a is None or desc_b is None or len(desc_a) < 4 or len(desc_b) < 4:
        return {"score": 0.0, "inliers": 0, "good_matches": 0, "inlier_ratio": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    try:
        knn = matcher.knnMatch(desc_a, desc_b, k=2)
    except Exception:
        return {"score": 0.0, "inliers": 0, "good_matches": 0, "inlier_ratio": 0.0}
    ratio = 0.80 if species == "SalamanderID2025" else 0.76
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 4:
        return {"score": 0.0, "inliers": 0, "good_matches": len(good), "inlier_ratio": 0.0}
    pts_a = np.float32([kps_a[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in good]).reshape(-1, 2)
    _, inlier_mask = cv2.estimateAffinePartial2D(
        pts_a,
        pts_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=1600,
        confidence=0.995,
    )
    inliers = int(inlier_mask.reshape(-1).sum()) if inlier_mask is not None else 0
    denom = max(1, min(len(kps_a), len(kps_b), len(good)))
    inlier_ratio = float(inliers / denom)
    inlier_term = min(1.0, inliers / (28.0 if species != "SalamanderID2025" else 18.0))
    ratio_term = min(1.0, inlier_ratio / 0.25)
    match_term = min(1.0, len(good) / 60.0)
    score = 0.55 * inlier_term + 0.30 * ratio_term + 0.15 * match_term
    return {"score": float(score), "inliers": inliers, "good_matches": int(len(good)), "inlier_ratio": inlier_ratio}


def sample_train_rows(rows: pd.DataFrame, max_images: int, max_per_identity: int) -> pd.DataFrame:
    labeled = rows[rows["identity"].fillna("").astype(str).str.len().gt(0)].copy()
    if labeled.empty:
        return labeled
    rng = random.Random(SEED)
    chosen = []
    for _, group in labeled.groupby("identity"):
        recs = group.sort_values("image_id").to_dict("records")
        if len(recs) > max_per_identity:
            recs = rng.sample(recs, max_per_identity)
        chosen.extend(recs)
    if max_images and len(chosen) > max_images:
        chosen = rng.sample(chosen, max_images)
    chosen.sort(key=lambda r: int(r["image_id"]))
    return pd.DataFrame(chosen)


def sample_validation_pairs(items: list[dict], budget: int) -> list[tuple[int, int]]:
    rng = random.Random(SEED)
    ids = [int(r["image_id"]) for r in items]
    identities = {int(r["image_id"]): str(r.get("identity", "")) for r in items}
    by_identity: dict[str, list[int]] = {}
    for i in ids:
        by_identity.setdefault(identities[i], []).append(i)
    positives = []
    for members in by_identity.values():
        if len(members) < 2:
            continue
        combos = list(itertools.combinations(sorted(members), 2))
        if len(combos) > 10:
            combos = rng.sample(combos, 10)
        positives.extend(combos)
    if len(positives) > budget // 2:
        positives = rng.sample(positives, max(1, budget // 2))
    negatives = set()
    target_neg = min(budget - len(positives), max(600 if positives else budget, len(positives) * 2))
    attempts = 0
    while len(negatives) < target_neg and attempts < max(100, target_neg * 35) and len(ids) >= 2:
        a, b = rng.sample(ids, 2)
        attempts += 1
        if identities[a] == identities[b]:
            continue
        negatives.add((a, b) if a < b else (b, a))
    pairs = positives + sorted(negatives)
    if len(pairs) > budget:
        pairs = rng.sample(pairs, budget)
    return [(int(a), int(b)) for a, b in pairs]


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def validate_reused_species(metadata: pd.DataFrame, args: argparse.Namespace, reports_dir: Path) -> pd.DataFrame:
    rows_out = []
    try:
        detector = cv2.SIFT_create(nfeatures=1100, contrastThreshold=0.012, edgeThreshold=12)
    except Exception:
        print("[warn] SIFT unavailable; reused-species validation skipped.")
        return pd.DataFrame()
    train_all = metadata[metadata["split"].eq("train")].copy()
    for species in REUSED_SPECIES:
        print(f"[{species}] 2025-style local validation")
        train_rows = train_all[train_all["species_id"].eq(species)].copy()
        train_rows = sample_train_rows(train_rows, args.reused_max_train_images, args.reused_max_per_identity)
        if args.smoke:
            train_rows = train_rows.head(44)
        records = train_rows.to_dict("records")
        if not records:
            continue
        feats = {}
        failures = 0
        for idx, row in enumerate(records, start=1):
            try:
                feats[int(row["image_id"])] = extract_local_features(row, species, args.max_side, detector)
            except Exception as exc:
                failures += 1
                feats[int(row["image_id"])] = ([], None, np.zeros((8, 8), dtype=np.uint8))
                print(f"[warn] {species} feature fail image_id={row.get('image_id')}: {exc}")
            if idx % 100 == 0:
                print(f"[{species}] local features {idx}/{len(records)}")
        pairs = sample_validation_pairs(records, min(args.reused_train_pair_budget, 350 if args.smoke else args.reused_train_pair_budget))
        score_rows = []
        identity_by_id = {int(r["image_id"]): str(r.get("identity", "")) for r in records}
        for a, b in pairs:
            s = local_pair_score(feats[a], feats[b], species)
            same = bool(identity_by_id[a] and identity_by_id[a] == identity_by_id[b])
            score_rows.append({"species": species, "image_id_a": a, "image_id_b": b, "same_identity": same, **s})
        pair_df = pd.DataFrame(score_rows)
        pair_df.to_csv(reports_dir / f"{VERSION}_{species}_train_local_pair_scores.csv", index=False)
        if pair_df.empty:
            continue
        y = pair_df["same_identity"].astype(bool).to_numpy()
        scores = pair_df["score"].astype(float).to_numpy()
        row = {
            "species": species,
            "n_pairs": int(len(pair_df)),
            "n_positive": int(y.sum()),
            "n_negative": int((~y).sum()),
            "auc": auc_rank(y, scores),
            "feature_failures": int(failures),
            "median_inliers_positive": float(pair_df.loc[pair_df["same_identity"], "inliers"].median()) if y.any() else 0.0,
            "median_inliers_negative": float(pair_df.loc[~pair_df["same_identity"], "inliers"].median()) if (~y).any() else 0.0,
        }
        for thr in [0.35, 0.45, 0.55, 0.65]:
            pred = scores >= thr
            tp = int((pred & y).sum())
            fp = int((pred & (~y)).sum())
            fn = int(((~pred) & y).sum())
            row[f"precision_at_{thr:.2f}"] = float(tp / max(1, tp + fp))
            row[f"recall_at_{thr:.2f}"] = float(tp / max(1, tp + fn))
            row[f"accepted_at_{thr:.2f}"] = int(pred.sum())
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def build_submission(current: pd.DataFrame, test_rows: pd.DataFrame, texas_labels: dict[int, str], out_path: Path) -> pd.DataFrame:
    sub = current.copy()
    texas_ids = set(test_rows.loc[test_rows["species_id"].eq(TEXAS), "image_id"].astype(int).tolist())
    current_map = dict(zip(sub["image_id"].astype(int), sub["cluster"].astype(str)))
    sub["cluster"] = sub["image_id"].astype(int).map(
        lambda i: texas_labels.get(i, current_map[i]) if i in texas_ids else current_map[i]
    )
    sub.to_csv(out_path, index=False)
    return sub


def summarize_submission(sub: pd.DataFrame, current: pd.DataFrame, test_rows: pd.DataFrame, variant: str) -> list[dict]:
    cur_map = dict(zip(current["image_id"].astype(int), current["cluster"].astype(str)))
    sub_map = dict(zip(sub["image_id"].astype(int), sub["cluster"].astype(str)))
    rows = []
    for species in core.SPECIES_ORDER:
        ids = test_rows.loc[test_rows["species_id"].eq(species), "image_id"].astype(int).tolist()
        labels = [sub_map[i] for i in ids]
        counts = pd.Series(labels).value_counts()
        rows.append(
            {
                "variant": variant,
                "species": species,
                "n_images": int(len(ids)),
                "n_clusters": int(counts.shape[0]),
                "singletons": int((counts == 1).sum()),
                "max_cluster_size": int(counts.max()) if not counts.empty else 0,
                "rows_changed_vs_current": int(sum(1 for i in ids if sub_map[i] != cur_map[i])),
            }
        )
    return rows


def heat_to_image(heat: np.ndarray, mask: np.ndarray) -> Image.Image:
    h = np.clip(heat, 0, 1)
    arr = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(h * 255, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip((1.0 - h) * 120, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip((1.0 - h) * 90, 0, 255).astype(np.uint8)
    arr[mask == 0] = BACKGROUND
    return Image.fromarray(arr)


def draw_texas_item(item: TexasDotItem, size: tuple[int, int]) -> Image.Image:
    rgb = item.belly_rgb.copy()
    mask = item.belly_mask
    dim = (rgb.astype(np.float32) * 0.40 + BACKGROUND.astype(np.float32) * 0.60).astype(np.uint8)
    rgb[mask == 0] = dim[mask == 0]
    contours, _ = cv2.findContours(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (30, 255, 70), 2)
    h, w = mask.shape[:2]
    for p in item.dot_points[:160]:
        x = int(round(float(p[0]) * max(1, w - 1)))
        y = int(round(float(p[1]) * max(1, h - 1)))
        cv2.circle(rgb, (x, y), 2, (255, 30, 30), 1, cv2.LINE_AA)
    img = Image.fromarray(rgb)
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def thumb(path: str, size: tuple[int, int]) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (25, 25, 25))
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def save_texas_preview(items: list[TexasDotItem], out_path: Path, limit: int) -> None:
    chosen = items[:limit]
    if not chosen:
        return
    tile_w, tile_h = 190, 190
    label_h = 22
    cols = 4
    canvas = Image.new("RGB", (cols * tile_w, len(chosen) * (tile_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for r, item in enumerate(chosen):
        y = r * (tile_h + label_h)
        panels = [
            thumb(item.source_path, (tile_w, tile_h)),
            thumb(item.view_path, (tile_w, tile_h)),
            draw_texas_item(item, (tile_w, tile_h)),
            heat_to_image(item.dot_heat, item.belly_mask),
        ]
        labels = [f"orig {item.image_id}", item.view_source[:20], "belly dots", "dot heat"]
        for c, panel in enumerate(panels):
            panel = panel.copy()
            panel.thumbnail((tile_w, tile_h), Image.Resampling.BILINEAR)
            x = c * tile_w
            draw.text((x + 5, y + 4), labels[c], fill=(245, 240, 145))
            canvas.paste(panel, (x + (tile_w - panel.width) // 2, y + label_h + (tile_h - panel.height) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def save_pair_preview(items: list[TexasDotItem], pair_scores: pd.DataFrame, out_path: Path, limit: int, cross_only: bool = False) -> None:
    if pair_scores.empty:
        return
    g = pair_scores.copy()
    if cross_only:
        g = g[~g["same_current_cluster"].astype(bool)]
    g = g.sort_values("score", ascending=False).head(limit)
    if g.empty:
        return
    by_id = {it.image_id: it for it in items}
    panel_w, panel_h = 840, 260
    rows = []
    for edge in g.to_dict("records"):
        a = by_id.get(int(edge["image_id_a"]))
        b = by_id.get(int(edge["image_id_b"]))
        if a is None or b is None:
            continue
        pa = heat_to_image(a.dot_heat, a.belly_mask)
        pb = heat_to_image(b.dot_heat, b.belly_mask)
        pa.thumbnail((300, 220), Image.Resampling.BILINEAR)
        pb.thumbnail((300, 220), Image.Resampling.BILINEAR)
        row_img = Image.new("RGB", (panel_w, panel_h), (18, 18, 18))
        draw = ImageDraw.Draw(row_img)
        text = (
            f"{a.image_id} vs {b.image_id} score={float(edge['score']):.3f} "
            f"corr={float(edge['corr']):.3f} pts={float(edge['point_score']):.3f} "
            f"stack={float(edge['stack_gain']):.3f} same_current={edge['same_current_cluster']}"
        )
        draw.text((6, 5), text, fill=(255, 240, 120))
        row_img.paste(pa, (12, 35))
        row_img.paste(pb, (440, 35))
        rows.append(row_img)
    canvas = Image.new("RGB", (panel_w, panel_h * len(rows)), (18, 18, 18))
    for idx, row in enumerate(rows):
        canvas.paste(row, (0, idx * panel_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    args = parse_args()
    if args.smoke:
        args.texas_pair_budget = min(args.texas_pair_budget or 600, 600)
        args.reused_train_pair_budget = min(args.reused_train_pair_budget, 260)
        args.reused_max_train_images = min(args.reused_max_train_images, 48)
        args.save_visualizations = True

    data_root = find_data_root(args.data_root)
    sam_manifest = find_sam_manifest(args.sam_manifest)
    metadata, manifest_info = prepare_metadata(data_root, sam_manifest)
    metadata = metadata[metadata["species_id"].isin(core.SPECIES_ORDER)].copy()
    test_rows = metadata[metadata["split"].eq("test")].copy()
    output_root = Path(args.output_root) if args.output_root else Path.cwd() / f"animalclef_{VERSION}"
    reports_dir = output_root / "reports"
    sub_dir = output_root / "submissions"
    viz_dir = output_root / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        viz_dir.mkdir(parents=True, exist_ok=True)

    current_best_path = find_current_best(args.current_best_submission, data_root)
    current = core.load_submission(current_best_path)
    current_labels = dict(zip(current["image_id"].astype(int), current["cluster"].astype(str)))

    print(f"VERSION={VERSION}")
    print(f"data_root={data_root}")
    print(f"sam_manifest={sam_manifest}")
    print(f"current_best={current_best_path}")
    print(f"output_root={output_root}")
    print(json.dumps(manifest_info, indent=2))

    texas_rows = test_rows[test_rows["species_id"].eq(TEXAS)].sort_values("image_id").copy()
    if args.smoke:
        texas_rows = texas_rows.head(36)
    texas_items: list[TexasDotItem] = []
    for idx, row in enumerate(texas_rows.to_dict("records"), start=1):
        image_id = int(row["image_id"])
        try:
            texas_items.append(texas_belly_template(row, current_labels[image_id], args))
        except Exception as exc:
            print(f"[warn] Texas template failed image_id={image_id}: {exc}")
        if idx % 75 == 0:
            print(f"[Texas astro-dot] templates {idx}/{len(texas_rows)}")

    if args.save_visualizations:
        save_texas_preview(texas_items, viz_dir / f"{VERSION}_Texas_template_preview.jpg", args.visual_limit)

    texas_pair_scores = score_all_texas_pairs(texas_items, args.texas_pair_budget)
    texas_pair_scores.to_csv(reports_dir / f"{VERSION}_Texas_pair_scores.csv", index=False)
    if args.save_visualizations:
        save_pair_preview(texas_items, texas_pair_scores, viz_dir / f"{VERSION}_Texas_top_pairs_all.jpg", max(5, args.visual_limit // 2), False)
        save_pair_preview(texas_items, texas_pair_scores, viz_dir / f"{VERSION}_Texas_top_pairs_cross_cluster.jpg", max(5, args.visual_limit // 2), True)

    reused_report = pd.DataFrame()
    if not args.skip_reused_validation:
        reused_report = validate_reused_species(metadata, args, reports_dir)
        reused_report.to_csv(reports_dir / f"{VERSION}_reused_species_2025style_validation.csv", index=False)

    variants = ["split_strict", "merge_ultra", "splitmerge_guarded", "splitmerge_swing"]
    candidate_rows: list[dict] = []
    texas_variant_rows: list[dict] = []
    texas_current = {it.image_id: it.current_cluster for it in texas_items}
    for variant in variants:
        texas_labels = texas_variant_labels(texas_items, texas_pair_scores, variant)
        texas_variant_rows.append(summarize_texas_variant(texas_labels, texas_pair_scores, variant, texas_current))
        out_path = sub_dir / f"submission_{VERSION}_{variant}.csv"
        sub = build_submission(current, test_rows, texas_labels, out_path)
        candidate_rows.extend(summarize_submission(sub, current, test_rows, variant))
        print(f"wrote {out_path}")

    texas_variant_report = pd.DataFrame(texas_variant_rows)
    candidate_report = pd.DataFrame(candidate_rows)
    texas_variant_report.to_csv(reports_dir / f"{VERSION}_texas_variant_report.csv", index=False)
    candidate_report.to_csv(reports_dir / f"{VERSION}_candidate_report.csv", index=False)

    summary = {
        "version": VERSION,
        "data_root": str(data_root),
        "sam_manifest": str(sam_manifest) if sam_manifest else None,
        "current_best": str(current_best_path),
        "manifest_info": manifest_info,
        "outputs": {
            "reports_dir": str(reports_dir),
            "submissions_dir": str(sub_dir),
            "visualizations_dir": str(viz_dir),
        },
        "texas_items": int(len(texas_items)),
        "texas_pairs": int(len(texas_pair_scores)),
        "texas_variant_report": texas_variant_report.to_dict("records"),
        "candidate_report": candidate_report.to_dict("records"),
        "reused_validation": reused_report.to_dict("records") if not reused_report.empty else [],
    }
    (reports_dir / f"{VERSION}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTexas variant report:")
    print(texas_variant_report.to_string(index=False))
    print("\nCandidate report:")
    print(candidate_report.to_string(index=False))
    if not reused_report.empty:
        print("\nReused-species 2025-style validation:")
        print(reused_report.to_string(index=False))


if __name__ == "__main__":
    main()
