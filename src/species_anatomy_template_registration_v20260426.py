
import argparse
import itertools
import json
import math
import random
import sys
from collections import deque
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


VERSION = "species_anatomy_template_registration_v20260426"
SEED = 20260426
BACKGROUND = np.array([238, 238, 232], dtype=np.uint8)


@dataclass(frozen=True)
class AnatomyConfig:
    species: str
    target_size: tuple[int, int]
    template_kind: str
    top_k: int
    grid: tuple[int, int]
    strict_thr: float
    balanced_thr: float
    aggressive_thr: float
    force_thr: float
    min_overlap: float
    shift_px: tuple[int, ...]


CONFIGS: dict[str, AnatomyConfig] = {
    "LynxID2025": AnatomyConfig(
        species="LynxID2025",
        target_size=(288, 168),
        template_kind="lynx_flank_spot_template",
        top_k=58,
        grid=(6, 12),
        strict_thr=0.54,
        balanced_thr=0.49,
        aggressive_thr=0.45,
        force_thr=0.40,
        min_overlap=0.11,
        shift_px=(-10, 0, 10),
    ),
    "SalamanderID2025": AnatomyConfig(
        species="SalamanderID2025",
        target_size=(192, 64),
        template_kind="salamander_centerline_unwrap",
        top_k=65,
        grid=(4, 18),
        strict_thr=0.56,
        balanced_thr=0.50,
        aggressive_thr=0.46,
        force_thr=0.42,
        min_overlap=0.14,
        shift_px=(-8, 0, 8),
    ),
    "SeaTurtleID2022": AnatomyConfig(
        species="SeaTurtleID2022",
        target_size=(224, 224),
        template_kind="turtle_head_scute_template",
        top_k=40,
        grid=(8, 8),
        strict_thr=0.58,
        balanced_thr=0.52,
        aggressive_thr=0.47,
        force_thr=0.42,
        min_overlap=0.14,
        shift_px=(-10, 0, 10),
    ),
    "TexasHornedLizards": AnatomyConfig(
        species="TexasHornedLizards",
        target_size=(224, 320),
        template_kind="texas_belly_dot_template",
        top_k=75,
        grid=(10, 8),
        strict_thr=0.50,
        balanced_thr=0.46,
        aggressive_thr=0.42,
        force_thr=0.38,
        min_overlap=0.12,
        shift_px=(-12, 0, 12),
    ),
}


@dataclass
class AnatomyItem:
    image_id: int
    row_idx: int
    identity: str
    split: str
    species: str
    orientation: str
    source_path: str
    sam_view_path: str
    view_path: str
    mask_path: str
    view_source: str
    quality: float
    template_rgb: np.ndarray
    template_mask: np.ndarray
    pattern: np.ndarray
    vector: np.ndarray
    points: np.ndarray
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
            "AnimalCLEF2026 anatomy-template registration branch. It is not a "
            "current-best safeguard merge. It uses SAM-clean foreground views, "
            "species-specific anatomy canonicalization, occlusion-aware template "
            "overlap, train-label validation where available, and independent "
            "test clustering."
        )
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sam-manifest", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--species", nargs="*", default=core.SPECIES_ORDER)
    parser.add_argument("--profiles", nargs="*", default=["strict", "balanced", "aggressive", "force"])
    parser.add_argument("--max-side", type=int, default=760)
    parser.add_argument("--top-k-scale", type=float, default=1.0)
    parser.add_argument("--test-pair-budget-per-species", type=int, default=90000)
    parser.add_argument("--train-pair-budget-per-species", type=int, default=5000)
    parser.add_argument("--max-train-images-per-species", type=int, default=700)
    parser.add_argument("--max-test-images-per-species", type=int, default=0)
    parser.add_argument("--max-train-per-identity", type=int, default=8)
    parser.add_argument("--skip-train-validation", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=14)
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
        if "dataset" in metadata.columns:
            metadata["species_id"] = metadata["dataset"].astype(str)
        else:
            metadata["species_id"] = metadata["path"].str.replace("\\", "/", regex=False).str.split("/").str[1]
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


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    h, w = m.shape[:2]
    flood = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, holes)


def clean_mask(mask: np.ndarray, rgb_shape: tuple[int, int] | None = None) -> np.ndarray:
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    if rgb_shape is not None and m.shape[:2] != rgb_shape:
        m = cv2.resize(m, (rgb_shape[1], rgb_shape[0]), interpolation=cv2.INTER_NEAREST)
        m = np.where(m > 0, 255, 0).astype(np.uint8)
    if float(m.mean() / 255.0) < 0.01:
        return np.full(m.shape[:2], 255, dtype=np.uint8)
    h, w = m.shape[:2]
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    m = fill_holes(m)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if len(areas) else 0
    keep = np.zeros_like(m)
    min_area = max(25, int(largest * 0.035), int(h * w * 0.0015))
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == idx] = 255
    if float(keep.mean() / 255.0) < 0.01:
        biggest = 1 + int(np.argmax(areas))
        keep[labels == biggest] = 255
    return keep.astype(np.uint8)


def read_rgb_mask(row: dict, max_side: int) -> tuple[np.ndarray, np.ndarray, float]:
    rgb, mask = core.read_rgb_with_optional_mask(row.get("view_path", row.get("source_path")), row.get("mask_path"), max_side)
    if mask is None:
        mask = core.estimate_foreground_mask(rgb)
        quality = 0.86
    else:
        quality = 1.0
    mask = clean_mask(mask, rgb.shape[:2])
    coverage = float(mask.mean() / 255.0)
    if coverage < 0.015 or coverage > 0.98:
        mask = core.estimate_foreground_mask(rgb)
        mask = clean_mask(mask, rgb.shape[:2])
        quality *= 0.78
    return rgb, mask, quality


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, pad_frac: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = core.bbox_from_mask(mask, pad_frac)
    return rgb[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def pca_align_template(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int],
    major_axis: str = "horizontal",
    pad_frac: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, dict]:
    crop_rgb, crop_mask = crop_to_mask(rgb, mask, pad_frac)
    angle = core.pca_angle_degrees(crop_mask)
    if major_axis == "vertical":
        rotate_angle = 90.0 - angle
    else:
        rotate_angle = -angle
    if abs(rotate_angle) > 1.5:
        crop_rgb, crop_mask = core.rotate_bound(crop_rgb, crop_mask, rotate_angle)
        crop_rgb, crop_mask = crop_to_mask(crop_rgb, crop_mask, 0.04)
    w, h = target_size
    out_rgb = cv2.resize(crop_rgb, (w, h), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    out_mask = clean_mask(out_mask)
    return out_rgb, out_mask, {"pca_angle": float(angle), "rotate_angle": float(rotate_angle)}


def skeletonize(mask: np.ndarray) -> np.ndarray:
    img = np.where(mask > 0, 255, 0).astype(np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(700):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def bfs_farthest(start: int, neighbors: list[list[int]]) -> tuple[int, np.ndarray]:
    n = len(neighbors)
    parent = np.full(n, -1, dtype=np.int32)
    dist = np.full(n, -1, dtype=np.int32)
    q: deque[int] = deque([start])
    dist[start] = 0
    far = start
    while q:
        node = q.popleft()
        if dist[node] > dist[far]:
            far = node
        for nxt in neighbors[node]:
            if dist[nxt] >= 0:
                continue
            dist[nxt] = dist[node] + 1
            parent[nxt] = node
            q.append(nxt)
    return far, parent


def longest_skeleton_path(skel: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(skel > 0)
    if len(xs) < 12:
        return None
    coords = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
    index = -np.ones(skel.shape[:2], dtype=np.int32)
    index[ys, xs] = np.arange(len(xs), dtype=np.int32)
    neighbors: list[list[int]] = [[] for _ in range(len(xs))]
    for idx, (x, y) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if 0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1]:
                    j = int(index[yy, xx])
                    if j >= 0:
                        neighbors[idx].append(j)
    degrees = np.array([len(n) for n in neighbors])
    endpoints = np.where(degrees == 1)[0]
    start = int(endpoints[0]) if len(endpoints) else 0
    a, _ = bfs_farthest(start, neighbors)
    b, parent = bfs_farthest(a, neighbors)
    path_idx = [b]
    cur = b
    while cur != a and cur >= 0:
        cur = int(parent[cur])
        if cur >= 0:
            path_idx.append(cur)
        if len(path_idx) > len(coords):
            break
    if len(path_idx) < 8:
        return None
    path = coords[np.array(path_idx[::-1], dtype=np.int32)].astype(np.float32)
    return path


def resample_polyline(path: np.ndarray, n_points: int) -> np.ndarray:
    if path is None or len(path) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
    diffs = np.diff(path, axis=0)
    seg = np.sqrt((diffs * diffs).sum(axis=1))
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total <= 1e-6:
        return np.repeat(path[:1], n_points, axis=0).astype(np.float32)
    target = np.linspace(0.0, total, n_points)
    x = np.interp(target, dist, path[:, 0])
    y = np.interp(target, dist, path[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def unwrap_centerline_strip(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    crop_rgb, crop_mask = crop_to_mask(rgb, mask, 0.08)
    h0, w0 = crop_mask.shape[:2]
    scale = min(1.0, 380.0 / float(max(h0, w0)))
    small_mask = cv2.resize(
        crop_mask,
        (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    small_mask = clean_mask(small_mask)
    skel = skeletonize(small_mask)
    path_small = longest_skeleton_path(skel)
    if path_small is None:
        aligned_rgb, aligned_mask, debug = pca_align_template(crop_rgb, crop_mask, target_size, "horizontal", 0.05)
        debug["unwrap_fallback"] = True
        return aligned_rgb, aligned_mask, debug

    path = path_small / max(scale, 1e-6)
    strip_w, strip_h = target_size
    path = resample_polyline(path, strip_w)
    dt = cv2.distanceTransform(np.where(crop_mask > 0, 1, 0).astype(np.uint8), cv2.DIST_L2, 5)
    hw = []
    for x, y in path:
        xi = int(np.clip(round(float(x)), 0, w0 - 1))
        yi = int(np.clip(round(float(y)), 0, h0 - 1))
        hw.append(float(dt[yi, xi]))
    hw_arr = np.asarray(hw, dtype=np.float32)
    valid = hw_arr[hw_arr > 1.0]
    median_hw = float(np.median(valid)) if valid.size else max(4.0, min(h0, w0) * 0.08)
    hw_arr = np.clip(hw_arr * 1.30, max(3.0, median_hw * 0.38), max(6.0, median_hw * 1.95))

    tangent = np.gradient(path, axis=0)
    norm = np.sqrt((tangent * tangent).sum(axis=1, keepdims=True))
    tangent = tangent / np.maximum(norm, 1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offsets = np.linspace(-1.15, 1.15, strip_h, dtype=np.float32)
    map_x = np.zeros((strip_h, strip_w), dtype=np.float32)
    map_y = np.zeros((strip_h, strip_w), dtype=np.float32)
    for col in range(strip_w):
        p = path[col]
        n = normal[col]
        pts = p[None, :] + offsets[:, None] * hw_arr[col] * n[None, :]
        map_x[:, col] = pts[:, 0]
        map_y[:, col] = pts[:, 1]

    strip_rgb = cv2.remap(
        crop_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(238, 238, 232),
    )
    strip_mask = cv2.remap(
        crop_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    strip_mask = clean_mask(strip_mask)
    debug = {
        "unwrap_fallback": False,
        "path_points": int(len(path_small)),
        "median_half_width": float(median_hw),
        "small_scale": float(scale),
    }
    return strip_rgb, strip_mask, debug


def clahe_u8(channel: np.ndarray) -> np.ndarray:
    channel = np.clip(channel, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(channel)


def pattern_channels(rgb: np.ndarray, mask: np.ndarray, species: str) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    l = clahe_u8(lab[:, :, 0])
    sat = clahe_u8(hsv[:, :, 1])
    b = clahe_u8(lab[:, :, 2])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
    if species == "SalamanderID2025":
        yellow = cv2.addWeighted(b, 0.62, sat, 0.38, 0).astype(np.float32) / 255.0
        dark = (255.0 - l.astype(np.float32)) / 255.0
        pat = np.stack([yellow, dark, edges], axis=2)
    elif species in {"LynxID2025", "TexasHornedLizards"}:
        dark = (255.0 - l.astype(np.float32)) / 255.0
        blackhat = cv2.morphologyEx(l, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        spot = cv2.addWeighted((blackhat.astype(np.float32) / 255.0), 0.55, dark, 0.45, 0)
        texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        texture = np.clip(np.abs(texture) / 95.0, 0, 1)
        pat = np.stack([spot, dark, texture], axis=2)
    else:
        a = clahe_u8(lab[:, :, 1]).astype(np.float32) / 255.0
        bb = b.astype(np.float32) / 255.0
        ll = l.astype(np.float32) / 255.0
        pat = np.stack([ll, a, 0.5 * bb + 0.5 * edges], axis=2)
    valid = (mask > 0).astype(np.float32)
    pat *= valid[:, :, None]
    return pat.astype(np.float32)


def apply_species_template_mask(mask: np.ndarray, species: str) -> np.ndarray:
    out = np.where(mask > 0, 255, 0).astype(np.uint8)
    h, w = out.shape[:2]
    if species == "TexasHornedLizards":
        ellipse = np.zeros_like(out)
        cv2.ellipse(
            ellipse,
            (w // 2, int(h * 0.54)),
            (int(w * 0.40), int(h * 0.42)),
            0,
            0,
            360,
            255,
            -1,
        )
        out = cv2.bitwise_and(out, ellipse)
    elif species == "LynxID2025":
        flank = np.zeros_like(out)
        flank[int(h * 0.10) : int(h * 0.90), int(w * 0.06) : int(w * 0.96)] = 255
        out = cv2.bitwise_and(out, flank)
    elif species == "SalamanderID2025":
        band = np.zeros_like(out)
        band[int(h * 0.08) : int(h * 0.92), :] = 255
        out = cv2.bitwise_and(out, band)
    return out


def grid_descriptor(pattern: np.ndarray, mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    gy, gx = grid
    h, w, c = pattern.shape
    feats: list[float] = []
    valid = mask > 0
    for iy in range(gy):
        y1 = int(round(iy * h / gy))
        y2 = int(round((iy + 1) * h / gy))
        for ix in range(gx):
            x1 = int(round(ix * w / gx))
            x2 = int(round((ix + 1) * w / gx))
            cell_mask = valid[y1:y2, x1:x2]
            coverage = float(cell_mask.mean()) if cell_mask.size else 0.0
            feats.append(coverage)
            if coverage < 0.04:
                feats.extend([0.0] * (c * 3))
                continue
            vals = pattern[y1:y2, x1:x2][cell_mask]
            feats.extend(vals.mean(axis=0).astype(float).tolist())
            feats.extend(vals.std(axis=0).astype(float).tolist())
            feats.extend(np.percentile(vals, 85, axis=0).astype(float).tolist())
    weighted = pattern * valid[:, :, None].astype(np.float32)
    denom_y = np.maximum(1.0, valid.sum(axis=0).astype(np.float32))
    denom_x = np.maximum(1.0, valid.sum(axis=1).astype(np.float32))
    proj_x = weighted[:, :, 0].sum(axis=0) / denom_y
    proj_y = weighted[:, :, 0].sum(axis=1) / denom_x
    proj_x = cv2.resize(proj_x.reshape(1, -1), (48, 1), interpolation=cv2.INTER_AREA).reshape(-1)
    proj_y = cv2.resize(proj_y.reshape(-1, 1), (1, 32), interpolation=cv2.INTER_AREA).reshape(-1)
    feats.extend(proj_x.astype(float).tolist())
    feats.extend(proj_y.astype(float).tolist())
    return normalize(np.asarray(feats, dtype=np.float32))


def detect_pattern_points(pattern: np.ndarray, mask: np.ndarray, species: str) -> np.ndarray:
    heat = pattern[:, :, 0].copy()
    valid = mask > 0
    if valid.mean() < 0.025:
        return np.zeros((0, 4), dtype=np.float32)
    vals = heat[valid]
    if vals.size < 32:
        return np.zeros((0, 4), dtype=np.float32)
    pct = 82.0 if species in {"LynxID2025", "TexasHornedLizards"} else 88.0
    thr = max(float(np.percentile(vals, pct)), float(vals.mean() + 0.55 * vals.std()))
    binary = np.zeros(heat.shape, dtype=np.uint8)
    binary[(heat >= thr) & valid] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    h, w = heat.shape[:2]
    total = float(h * w)
    pts: list[list[float]] = []
    min_area = max(2.0, total * (0.00006 if species != "SeaTurtleID2022" else 0.00012))
    max_area = total * (0.020 if species != "SeaTurtleID2022" else 0.045)
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = max(1.0, float(stats[idx, cv2.CC_STAT_WIDTH]))
        bh = max(1.0, float(stats[idx, cv2.CC_STAT_HEIGHT]))
        if max(bw / bh, bh / bw) > 7.5:
            continue
        cx, cy = centroids[idx]
        strength = float(heat[labels == idx].mean())
        pts.append([float(cx) / max(1, w - 1), float(cy) / max(1, h - 1), area / total, strength])
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    pts_arr = np.asarray(pts, dtype=np.float32)
    order = np.argsort(-pts_arr[:, 3])
    return pts_arr[order[:220]]


def build_template(row: dict, cfg: AnatomyConfig, max_side: int) -> AnatomyItem:
    rgb, mask, quality = read_rgb_mask(row, max_side)
    debug: dict = {}
    if cfg.species == "SalamanderID2025":
        template_rgb, template_mask, debug = unwrap_centerline_strip(rgb, mask, cfg.target_size)
    elif cfg.species == "TexasHornedLizards":
        template_rgb, template_mask, debug = pca_align_template(rgb, mask, cfg.target_size, "vertical", 0.08)
    elif cfg.species == "LynxID2025":
        template_rgb, template_mask, debug = pca_align_template(rgb, mask, cfg.target_size, "horizontal", 0.08)
    else:
        template_rgb, template_mask, debug = pca_align_template(rgb, mask, cfg.target_size, "horizontal", 0.04)
    template_mask = apply_species_template_mask(template_mask, cfg.species)
    if float(template_mask.mean() / 255.0) < 0.025:
        template_mask = np.where(cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY) >= 0, 255, 0).astype(np.uint8)
        quality *= 0.70
    pattern = pattern_channels(template_rgb, template_mask, cfg.species)
    vector = grid_descriptor(pattern, template_mask, cfg.grid)
    points = detect_pattern_points(pattern, template_mask, cfg.species)
    quality *= min(1.0, 0.65 + 0.35 * float(template_mask.mean() / 255.0) / max(0.08, cfg.min_overlap))
    return AnatomyItem(
        image_id=int(row["image_id"]),
        row_idx=int(row.get("row_idx", row["image_id"])),
        identity=str(row.get("identity", "") or ""),
        split=str(row.get("split", "")),
        species=cfg.species,
        orientation=str(row.get("orientation", "unknown") or "unknown").lower(),
        source_path=str(row.get("source_path", "")),
        sam_view_path=str(row.get("sam_view_path", "") or ""),
        view_path=str(row.get("view_path", row.get("source_path", ""))),
        mask_path=str(row.get("mask_path", "") or ""),
        view_source=str(row.get("view_source", "")),
        quality=float(quality),
        template_rgb=template_rgb,
        template_mask=template_mask,
        pattern=pattern,
        vector=vector,
        points=points,
        debug=debug,
    )


def transform_item_arrays(
    pattern: np.ndarray,
    mask: np.ndarray,
    points: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pat = pattern
    m = mask
    pts = points.copy()
    if transform in {"rev_x", "rev_xy"}:
        pat = pat[:, ::-1, :]
        m = m[:, ::-1]
        if len(pts):
            pts[:, 0] = 1.0 - pts[:, 0]
    if transform in {"rev_y", "rev_xy"}:
        pat = pat[::-1, :, :]
        m = m[::-1, :]
        if len(pts):
            pts[:, 1] = 1.0 - pts[:, 1]
    return pat, m, pts


def shifted_slices(shape: tuple[int, int], dx: int, dy: int):
    h, w = shape
    xa1 = max(0, dx)
    xb1 = max(0, -dx)
    ya1 = max(0, dy)
    yb1 = max(0, -dy)
    ww = w - abs(dx)
    hh = h - abs(dy)
    if ww <= 4 or hh <= 4:
        return None
    return (slice(ya1, ya1 + hh), slice(xa1, xa1 + ww)), (slice(yb1, yb1 + hh), slice(xb1, xb1 + ww))


def masked_template_similarity(
    pat_a: np.ndarray,
    mask_a: np.ndarray,
    pat_b: np.ndarray,
    mask_b: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[float, float]:
    slices = shifted_slices(mask_a.shape[:2], dx, dy)
    if slices is None:
        return 0.0, 0.0
    sa, sb = slices
    ma = mask_a[sa] > 0
    mb = mask_b[sb] > 0
    overlap_mask = ma & mb
    overlap = float(overlap_mask.mean()) if overlap_mask.size else 0.0
    if overlap < 0.025 or int(overlap_mask.sum()) < 30:
        return 0.0, overlap
    sub_a = pat_a[sa]
    sub_b = pat_b[sb]
    comp_a = 0.66 * sub_a[:, :, 0] + 0.24 * sub_a[:, :, 1] + 0.10 * sub_a[:, :, 2]
    comp_b = 0.66 * sub_b[:, :, 0] + 0.24 * sub_b[:, :, 1] + 0.10 * sub_b[:, :, 2]
    vals_a = comp_a[overlap_mask].astype(np.float32)
    vals_b = comp_b[overlap_mask].astype(np.float32)
    if vals_a.size == 0:
        return 0.0, overlap
    am = vals_a - float(vals_a.mean())
    bm = vals_b - float(vals_b.mean())
    denom = float(np.linalg.norm(am) * np.linalg.norm(bm))
    corr = float(np.dot(am, bm) / denom) if denom > 1e-6 else 0.0
    l1 = float(1.0 - np.mean(np.abs(vals_a - vals_b)))
    score = 0.62 * ((corr + 1.0) * 0.5) + 0.38 * np.clip(l1, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0)), overlap


def point_chamfer_score(points_a: np.ndarray, points_b: np.ndarray, dx_norm: float, dy_norm: float) -> float:
    if len(points_a) < 3 or len(points_b) < 3:
        return 0.0
    a = points_a[:, :2].astype(np.float32)
    b = points_b[:, :2].astype(np.float32).copy()
    b[:, 0] += dx_norm
    b[:, 1] += dy_norm
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.maximum(0.0, (diff * diff).sum(axis=2)))
    da = dist.min(axis=1)
    db = dist.min(axis=0)
    keep_a = np.argsort(da)[: min(len(da), 80)]
    keep_b = np.argsort(db)[: min(len(db), 80)]
    mean_d = 0.5 * (float(da[keep_a].mean()) + float(db[keep_b].mean()))
    return float(np.exp(-mean_d / 0.055))


def allowed_transforms(a: AnatomyItem, b: AnatomyItem) -> list[str]:
    if a.species == "SalamanderID2025":
        return ["identity", "rev_x"]
    if a.species == "TexasHornedLizards":
        return ["identity", "rev_y", "rev_xy"]
    if a.species == "SeaTurtleID2022":
        return ["identity", "rev_x", "rev_y"]
    if a.species == "LynxID2025":
        sides = {"left", "right"}
        if a.orientation in sides and b.orientation in sides and a.orientation != b.orientation:
            return ["identity"]
        if a.orientation == "unknown" or b.orientation == "unknown":
            return ["identity", "rev_x"]
    return ["identity"]


def orientation_penalty(a: AnatomyItem, b: AnatomyItem) -> tuple[float, str]:
    if a.species != "LynxID2025":
        return 1.0, "not_applicable"
    sides = {"left", "right"}
    if a.orientation in sides and b.orientation in sides and a.orientation != b.orientation:
        return 0.55, "lynx_opposite_flank"
    if a.orientation in {"front", "back"} or b.orientation in {"front", "back"}:
        return 0.84, "lynx_weak_pose"
    return 1.0, "lynx_same_or_unknown"


def template_match_score(a: AnatomyItem, b: AnatomyItem, cfg: AnatomyConfig) -> dict:
    penalty, orient_rule = orientation_penalty(a, b)
    desc_cos = float(np.dot(a.vector, b.vector))
    best = {
        "map_score": 0.0,
        "overlap": 0.0,
        "point_score": 0.0,
        "transform": "none",
        "dx": 0,
        "dy": 0,
    }
    h, w = a.template_mask.shape[:2]
    shifts = cfg.shift_px
    if cfg.species == "SalamanderID2025":
        shift_pairs = [(dx, dy) for dx in shifts for dy in (-4, 0, 4)]
    else:
        shift_pairs = [(dx, dy) for dx in shifts for dy in shifts]
    for transform in allowed_transforms(a, b):
        pat_b, mask_b, pts_b = transform_item_arrays(b.pattern, b.template_mask, b.points, transform)
        for dx, dy in shift_pairs:
            map_score, overlap = masked_template_similarity(a.pattern, a.template_mask, pat_b, mask_b, dx, dy)
            if overlap < 0.025:
                continue
            point_score = point_chamfer_score(a.points, pts_b, dx / max(1, w - 1), dy / max(1, h - 1))
            if cfg.species == "SalamanderID2025":
                fused = 0.86 * map_score + 0.14 * max(0.0, desc_cos)
            elif cfg.species in {"LynxID2025", "TexasHornedLizards"}:
                fused = 0.58 * map_score + 0.30 * point_score + 0.12 * max(0.0, desc_cos)
            else:
                fused = 0.72 * map_score + 0.10 * point_score + 0.18 * max(0.0, desc_cos)
            # Reward meaningful overlap, but still allow corrupted/missing body parts
            # when the visible fingerprint region matches strongly.
            overlap_factor = min(1.0, 0.70 + 0.30 * overlap / max(0.08, cfg.min_overlap))
            fused = float(fused * overlap_factor * penalty * min(a.quality, b.quality))
            if fused > best.get("score", -1.0):
                best = {
                    "score": fused,
                    "map_score": float(map_score),
                    "overlap": float(overlap),
                    "point_score": float(point_score),
                    "transform": transform,
                    "dx": int(dx),
                    "dy": int(dy),
                }
    if "score" not in best:
        best["score"] = 0.0
    best["descriptor_cosine"] = desc_cos
    best["orientation_rule"] = orient_rule
    best["quality_a"] = float(a.quality)
    best["quality_b"] = float(b.quality)
    best["points_a"] = int(len(a.points))
    best["points_b"] = int(len(b.points))
    return best


def shortlist_pairs(items: list[AnatomyItem], cfg: AnatomyConfig, top_k_scale: float, pair_budget: int) -> list[tuple[int, int]]:
    if len(items) < 2:
        return []
    vectors = np.stack([it.vector for it in items]).astype(np.float32)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)
    ids = [it.image_id for it in items]
    item_by_id = {it.image_id: it for it in items}
    top_k = max(1, int(round(cfg.top_k * top_k_scale)))
    pairs: dict[tuple[int, int], float] = {}
    for i, image_id in enumerate(ids):
        k = min(top_k, len(ids) - 1)
        if k <= 0:
            continue
        idx = np.argpartition(-sim[i], kth=k - 1)[:k]
        for j in idx:
            other_id = ids[int(j)]
            if other_id == image_id:
                continue
            a, b = (image_id, other_id) if image_id < other_id else (other_id, image_id)
            ita = item_by_id[a]
            itb = item_by_id[b]
            if cfg.species == "LynxID2025":
                sides = {"left", "right"}
                if ita.orientation in sides and itb.orientation in sides and ita.orientation != itb.orientation:
                    if float(sim[i, int(j)]) < 0.78:
                        continue
            pairs[(a, b)] = max(pairs.get((a, b), -1.0), float(sim[i, int(j)]))
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])
    if pair_budget and len(ranked) > pair_budget:
        ranked = ranked[:pair_budget]
    return [p for p, _ in ranked]


def extract_items(rows: pd.DataFrame, cfg: AnatomyConfig, args: argparse.Namespace, split_name: str) -> tuple[list[AnatomyItem], dict]:
    records = rows.sort_values("image_id").to_dict("records")
    limit_attr = "max_test_images_per_species" if split_name == "test" else "max_train_images_per_species"
    limit = int(getattr(args, limit_attr, 0) or 0)
    if limit > 0 and len(records) > limit:
        records = records[:limit]
    if args.smoke:
        records = records[: min(len(records), 20 if split_name == "test" else 28)]
    items: list[AnatomyItem] = []
    failures = 0
    for idx, row in enumerate(records, start=1):
        try:
            items.append(build_template(row, cfg, args.max_side))
        except Exception as exc:
            failures += 1
            w, h = cfg.target_size
            blank_rgb = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)
            blank_mask = np.zeros((h, w), dtype=np.uint8)
            blank_pattern = np.zeros((h, w, 3), dtype=np.float32)
            items.append(
                AnatomyItem(
                    image_id=int(row["image_id"]),
                    row_idx=int(row.get("row_idx", row["image_id"])),
                    identity=str(row.get("identity", "") or ""),
                    split=str(row.get("split", split_name)),
                    species=cfg.species,
                    orientation=str(row.get("orientation", "unknown") or "unknown").lower(),
                    source_path=str(row.get("source_path", "")),
                    sam_view_path=str(row.get("sam_view_path", "") or ""),
                    view_path=str(row.get("view_path", row.get("source_path", ""))),
                    mask_path=str(row.get("mask_path", "") or ""),
                    view_source="failed",
                    quality=0.0,
                    template_rgb=blank_rgb,
                    template_mask=blank_mask,
                    pattern=blank_pattern,
                    vector=np.zeros_like(grid_descriptor(blank_pattern, blank_mask, cfg.grid)),
                    points=np.zeros((0, 4), dtype=np.float32),
                    debug={"error": str(exc)},
                )
            )
            print(f"[warn] {cfg.species} {split_name}: failed image_id={row.get('image_id')}: {exc}")
        if idx % 100 == 0:
            print(f"[{cfg.species} {split_name}] templates {idx}/{len(records)}")
    info = {
        "species": cfg.species,
        "split": split_name,
        "n_items": int(len(items)),
        "failures": int(failures),
        "sam_clean_primary": int(sum(it.view_source == "sam_clean_primary" for it in items)),
        "mean_quality": float(np.mean([it.quality for it in items])) if items else 0.0,
        "median_points": float(np.median([len(it.points) for it in items])) if items else 0.0,
        "mean_template_coverage": float(np.mean([it.template_mask.mean() / 255.0 for it in items])) if items else 0.0,
    }
    return items, info


def score_item_pairs(
    species: str,
    items: list[AnatomyItem],
    pairs: list[tuple[int, int]],
    cfg: AnatomyConfig,
) -> pd.DataFrame:
    by_id = {it.image_id: it for it in items}
    rows = []
    for idx, (a_id, b_id) in enumerate(pairs, start=1):
        a = by_id.get(int(a_id))
        b = by_id.get(int(b_id))
        if a is None or b is None:
            continue
        score = template_match_score(a, b, cfg)
        same_identity = bool(a.identity and b.identity and a.identity == b.identity)
        rows.append(
            {
                "species": species,
                "image_id_a": int(a_id),
                "image_id_b": int(b_id),
                "identity_a": a.identity,
                "identity_b": b.identity,
                "same_identity": same_identity,
                "orientation_a": a.orientation,
                "orientation_b": b.orientation,
                **score,
            }
        )
        if idx % 5000 == 0:
            print(f"[{species}] scored {idx}/{len(pairs)} template pairs")
    return pd.DataFrame(rows)


def sample_train_rows(rows: pd.DataFrame, max_images: int, max_per_identity: int) -> pd.DataFrame:
    labeled = rows[rows["identity"].astype(str).str.len().gt(0)].copy()
    if labeled.empty:
        return labeled
    rng = random.Random(SEED)
    chosen = []
    for _, group in labeled.groupby("identity"):
        recs = group.sort_values("image_id").to_dict("records")
        if len(recs) > max_per_identity:
            recs = rng.sample(recs, max_per_identity)
        chosen.extend(recs)
    chosen.sort(key=lambda r: int(r["image_id"]))
    if max_images > 0 and len(chosen) > max_images:
        chosen = rng.sample(chosen, max_images)
        chosen.sort(key=lambda r: int(r["image_id"]))
    return pd.DataFrame(chosen)


def sample_validation_pairs(items: list[AnatomyItem], pair_budget: int) -> list[tuple[int, int]]:
    rng = random.Random(SEED)
    by_identity: dict[str, list[int]] = {}
    ids = []
    identities = {}
    for it in items:
        if not it.identity:
            continue
        by_identity.setdefault(it.identity, []).append(it.image_id)
        ids.append(it.image_id)
        identities[it.image_id] = it.identity
    positives = []
    for members in by_identity.values():
        if len(members) < 2:
            continue
        combos = list(itertools.combinations(sorted(members), 2))
        if len(combos) > 10:
            combos = rng.sample(combos, 10)
        positives.extend(combos)
    if len(positives) > pair_budget // 2:
        positives = rng.sample(positives, max(1, pair_budget // 2))
    negatives = set()
    target_neg = min(pair_budget - len(positives), max(len(positives) * 2, 500 if positives else pair_budget))
    attempts = 0
    while len(negatives) < target_neg and attempts < target_neg * 25 and len(ids) >= 2:
        a, b = rng.sample(ids, 2)
        attempts += 1
        if identities.get(a) == identities.get(b):
            continue
        key = (a, b) if a < b else (b, a)
        negatives.add(key)
    pairs = list(positives) + sorted(negatives)
    if len(pairs) > pair_budget:
        pairs = rng.sample(pairs, pair_budget)
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


def validation_summary(pair_scores: pd.DataFrame, cfg: AnatomyConfig) -> dict:
    if pair_scores.empty or "same_identity" not in pair_scores.columns:
        return {"species": cfg.species, "auc": float("nan"), "n_pairs": int(len(pair_scores))}
    y = pair_scores["same_identity"].astype(bool).to_numpy()
    scores = pair_scores["score"].astype(float).to_numpy()
    summary = {
        "species": cfg.species,
        "n_pairs": int(len(pair_scores)),
        "n_positive": int(y.sum()),
        "n_negative": int((~y).sum()),
        "auc": auc_rank(y, scores),
    }
    for profile in ["strict", "balanced", "aggressive", "force"]:
        thr = getattr(cfg, f"{profile}_thr")
        pred = (scores >= thr) & (pair_scores["overlap"].astype(float).to_numpy() >= cfg.min_overlap)
        tp = int((pred & y).sum())
        fp = int((pred & (~y)).sum())
        fn = int(((~pred) & y).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        summary[f"{profile}_thr"] = float(thr)
        summary[f"{profile}_accepted"] = int(pred.sum())
        summary[f"{profile}_precision"] = float(precision)
        summary[f"{profile}_recall"] = float(recall)
    return summary


def threshold_for_profile(cfg: AnatomyConfig, profile: str) -> float:
    if profile not in {"strict", "balanced", "aggressive", "force"}:
        raise ValueError(f"Unknown profile: {profile}")
    return float(getattr(cfg, f"{profile}_thr"))


def accepted_pair_rows(pair_scores: pd.DataFrame, cfg: AnatomyConfig, profile: str) -> pd.DataFrame:
    if pair_scores.empty:
        return pair_scores.copy()
    thr = threshold_for_profile(cfg, profile)
    g = pair_scores.copy()
    ok = (g["score"].astype(float) >= thr) & (g["overlap"].astype(float) >= cfg.min_overlap)
    if cfg.species == "LynxID2025" and profile != "force":
        ok &= ~g["orientation_rule"].astype(str).eq("lynx_opposite_flank")
    g = g[ok].sort_values("score", ascending=False).copy()
    return g


def cluster_from_pairs(items: list[AnatomyItem], pair_scores: pd.DataFrame, cfg: AnatomyConfig, profile: str) -> dict[int, str]:
    ids = [it.image_id for it in items]
    uf = UnionFind(ids)
    accepted = accepted_pair_rows(pair_scores, cfg, profile)
    for row in accepted.itertuples(index=False):
        uf.union(int(row.image_id_a), int(row.image_id_b))
    comp_order: dict[int, int] = {}
    labels: dict[int, str] = {}
    for image_id in sorted(ids):
        comp = uf.find(image_id)
        if comp not in comp_order:
            comp_order[comp] = len(comp_order)
        labels[image_id] = f"cluster_{cfg.species}_anatomy_{profile}_{comp_order[comp]}"
    return labels


def summarize_labels(labels: dict[int, str], cfg: AnatomyConfig, profile: str, pair_scores: pd.DataFrame) -> dict:
    counts = pd.Series(list(labels.values())).value_counts()
    accepted = accepted_pair_rows(pair_scores, cfg, profile)
    return {
        "variant": profile,
        "species": cfg.species,
        "n_images": int(len(labels)),
        "n_clusters": int(counts.shape[0]) if not counts.empty else 0,
        "singletons": int((counts == 1).sum()) if not counts.empty else 0,
        "max_cluster_size": int(counts.max()) if not counts.empty else 0,
        "accepted_edges": int(len(accepted)),
        "mean_accepted_score": float(accepted["score"].mean()) if not accepted.empty else 0.0,
        "max_accepted_score": float(accepted["score"].max()) if not accepted.empty else 0.0,
    }


def save_submission(
    sample_submission: pd.DataFrame,
    labels_by_species: dict[str, dict[int, str]],
    profile: str,
    output_path: Path,
) -> pd.DataFrame:
    label_map: dict[int, str] = {}
    for labels in labels_by_species.values():
        label_map.update(labels)
    sub = sample_submission.copy()
    sub["image_id"] = sub["image_id"].astype(int)
    sub["cluster"] = sub["image_id"].map(lambda i: label_map.get(int(i), f"cluster_missing_{int(i)}"))
    sub.to_csv(output_path, index=False)
    return sub


def thumb(path: str, size: tuple[int, int]) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (25, 25, 25))
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (20, 20, 20))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def pattern_to_image(pattern: np.ndarray, mask: np.ndarray) -> Image.Image:
    pat = np.clip(pattern.copy(), 0, 1)
    if pat.shape[2] >= 3:
        arr = (pat[:, :, :3] * 255).astype(np.uint8)
    else:
        arr = np.repeat((pat[:, :, :1] * 255).astype(np.uint8), 3, axis=2)
    arr[mask == 0] = BACKGROUND
    return Image.fromarray(arr)


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, points: np.ndarray) -> Image.Image:
    canvas = rgb.copy()
    dim = (canvas.astype(np.float32) * 0.45 + BACKGROUND.astype(np.float32) * 0.55).astype(np.uint8)
    canvas[mask == 0] = dim[mask == 0]
    contours, _ = cv2.findContours(np.where(mask > 0, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (40, 255, 80), 2)
    h, w = mask.shape[:2]
    for p in points[:140]:
        x = int(round(float(p[0]) * max(1, w - 1)))
        y = int(round(float(p[1]) * max(1, h - 1)))
        cv2.circle(canvas, (x, y), 2, (255, 30, 30), 1, cv2.LINE_AA)
    return Image.fromarray(canvas)


def save_template_preview(items: list[AnatomyItem], out_path: Path, limit: int) -> None:
    chosen = items[:limit]
    if not chosen:
        return
    tile_w, tile_h = 210, 136
    label_h = 22
    cols = 5
    canvas = Image.new("RGB", (cols * tile_w, len(chosen) * (tile_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for r, item in enumerate(chosen):
        y = r * (tile_h + label_h)
        sam_path = item.sam_view_path or item.view_path
        panels = [
            thumb(item.source_path, (tile_w, tile_h)),
            thumb(sam_path, (tile_w, tile_h)),
            mask_overlay(item.template_rgb, item.template_mask, item.points),
            Image.fromarray(item.template_rgb),
            pattern_to_image(item.pattern, item.template_mask),
        ]
        labels = [
            f"orig {item.image_id}",
            item.view_source[:22],
            "template mask + points",
            "canonical anatomy",
            "pattern map",
        ]
        for c, panel in enumerate(panels):
            panel = panel.copy()
            panel.thumbnail((tile_w, tile_h), Image.Resampling.BILINEAR)
            x = c * tile_w
            draw.text((x + 5, y + 4), labels[c], fill=(245, 240, 145))
            canvas.paste(panel, (x + (tile_w - panel.width) // 2, y + label_h + (tile_h - panel.height) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def save_pair_preview(
    items: list[AnatomyItem],
    pair_scores: pd.DataFrame,
    cfg: AnatomyConfig,
    profile: str,
    out_path: Path,
    limit: int,
) -> None:
    accepted = accepted_pair_rows(pair_scores, cfg, profile).head(limit)
    if accepted.empty:
        return
    by_id = {it.image_id: it for it in items}
    panel_w, panel_h = 920, 260
    rows = []
    for edge in accepted.to_dict("records"):
        a = by_id.get(int(edge["image_id_a"]))
        b = by_id.get(int(edge["image_id_b"]))
        if a is None or b is None:
            continue
        pa = pattern_to_image(a.pattern, a.template_mask)
        pb = pattern_to_image(b.pattern, b.template_mask)
        pa.thumbnail((440, 205), Image.Resampling.BILINEAR)
        pb.thumbnail((440, 205), Image.Resampling.BILINEAR)
        row_img = Image.new("RGB", (panel_w, panel_h), (18, 18, 18))
        draw = ImageDraw.Draw(row_img)
        text = (
            f"{cfg.species} {profile}: {a.image_id} vs {b.image_id} "
            f"score={float(edge['score']):.3f} map={float(edge['map_score']):.3f} "
            f"pts={float(edge['point_score']):.3f} overlap={float(edge['overlap']):.3f} "
            f"{edge['transform']} dx={edge['dx']} dy={edge['dy']}"
        )
        draw.text((6, 5), text, fill=(255, 240, 120))
        row_img.paste(pa, (8, 42))
        row_img.paste(pb, (470, 42))
        rows.append(row_img)
    if not rows:
        return
    canvas = Image.new("RGB", (panel_w, panel_h * len(rows)), (18, 18, 18))
    for idx, row_img in enumerate(rows):
        canvas.paste(row_img, (0, idx * panel_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def update_experiment_log(output_root: Path, summary: dict) -> None:
    log_path = Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423\experiment_log.json")
    if not log_path.exists():
        return
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return
    entry = {
        "run_id": VERSION,
        "date": "2026-04-26",
        "status": "notebook_built_ready_for_kaggle",
        "goal": "Independent species-specific anatomy-template registration: SAM-clean foreground, canonical body maps, occlusion-aware fingerprint overlap, and template-space clustering.",
        "notebook": "notebooks/AnimalCLEF2026_Anatomy_Template_Registration_v20260426.ipynb",
        "output_root": str(output_root),
        "summary": summary,
    }
    if isinstance(log, dict):
        runs = log.setdefault("runs", [])
        log["runs"] = [r for r in runs if not (isinstance(r, dict) and r.get("run_id") == VERSION)]
        log["runs"].append(entry)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    args = parse_args()
    if args.smoke:
        args.max_train_images_per_species = min(args.max_train_images_per_species, 40)
        args.max_test_images_per_species = args.max_test_images_per_species or 18
        args.train_pair_budget_per_species = min(args.train_pair_budget_per_species, 400)
        args.test_pair_budget_per_species = min(args.test_pair_budget_per_species, 650)
        args.top_k_scale = min(args.top_k_scale, 0.35)
        args.save_visualizations = True

    data_root = find_data_root(args.data_root)
    sam_manifest = find_sam_manifest(args.sam_manifest)
    metadata, manifest_info = prepare_metadata(data_root, sam_manifest)
    metadata = metadata[metadata["species_id"].isin(core.SPECIES_ORDER)].copy()
    output_root = Path(args.output_root) if args.output_root else Path.cwd() / f"animalclef_{VERSION}"
    reports_dir = output_root / "reports"
    submissions_dir = output_root / "submissions"
    visual_dir = output_root / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        visual_dir.mkdir(parents=True, exist_ok=True)

    selected_species = [s for s in args.species if s in CONFIGS]
    sample_submission = pd.read_csv(data_root / "sample_submission.csv")
    sample_submission["image_id"] = sample_submission["image_id"].astype(int)
    test_rows_all = metadata[metadata["split"].eq("test")].copy()
    train_rows_all = metadata[metadata["split"].eq("train")].copy()

    print(f"VERSION={VERSION}")
    print(f"data_root={data_root}")
    print(f"sam_manifest={sam_manifest}")
    print(f"output_root={output_root}")
    print(json.dumps(manifest_info, indent=2))

    test_items_by_species: dict[str, list[AnatomyItem]] = {}
    test_pairs_by_species: dict[str, pd.DataFrame] = {}
    extraction_infos: list[dict] = []
    validation_rows: list[dict] = []

    for species in selected_species:
        cfg = CONFIGS[species]
        print(f"\n=== {species}: test anatomy templates ===")
        test_rows = test_rows_all[test_rows_all["species_id"].eq(species)].copy()
        test_items, info = extract_items(test_rows, cfg, args, "test")
        extraction_infos.append(info)
        test_items_by_species[species] = test_items
        pairs = shortlist_pairs(test_items, cfg, args.top_k_scale, args.test_pair_budget_per_species)
        print(f"[{species}] shortlisted {len(pairs)} test pairs")
        pair_scores = score_item_pairs(species, test_items, pairs, cfg)
        test_pairs_by_species[species] = pair_scores
        pair_scores.to_csv(reports_dir / f"{VERSION}_{species}_test_pair_scores.csv", index=False)
        if args.save_visualizations:
            save_template_preview(test_items, visual_dir / f"{VERSION}_{species}_template_preview.jpg", args.visual_limit)
            for profile in args.profiles:
                save_pair_preview(
                    test_items,
                    pair_scores,
                    cfg,
                    profile,
                    visual_dir / f"{VERSION}_{species}_{profile}_top_matches.jpg",
                    max(3, args.visual_limit // 2),
                )

        if args.skip_train_validation or species == "TexasHornedLizards":
            continue
        print(f"\n=== {species}: train-label validation ===")
        train_rows = train_rows_all[train_rows_all["species_id"].eq(species)].copy()
        train_rows = sample_train_rows(train_rows, args.max_train_images_per_species, args.max_train_per_identity)
        if train_rows.empty:
            continue
        train_items, train_info = extract_items(train_rows, cfg, args, "train")
        extraction_infos.append(train_info)
        val_pairs = sample_validation_pairs(train_items, args.train_pair_budget_per_species)
        print(f"[{species}] validation pairs {len(val_pairs)}")
        val_scores = score_item_pairs(species, train_items, val_pairs, cfg)
        val_scores.to_csv(reports_dir / f"{VERSION}_{species}_train_validation_pairs.csv", index=False)
        validation_rows.append(validation_summary(val_scores, cfg))

    all_test_pairs = pd.concat([df for df in test_pairs_by_species.values() if not df.empty], ignore_index=True) if test_pairs_by_species else pd.DataFrame()
    if not all_test_pairs.empty:
        all_test_pairs.to_csv(reports_dir / f"{VERSION}_all_test_pair_scores.csv", index=False)

    labels_by_profile: dict[str, dict[str, dict[int, str]]] = {profile: {} for profile in args.profiles}
    candidate_rows: list[dict] = []
    for profile in args.profiles:
        for species in selected_species:
            cfg = CONFIGS[species]
            items = test_items_by_species.get(species, [])
            pair_scores = test_pairs_by_species.get(species, pd.DataFrame())
            labels = cluster_from_pairs(items, pair_scores, cfg, profile)
            labels_by_profile[profile][species] = labels
            candidate_rows.append(summarize_labels(labels, cfg, profile, pair_scores))
        sub_path = submissions_dir / f"submission_{VERSION}_{profile}.csv"
        save_submission(sample_submission, labels_by_profile[profile], profile, sub_path)
        print(f"wrote {sub_path}")

    extraction_report = pd.DataFrame(extraction_infos)
    validation_report = pd.DataFrame(validation_rows)
    candidate_report = pd.DataFrame(candidate_rows)
    extraction_report.to_csv(reports_dir / f"{VERSION}_extraction_report.csv", index=False)
    validation_report.to_csv(reports_dir / f"{VERSION}_train_validation_report.csv", index=False)
    candidate_report.to_csv(reports_dir / f"{VERSION}_candidate_report.csv", index=False)

    summary = {
        "version": VERSION,
        "data_root": str(data_root),
        "sam_manifest": str(sam_manifest) if sam_manifest else None,
        "manifest_info": manifest_info,
        "selected_species": selected_species,
        "profiles": args.profiles,
        "outputs": {
            "reports_dir": str(reports_dir),
            "submissions_dir": str(submissions_dir),
            "visualizations_dir": str(visual_dir),
        },
        "extraction_report": extraction_report.to_dict("records"),
        "validation_report": validation_report.to_dict("records"),
        "candidate_report": candidate_report.to_dict("records"),
    }
    (reports_dir / f"{VERSION}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_experiment_log(output_root, summary)
    print("\nCandidate report:")
    print(candidate_report.to_string(index=False))
    if not validation_report.empty:
        print("\nTrain validation report:")
        print(validation_report.to_string(index=False))


if __name__ == "__main__":
    main()
