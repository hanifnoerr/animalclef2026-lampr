
import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

VERSION = "species_fingerprint_final_swing_v20260426"
SEED = 20260426
BACKGROUND_RGB = np.array([238, 238, 232], dtype=np.float32)

SPECIES_ORDER = [
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
]

CURRENT_BEST_FILENAME = "submission_hybrid_v3rescue_p06_salamander_turtle_v20260425.csv"


@dataclass(frozen=True)
class EdgeRule:
    name: str
    min_inliers: int
    min_inlier_ratio: float
    min_score: float
    allow_lynx_opposite_side: bool = False


@dataclass(frozen=True)
class SpeciesConfig:
    species: str
    roi_kind: str
    target_size: tuple[int, int]
    top_k: int
    max_keypoints: int
    ratio_test: float
    ransac_reproj: float
    split_large_at: int
    preserve_clusters_up_to: int
    max_cluster_pair_size: int
    conservative: EdgeRule
    balanced: EdgeRule
    swing: EdgeRule


SPECIES_CONFIGS: dict[str, SpeciesConfig] = {
    "LynxID2025": SpeciesConfig(
        species="LynxID2025",
        roi_kind="lynx_flank_spots",
        target_size=(448, 288),
        top_k=34,
        max_keypoints=900,
        ratio_test=0.78,
        ransac_reproj=5.0,
        split_large_at=42,
        preserve_clusters_up_to=18,
        max_cluster_pair_size=90,
        conservative=EdgeRule("conservative", min_inliers=24, min_inlier_ratio=0.22, min_score=0.42),
        balanced=EdgeRule("balanced", min_inliers=15, min_inlier_ratio=0.16, min_score=0.30),
        swing=EdgeRule(
            "swing",
            min_inliers=9,
            min_inlier_ratio=0.11,
            min_score=0.20,
            allow_lynx_opposite_side=False,
        ),
    ),
    "SalamanderID2025": SpeciesConfig(
        species="SalamanderID2025",
        roi_kind="salamander_straight_strip",
        target_size=(544, 176),
        top_k=30,
        max_keypoints=850,
        ratio_test=0.80,
        ransac_reproj=5.5,
        split_large_at=9999,
        preserve_clusters_up_to=9999,
        max_cluster_pair_size=40,
        conservative=EdgeRule("conservative", min_inliers=18, min_inlier_ratio=0.18, min_score=0.36),
        balanced=EdgeRule("balanced", min_inliers=11, min_inlier_ratio=0.13, min_score=0.25),
        swing=EdgeRule("swing", min_inliers=7, min_inlier_ratio=0.09, min_score=0.18),
    ),
    "SeaTurtleID2022": SpeciesConfig(
        species="SeaTurtleID2022",
        roi_kind="turtle_head_scutes",
        target_size=(416, 416),
        top_k=22,
        max_keypoints=900,
        ratio_test=0.76,
        ransac_reproj=4.5,
        split_large_at=9999,
        preserve_clusters_up_to=9999,
        max_cluster_pair_size=35,
        conservative=EdgeRule("conservative", min_inliers=28, min_inlier_ratio=0.24, min_score=0.47),
        balanced=EdgeRule("balanced", min_inliers=22, min_inlier_ratio=0.20, min_score=0.39),
        swing=EdgeRule("swing", min_inliers=17, min_inlier_ratio=0.15, min_score=0.32),
    ),
    "TexasHornedLizards": SpeciesConfig(
        species="TexasHornedLizards",
        roi_kind="texas_ventral_dots",
        target_size=(448, 320),
        top_k=74,
        max_keypoints=900,
        ratio_test=0.78,
        ransac_reproj=5.0,
        split_large_at=20,
        preserve_clusters_up_to=10,
        max_cluster_pair_size=60,
        conservative=EdgeRule("conservative", min_inliers=18, min_inlier_ratio=0.26, min_score=0.34),
        balanced=EdgeRule("balanced", min_inliers=12, min_inlier_ratio=0.20, min_score=0.25),
        swing=EdgeRule("swing", min_inliers=8, min_inlier_ratio=0.16, min_score=0.18),
    ),
}


OPTIONAL_SUBMISSION_FILENAMES = {
    "current_best": CURRENT_BEST_FILENAME,
    "arwildfusion_v3_rescue_v2": "submission_sam_arwildfusion_v3_rescue_lbshape_v2.csv",
    "p06_miewid_mega": "submission_p06_miewid_plus_mega_l384.csv",
    "dualfoundation_species_hybrid": "submission_species_hybrid_v20260425.csv",
    "dualfoundation_selected": "submission_selected_specieswise_hybrid.csv",
    "specieswise_best_localari": "submission_specieswise_best_localari_hybrid_v20260425.csv",
}


class UnionFind:
    def __init__(self, values: Iterable):
        self.parent = {v: v for v in values}
        self.size = {v: 1 for v in values}

    def find(self, value):
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

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


@dataclass
class PatternItem:
    image_id: int
    row_idx: int
    species: str
    orientation: str
    view_path: str
    view_source: str
    quality: float
    keypoints: np.ndarray
    descriptors: np.ndarray | None
    vector: np.ndarray
    fallback_vector: np.ndarray
    part_vectors: np.ndarray
    part_visibility: np.ndarray
    fallback_part_vectors: np.ndarray
    fallback_part_visibility: np.ndarray
    visibility_coverage: float
    n_keypoints: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final-swing species-specific fingerprint ensemble for AnimalCLEF2026. "
            "Uses saved SAM views when present, extracts different local pattern ROIs "
            "per species, verifies shortlisted pairs geometrically, and writes "
            "conservative/balanced/high-swing hybrid submissions over the current best."
        )
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--sam-manifest", type=str, default=None)
    parser.add_argument("--current-best-submission", type=str, default=None)
    parser.add_argument("--extra-submission-root", action="append", default=[])
    parser.add_argument("--species", nargs="*", default=SPECIES_ORDER)
    parser.add_argument("--max-images-per-species", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=820)
    parser.add_argument("--top-k-scale", type=float, default=1.0)
    parser.add_argument("--pair-budget-per-species", type=int, default=65000)
    parser.add_argument("--save-roi-preview", action="store_true")
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
        if base.exists():
            try:
                matches = list(base.rglob("view_manifest_sam3_all_species.csv"))
            except Exception:
                matches = []
            if matches:
                return matches[0].resolve()
    return None


def find_file_everywhere(filename: str, roots: Iterable[Path]) -> Path | None:
    for root in roots:
        candidate = root / filename
        if candidate.exists():
            return candidate.resolve()
    for root in roots:
        if not root.exists():
            continue
        try:
            matches = list(root.rglob(filename))
        except Exception:
            matches = []
        if matches:
            matches.sort(key=lambda p: len(str(p)))
            return matches[0].resolve()
    return None


def find_submission_sources(args: argparse.Namespace, data_root: Path) -> dict[str, Path]:
    roots: list[Path] = [
        Path.cwd(),
        Path.cwd().parent,
        data_root.parent,
        Path("/kaggle/input"),
    ]
    roots.extend(Path(p) for p in args.extra_submission_root)
    local_project = Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026")
    roots.extend(
        [
            local_project / "current_wildfusion_graph_v20260423",
            local_project / "AnimalCLEF2026 SAM ARWildFusion v3_update",
            local_project / "AnimalCLEF2026 SAM ARWildFusion v3_update" / "submissions",
            local_project / "archive" / "AnimalCLEF2026 v4 Backbone Sweep + Non-EfficientNet Candidates",
            local_project / "Masked Dual-Foundation Search v20260425 output",
        ]
    )
    roots = [p for p in roots if p is not None]

    found: dict[str, Path] = {}
    if args.current_best_submission:
        p = Path(args.current_best_submission)
        if p.exists():
            found["current_best"] = p.resolve()

    for label, filename in OPTIONAL_SUBMISSION_FILENAMES.items():
        if label in found:
            continue
        path = find_file_everywhere(filename, roots)
        if path is not None:
            found[label] = path

    if "current_best" not in found:
        raise FileNotFoundError(f"Could not find {CURRENT_BEST_FILENAME}.")
    return found


def export_root_from_manifest(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "reports":
        return manifest_path.parent.parent
    return manifest_path.parent


def remap_export_path(path_value, export_root: Path | None) -> Path | None:
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
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            if marker != "animalclef_sam3_views_cache/":
                rel = marker + rel
            candidate = export_root / Path(rel)
            if candidate.exists():
                return candidate.resolve()
    return None


def prepare_test_metadata(data_root: Path, sam_manifest: Path | None) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_csv(data_root / "metadata.csv").reset_index(drop=True)
    if "row_idx" not in metadata.columns:
        metadata["row_idx"] = np.arange(len(metadata), dtype=np.int64)
    if "dataset" in metadata.columns:
        metadata["species_id"] = metadata["dataset"].astype(str)
    else:
        metadata["species_id"] = metadata["path"].str.replace("\\", "/", regex=False).str.split("/").str[1]
    if "split" not in metadata.columns:
        metadata["split"] = np.where(metadata["path"].str.contains("/test/"), "test", "train")
    if "orientation" not in metadata.columns:
        metadata["orientation"] = "unknown"
    metadata["source_path"] = metadata["path"].map(lambda p: str(data_root / str(p)))

    test = metadata[metadata["split"].eq("test")].copy()
    test["view_path"] = test["source_path"]
    test["view_source"] = "original"
    test["mask_path"] = ""
    test["mask_ok"] = False

    info = {"manifest_path": str(sam_manifest) if sam_manifest else None, "manifest_rows": 0, "resolved_views": 0}
    if sam_manifest is None:
        return test, info

    manifest = pd.read_csv(sam_manifest)
    info["manifest_rows"] = int(len(manifest))
    export_root = export_root_from_manifest(sam_manifest)
    merge_key = "row_idx" if "row_idx" in manifest.columns else "image_id" if "image_id" in manifest.columns else None
    if merge_key is None:
        return test, info
    merged = test.merge(manifest, on=merge_key, how="left", suffixes=("", "_sam"))

    view_paths: list[str] = []
    sam_view_paths: list[str] = []
    mask_paths: list[str] = []
    view_sources: list[str] = []
    mask_ok: list[bool] = []
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
        # Corrected assumption: species is known and each image contains one
        # individual, so the original image remains the pattern source. SAM is
        # a guide for animal/body visibility, not the image to compare.
        view_paths.append(str(Path(row["source_path"])))
        sam_view_paths.append(str(resolved_sam_view) if resolved_sam_view else "")
        view_sources.append("original_sam_guided" if resolved_mask is not None else "original")
        mask_paths.append(str(resolved_mask) if resolved_mask else "")
        mask_ok.append(resolved_mask is not None)
    merged["view_path"] = view_paths
    merged["sam_view_path"] = sam_view_paths
    merged["view_source"] = view_sources
    merged["mask_path"] = mask_paths
    merged["mask_ok"] = mask_ok
    info["resolved_views"] = int(sum(mask_ok))
    return merged, info


def read_rgb(path: str | Path, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BILINEAR)
    return np.asarray(img)


def read_rgb_with_optional_mask(
    image_path: str | Path,
    mask_path: str | Path | None,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask_img = None
    if mask_path:
        p = Path(mask_path)
        if p.exists():
            try:
                mask_img = Image.open(p).convert("L")
                if mask_img.size != img.size:
                    mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)
            except Exception:
                mask_img = None
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
        if mask_img is not None:
            mask_img = mask_img.resize(new_size, Image.Resampling.NEAREST)
    rgb = np.asarray(img)
    mask = None
    if mask_img is not None:
        mask = np.where(np.asarray(mask_img) > 127, 255, 0).astype(np.uint8)
        if float(mask.mean() / 255.0) < 0.01:
            mask = None
    return rgb, mask


def estimate_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    border = np.concatenate(
        [
            arr[: max(2, h // 40), :, :].reshape(-1, 3),
            arr[-max(2, h // 40) :, :, :].reshape(-1, 3),
            arr[:, : max(2, w // 40), :].reshape(-1, 3),
            arr[:, -max(2, w // 40) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_rgb = np.median(border, axis=0)
    diff_border = np.linalg.norm(arr - border_rgb[None, None, :], axis=2)
    diff_export_bg = np.linalg.norm(arr - BACKGROUND_RGB[None, None, :], axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    mask = ((diff_export_bg > 20) & (diff_border > 8)) | ((diff_border > 28) & (sat > 18))
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    coverage = float(mask.mean() / 255.0)
    if coverage < 0.04 or coverage > 0.96:
        mask = np.full((h, w), 255, dtype=np.uint8)
    return mask


def bbox_from_mask(mask: np.ndarray, pad_frac: float = 0.05) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if len(xs) == 0:
        return 0, 0, w, h
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x2 - x1, y2 - y1) * pad_frac))
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, pad_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox_from_mask(mask, pad_frac)
    return rgb[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def pca_angle_degrees(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return 0.0
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    vec = vecs[:, int(np.argmax(vals))]
    return float(math.degrees(math.atan2(vec[1], vec[0])))


def rotate_bound(rgb: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rgb_rot = cv2.warpAffine(
        rgb,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(238, 238, 232),
    )
    mask_rot = cv2.warpAffine(
        mask,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rgb_rot, mask_rot


def align_and_crop(rgb: np.ndarray, mask: np.ndarray, do_pca: bool = True) -> tuple[np.ndarray, np.ndarray]:
    crop_rgb, crop_mask = crop_to_mask(rgb, mask, 0.08)
    if do_pca:
        angle = pca_angle_degrees(crop_mask)
        if abs(angle) > 2:
            crop_rgb, crop_mask = rotate_bound(crop_rgb, crop_mask, -angle)
            crop_rgb, crop_mask = crop_to_mask(crop_rgb, crop_mask, 0.05)
    return crop_rgb, crop_mask


def slice_roi(rgb: np.ndarray, mask: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    x1 = int(round(w * box[0]))
    y1 = int(round(h * box[1]))
    x2 = int(round(w * box[2]))
    y2 = int(round(h * box[3]))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    return rgb[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def resize_roi(rgb: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_resized = np.where(mask_resized > 0, 255, 0).astype(np.uint8)
    return rgb_resized, mask_resized


def species_roi(
    rgb: np.ndarray,
    species: str,
    orientation: str,
    config: SpeciesConfig,
    mask_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    mask = mask_override if mask_override is not None else estimate_foreground_mask(rgb)
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    quality = 1.0
    orientation = (orientation or "unknown").lower()
    if config.roi_kind == "turtle_head_scutes":
        aligned_rgb, aligned_mask = align_and_crop(rgb, mask, do_pca=False)
        roi_rgb, roi_mask = slice_roi(aligned_rgb, aligned_mask, (0.04, 0.02, 0.96, 0.96))
    else:
        aligned_rgb, aligned_mask = align_and_crop(rgb, mask, do_pca=True)
        if config.roi_kind == "lynx_flank_spots":
            if orientation in {"front", "back", "unknown", ""}:
                quality *= 0.72
            roi_rgb, roi_mask = slice_roi(aligned_rgb, aligned_mask, (0.08, 0.14, 0.94, 0.88))
        elif config.roi_kind == "salamander_straight_strip":
            roi_rgb, roi_mask = slice_roi(aligned_rgb, aligned_mask, (0.03, 0.10, 0.97, 0.90))
        elif config.roi_kind == "texas_ventral_dots":
            roi_rgb, roi_mask = slice_roi(aligned_rgb, aligned_mask, (0.10, 0.08, 0.90, 0.94))
        else:
            roi_rgb, roi_mask = aligned_rgb, aligned_mask

    roi_rgb, roi_mask = resize_roi(roi_rgb, roi_mask, config.target_size)
    if config.roi_kind == "texas_ventral_dots":
        h, w = roi_mask.shape[:2]
        ellipse = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            ellipse,
            (w // 2, int(h * 0.54)),
            (int(w * 0.38), int(h * 0.38)),
            0,
            0,
            360,
            255,
            -1,
        )
        roi_mask = cv2.bitwise_and(roi_mask, ellipse)
    if float(roi_mask.mean() / 255.0) < 0.05:
        roi_mask = np.full(roi_mask.shape, 255, dtype=np.uint8)
        quality *= 0.70
    return roi_rgb, roi_mask, quality


def pattern_gray(rgb: np.ndarray, species: str) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    l = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    if species in {"LynxID2025", "TexasHornedLizards"}:
        dark = 255 - l_eq
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        blackhat = cv2.morphologyEx(l_eq, cv2.MORPH_BLACKHAT, kernel)
        gray = cv2.addWeighted(dark, 0.62, blackhat, 0.38, 0)
    elif species == "SalamanderID2025":
        yellow = lab[:, :, 2]
        sat = hsv[:, :, 1]
        gray = cv2.addWeighted(clahe.apply(yellow), 0.55, clahe.apply(sat), 0.45, 0)
    else:
        a = clahe.apply(lab[:, :, 1])
        b = clahe.apply(lab[:, :, 2])
        gray = cv2.addWeighted(l_eq, 0.60, cv2.addWeighted(a, 0.5, b, 0.5, 0), 0.40, 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.uint8)


def create_detector(max_keypoints: int):
    try:
        return "sift", cv2.SIFT_create(nfeatures=max_keypoints, contrastThreshold=0.015, edgeThreshold=12)
    except Exception:
        return "orb", cv2.ORB_create(nfeatures=max_keypoints, fastThreshold=7)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def part_grid(config: SpeciesConfig) -> tuple[int, int]:
    if config.roi_kind == "salamander_straight_strip":
        return 3, 10
    if config.roi_kind == "turtle_head_scutes":
        return 4, 4
    if config.roi_kind == "texas_ventral_dots":
        return 5, 7
    return 4, 7


def neutralize_missing(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return gray
    valid = mask > 0
    if valid.mean() < 0.02:
        return gray
    neutral = gray.copy()
    fill = int(np.median(gray[valid]))
    neutral[~valid] = fill
    return neutral


def compute_vector(rgb: np.ndarray, gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    gray_for_thumb = neutralize_missing(gray, mask)
    small = cv2.resize(gray_for_thumb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist_parts = []
    for channel, bins, limit in [(0, 24, 180), (1, 16, 256), (2, 16, 256)]:
        hist_mask = mask if mask is not None and float(mask.mean()) > 0 else None
        hist = cv2.calcHist([hsv], [channel], hist_mask, [bins], [0, limit]).astype(np.float32).reshape(-1)
        hist /= max(1e-6, float(hist.sum()))
        hist_parts.append(hist)
    vec = np.concatenate([small, *hist_parts]).astype(np.float32)
    return normalize_vector(vec)


def compute_part_vectors(
    rgb: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    config: SpeciesConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    rows, cols = part_grid(config)
    h, w = gray.shape[:2]
    vectors: list[np.ndarray] = []
    visibility: list[bool] = []
    total_valid = float((mask > 0).mean())
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    for gy in range(rows):
        y1 = int(round(h * gy / rows))
        y2 = int(round(h * (gy + 1) / rows))
        for gx in range(cols):
            x1 = int(round(w * gx / cols))
            x2 = int(round(w * (gx + 1) / cols))
            cell_mask = mask[y1:y2, x1:x2]
            cell_gray = gray[y1:y2, x1:x2]
            cell_rgb = rgb[y1:y2, x1:x2]
            cell_hsv = hsv[y1:y2, x1:x2]
            coverage = float((cell_mask > 0).mean()) if cell_mask.size else 0.0
            visible = coverage >= 0.12
            visibility.append(visible)
            if not visible:
                vectors.append(np.zeros(64 + 32, dtype=np.float32))
                continue
            valid = cell_mask > 0
            cell_neutral = cell_gray.copy()
            cell_neutral[~valid] = int(np.median(cell_gray[valid]))
            thumb = cv2.resize(cell_neutral, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
            hist_parts = []
            for channel, bins, limit in [(0, 12, 180), (1, 10, 256), (2, 10, 256)]:
                hist = cv2.calcHist([cell_hsv], [channel], cell_mask, [bins], [0, limit]).astype(np.float32).reshape(-1)
                hist /= max(1e-6, float(hist.sum()))
                hist_parts.append(hist)
            vectors.append(normalize_vector(np.concatenate([thumb, *hist_parts]).astype(np.float32)))
    return np.stack(vectors).astype(np.float32), np.asarray(visibility, dtype=bool), total_valid


def part_similarity_from_arrays(
    vec_a: np.ndarray,
    vis_a: np.ndarray,
    vec_b: np.ndarray,
    vis_b: np.ndarray,
) -> tuple[float, float, int]:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0, 0.0, 0
    common = vis_a & vis_b
    n_common = int(common.sum())
    n_possible = int((vis_a | vis_b).sum())
    if n_common == 0:
        return 0.0, 0.0, 0
    sims = np.sum(vec_a[common] * vec_b[common], axis=1)
    sims = np.clip(sims, -1.0, 1.0)
    if len(sims) >= 4:
        # Partial matching: emphasize the best mutually visible zones instead
        # of punishing a crop where SAM removed a limb/body patch.
        keep = max(2, int(math.ceil(len(sims) * 0.70)))
        score = float(np.mean(np.sort(sims)[-keep:]))
    else:
        score = float(np.mean(sims))
    overlap = float(n_common / max(1, n_possible))
    return score, overlap, n_common


def occlusion_aware_similarity(a: PatternItem, b: PatternItem) -> tuple[float, float, int, float]:
    candidates = [
        part_similarity_from_arrays(a.part_vectors, a.part_visibility, b.part_vectors, b.part_visibility),
        part_similarity_from_arrays(
            a.fallback_part_vectors,
            a.fallback_part_visibility,
            b.fallback_part_vectors,
            b.fallback_part_visibility,
        ),
        part_similarity_from_arrays(a.part_vectors, a.part_visibility, b.fallback_part_vectors, b.fallback_part_visibility),
        part_similarity_from_arrays(a.fallback_part_vectors, a.fallback_part_visibility, b.part_vectors, b.part_visibility),
    ]
    part_score, part_overlap, part_cells = max(candidates, key=lambda x: (x[0], x[2]))
    global_primary = float(np.dot(a.vector, b.vector))
    global_fallback = float(np.dot(a.fallback_vector, b.fallback_vector))
    visual_sim = max(global_primary, global_fallback, part_score)
    return visual_sim, float(part_score), int(part_cells), float(part_overlap)


def extract_pattern_item(row: dict, config: SpeciesConfig, max_side: int, detector_name: str, detector) -> PatternItem:
    mask_path = str(row.get("mask_path", "")).strip()
    rgb, sam_mask = read_rgb_with_optional_mask(row["view_path"], mask_path, max_side=max_side)
    roi_rgb, roi_mask, quality = species_roi(
        rgb,
        row["species_id"],
        str(row.get("orientation", "unknown")),
        config,
        mask_override=sam_mask,
    )
    gray = pattern_gray(roi_rgb, row["species_id"])
    kps, desc = detector.detectAndCompute(gray, roi_mask)
    if kps is None:
        kps = []
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 2)
    if desc is not None and len(desc) > 0:
        if detector_name == "sift":
            desc = desc.astype(np.float32)
            desc /= np.maximum(1e-7, desc.sum(axis=1, keepdims=True))
            desc = np.sqrt(desc)
        else:
            desc = desc.astype(np.uint8)
    else:
        desc = None
    vec = compute_vector(roi_rgb, gray, roi_mask)
    part_vecs, part_vis, coverage = compute_part_vectors(roi_rgb, gray, roi_mask, config)
    fallback_vec = vec
    fallback_part_vecs = part_vecs
    fallback_part_vis = part_vis
    source_path = str(row.get("source_path", "")).strip()
    # Fallback descriptor deliberately ignores SAM if present. It helps
    # shortlist candidates when SAM masks are too conservative, while the
    # primary descriptor still uses the SAM-guided animal/body region.
    if source_path and Path(source_path).exists() and sam_mask is not None:
        try:
            original_rgb = read_rgb(source_path, max_side=max_side)
            original_roi_rgb, original_roi_mask, original_quality = species_roi(
                original_rgb,
                row["species_id"],
                str(row.get("orientation", "unknown")),
                config,
                mask_override=None,
            )
            original_gray = pattern_gray(original_roi_rgb, row["species_id"])
            fallback_vec = compute_vector(original_roi_rgb, original_gray, original_roi_mask)
            fallback_part_vecs, fallback_part_vis, original_coverage = compute_part_vectors(
                original_roi_rgb,
                original_gray,
                original_roi_mask,
                config,
            )
            quality = min(quality, 0.75 + 0.25 * original_quality)
            coverage = max(coverage, min(0.95, original_coverage))
        except Exception:
            fallback_vec = vec
            fallback_part_vecs = part_vecs
            fallback_part_vis = part_vis
    if coverage < 0.22 and row["species_id"] != "SeaTurtleID2022":
        quality *= 0.78
    combined_vec = normalize_vector(0.72 * vec + 0.28 * fallback_vec)
    return PatternItem(
        image_id=int(row["image_id"]),
        row_idx=int(row.get("row_idx", row["image_id"])),
        species=str(row["species_id"]),
        orientation=str(row.get("orientation", "unknown")).lower(),
        view_path=str(row["view_path"]),
        view_source=str(row.get("view_source", "original")),
        quality=float(quality),
        keypoints=pts,
        descriptors=desc,
        vector=combined_vec,
        fallback_vector=fallback_vec,
        part_vectors=part_vecs,
        part_visibility=part_vis,
        fallback_part_vectors=fallback_part_vecs,
        fallback_part_visibility=fallback_part_vis,
        visibility_coverage=float(coverage),
        n_keypoints=len(kps),
    )


def save_roi_preview(rows: list[dict], out_path: Path, config: SpeciesConfig, max_side: int, limit: int = 24) -> None:
    if not rows:
        return
    thumbs = []
    for row in rows[:limit]:
        try:
            mask_path = str(row.get("mask_path", "")).strip()
            rgb, sam_mask = read_rgb_with_optional_mask(row["view_path"], mask_path, max_side=max_side)
            roi_rgb, roi_mask, _ = species_roi(
                rgb,
                row["species_id"],
                str(row.get("orientation", "unknown")),
                config,
                mask_override=sam_mask,
            )
            overlay = roi_rgb.copy()
            overlay[roi_mask == 0] = (overlay[roi_mask == 0] * 0.35 + np.array([238, 238, 232]) * 0.65).astype(np.uint8)
            thumbs.append(Image.fromarray(overlay).resize((160, 112)))
        except Exception:
            continue
    if not thumbs:
        return
    cols = 4
    rows_n = int(math.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * 160, rows_n * 112), (238, 238, 232))
    for idx, thumb in enumerate(thumbs):
        canvas.paste(thumb, ((idx % cols) * 160, (idx // cols) * 112))
    canvas.save(out_path, quality=90)


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns.")
    df = df[["image_id", "cluster"]].copy()
    df["image_id"] = df["image_id"].astype(int)
    df["cluster"] = df["cluster"].astype(str)
    return df


def labels_for_species(submission: pd.DataFrame, species_rows: pd.DataFrame) -> dict[int, str]:
    merged = species_rows[["image_id"]].merge(submission, on="image_id", how="left")
    if merged["cluster"].isna().any():
        missing = merged.loc[merged["cluster"].isna(), "image_id"].head().tolist()
        raise ValueError(f"Submission is missing image ids: {missing}")
    return {int(r.image_id): str(r.cluster) for r in merged.itertuples(index=False)}


def pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def cluster_pair_votes(
    source_submissions: dict[str, pd.DataFrame],
    species_rows: pd.DataFrame,
    config: SpeciesConfig,
) -> dict[tuple[int, int], int]:
    votes: dict[tuple[int, int], int] = {}
    for source_name, sub in source_submissions.items():
        if source_name == "current_best":
            continue
        labels = labels_for_species(sub, species_rows)
        groups: dict[str, list[int]] = {}
        for image_id, cluster in labels.items():
            groups.setdefault(cluster, []).append(image_id)
        for members in groups.values():
            if 1 < len(members) <= config.max_cluster_pair_size:
                for a, b in itertools.combinations(sorted(members), 2):
                    key = pair_key(a, b)
                    votes[key] = votes.get(key, 0) + 1
    return votes


def current_cluster_pairs(labels: dict[int, str], max_cluster_pair_size: int) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = {}
    for image_id, cluster in labels.items():
        groups.setdefault(cluster, []).append(image_id)
    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        if 1 < len(members) <= max_cluster_pair_size:
            for a, b in itertools.combinations(sorted(members), 2):
                pairs.add(pair_key(a, b))
    return pairs


def orientation_compatible(species: str, o1: str, o2: str, allow_opposite_lynx: bool = False) -> tuple[bool, str]:
    o1 = (o1 or "unknown").lower()
    o2 = (o2 or "unknown").lower()
    if species != "LynxID2025":
        return True, "not_applicable"
    side_set = {"left", "right"}
    if o1 in side_set and o2 in side_set and o1 != o2:
        return allow_opposite_lynx, "lynx_opposite_flank"
    if o1 in {"front", "back"} or o2 in {"front", "back"}:
        return True, "lynx_weak_pose"
    return True, "lynx_same_or_unknown"


def shortlist_pairs(
    items: list[PatternItem],
    current_labels: dict[int, str],
    alt_votes: dict[tuple[int, int], int],
    config: SpeciesConfig,
    top_k_scale: float,
    pair_budget: int,
) -> set[tuple[int, int]]:
    image_ids = [it.image_id for it in items]
    item_by_id = {it.image_id: it for it in items}
    pairs = set(current_cluster_pairs(current_labels, config.max_cluster_pair_size))
    pairs.update(alt_votes.keys())

    vectors = np.stack([it.vector for it in items]).astype(np.float32)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)
    top_k = max(1, int(round(config.top_k * top_k_scale)))
    for i, image_id in enumerate(image_ids):
        k = min(top_k, len(image_ids) - 1)
        if k <= 0:
            continue
        idx = np.argpartition(-sim[i], kth=k - 1)[:k]
        for j in idx:
            other_id = image_ids[int(j)]
            if other_id == image_id:
                continue
            a, b = pair_key(image_id, other_id)
            ok, _ = orientation_compatible(
                config.species,
                item_by_id[a].orientation,
                item_by_id[b].orientation,
                allow_opposite_lynx=False,
            )
            if ok or current_labels.get(a) == current_labels.get(b):
                pairs.add((a, b))

    if len(pairs) <= pair_budget:
        return pairs

    id_to_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    priority = []
    for a, b in pairs:
        ia = id_to_idx[a]
        ib = id_to_idx[b]
        same_current = current_labels.get(a) == current_labels.get(b)
        vote = alt_votes.get((a, b), 0)
        priority.append((same_current, vote, float(sim[ia, ib]), a, b))
    priority.sort(reverse=True)
    return {pair_key(a, b) for _, _, _, a, b in priority[:pair_budget]}


def bf_matcher(detector_name: str):
    if detector_name == "sift":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def local_match_score(
    a: PatternItem,
    b: PatternItem,
    config: SpeciesConfig,
    detector_name: str,
    ratio_test: float,
) -> dict:
    visual_sim, part_score, part_cells, part_overlap = occlusion_aware_similarity(a, b)
    if a.descriptors is None or b.descriptors is None or len(a.descriptors) < 4 or len(b.descriptors) < 4:
        return {
            "inliers": 0,
            "good_matches": 0,
            "inlier_ratio": 0.0,
            "spatial_coverage": 0.0,
            "local_score": max(0.0, visual_sim) * 0.12,
            "visual_sim": visual_sim,
            "part_score": part_score,
            "part_overlap": part_overlap,
            "part_cells": part_cells,
            "coverage_a": float(a.visibility_coverage),
            "coverage_b": float(b.visibility_coverage),
        }
    matcher = bf_matcher(detector_name)
    try:
        knn = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    except Exception:
        return {
            "inliers": 0,
            "good_matches": 0,
            "inlier_ratio": 0.0,
            "spatial_coverage": 0.0,
            "local_score": max(0.0, visual_sim) * 0.12,
            "visual_sim": visual_sim,
            "part_score": part_score,
            "part_overlap": part_overlap,
            "part_cells": part_cells,
            "coverage_a": float(a.visibility_coverage),
            "coverage_b": float(b.visibility_coverage),
        }
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)
    if len(good) < 4:
        return {
            "inliers": 0,
            "good_matches": len(good),
            "inlier_ratio": 0.0,
            "spatial_coverage": 0.0,
            "local_score": max(0.0, visual_sim) * 0.12,
            "visual_sim": visual_sim,
            "part_score": part_score,
            "part_overlap": part_overlap,
            "part_cells": part_cells,
            "coverage_a": float(a.visibility_coverage),
            "coverage_b": float(b.visibility_coverage),
        }
    pts_a = np.float32([a.keypoints[m.queryIdx] for m in good]).reshape(-1, 2)
    pts_b = np.float32([b.keypoints[m.trainIdx] for m in good]).reshape(-1, 2)
    _, inlier_mask = cv2.estimateAffinePartial2D(
        pts_a,
        pts_b,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj,
        maxIters=2000,
        confidence=0.995,
    )
    if inlier_mask is None:
        inliers = 0
        inlier_flags = np.zeros(len(good), dtype=bool)
    else:
        inlier_flags = inlier_mask.reshape(-1).astype(bool)
        inliers = int(inlier_flags.sum())
    denom = max(1, min(len(a.keypoints), len(b.keypoints), len(good)))
    inlier_ratio = float(inliers / denom)
    if inliers > 1:
        pa = pts_a[inlier_flags]
        pb = pts_b[inlier_flags]
        cov_a = (float(pa[:, 0].max() - pa[:, 0].min()) * float(pa[:, 1].max() - pa[:, 1].min())) / max(
            1.0, float(config.target_size[0] * config.target_size[1])
        )
        cov_b = (float(pb[:, 0].max() - pb[:, 0].min()) * float(pb[:, 1].max() - pb[:, 1].min())) / max(
            1.0, float(config.target_size[0] * config.target_size[1])
        )
        spatial_coverage = float(min(1.0, math.sqrt(max(0.0, cov_a * cov_b)) * 4.0))
    else:
        spatial_coverage = 0.0
    inlier_term = min(1.0, inliers / max(1.0, config.conservative.min_inliers * 1.4))
    ratio_term = min(1.0, inlier_ratio / max(0.01, config.conservative.min_inlier_ratio * 1.2))
    part_term = max(0.0, min(1.0, (part_score + 0.05) / 1.05))
    overlap_term = min(1.0, part_overlap * 2.0)
    local_score = (
        0.42 * inlier_term
        + 0.24 * ratio_term
        + 0.10 * spatial_coverage
        + 0.16 * part_term
        + 0.04 * overlap_term
        + 0.04 * max(0.0, min(1.0, (visual_sim + 0.1) / 1.1))
    )
    if part_cells < 2 and inliers < max(10, config.balanced.min_inliers):
        local_score *= 0.68
    if min(a.visibility_coverage, b.visibility_coverage) < 0.12 and inliers < max(12, config.balanced.min_inliers):
        local_score *= 0.72
    local_score *= min(1.0, 0.65 + 0.35 * min(a.quality, b.quality))
    return {
        "inliers": inliers,
        "good_matches": len(good),
        "inlier_ratio": inlier_ratio,
        "spatial_coverage": spatial_coverage,
        "local_score": float(local_score),
        "visual_sim": visual_sim,
        "part_score": part_score,
        "part_overlap": part_overlap,
        "part_cells": part_cells,
        "coverage_a": float(a.visibility_coverage),
        "coverage_b": float(b.visibility_coverage),
    }


def accept_edge(row: dict, rule: EdgeRule) -> bool:
    if row["species"] == "LynxID2025" and row["orientation_rule"] == "lynx_opposite_flank" and not rule.allow_lynx_opposite_side:
        return False
    alt_vote = int(row.get("alt_votes", 0))
    min_inliers = max(4, rule.min_inliers - min(6, alt_vote * 2))
    min_ratio = max(0.05, rule.min_inlier_ratio - min(0.04, alt_vote * 0.012))
    min_score = max(0.10, rule.min_score - min(0.06, alt_vote * 0.018))
    # Occlusion-aware guard: missing SAM regions are allowed, but a merge
    # still needs either several mutually visible body zones or very strong
    # geometric evidence. This avoids treating a tiny unmasked fragment as a
    # full fingerprint match.
    if int(row.get("part_cells", 0)) < 2 and int(row["inliers"]) < (min_inliers + 4) and alt_vote == 0:
        return False
    if float(row.get("part_overlap", 0.0)) < 0.08 and int(row["inliers"]) < int(math.ceil(min_inliers * 1.5)) and alt_vote == 0:
        return False
    return (
        int(row["inliers"]) >= min_inliers
        and float(row["inlier_ratio"]) >= min_ratio
        and float(row["local_score"]) >= min_score
    )


def evaluate_pairs_for_species(
    species: str,
    species_rows: pd.DataFrame,
    source_submissions: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    reports_dir: Path,
) -> tuple[list[PatternItem], pd.DataFrame, dict]:
    config = SPECIES_CONFIGS[species]
    detector_name, detector = create_detector(config.max_keypoints)
    rows = species_rows.sort_values("image_id").to_dict("records")
    if args.max_images_per_species:
        rows = rows[: args.max_images_per_species]
    if args.smoke:
        rows = rows[: min(len(rows), 42)]

    if args.save_roi_preview:
        save_roi_preview(rows, reports_dir / f"{VERSION}_{species}_roi_preview.jpg", config, args.max_side)

    items: list[PatternItem] = []
    failures = 0
    for idx, row in enumerate(rows, start=1):
        try:
            items.append(extract_pattern_item(row, config, args.max_side, detector_name, detector))
        except Exception as exc:
            failures += 1
            fallback_vec = np.zeros(1024 + 56, dtype=np.float32)
            n_parts = part_grid(config)[0] * part_grid(config)[1]
            fallback_parts = np.zeros((n_parts, 96), dtype=np.float32)
            fallback_vis = np.zeros(n_parts, dtype=bool)
            items.append(
                PatternItem(
                    image_id=int(row["image_id"]),
                    row_idx=int(row.get("row_idx", row["image_id"])),
                    species=species,
                    orientation=str(row.get("orientation", "unknown")).lower(),
                    view_path=str(row.get("view_path", row.get("source_path", ""))),
                    view_source="failed",
                    quality=0.0,
                    keypoints=np.zeros((0, 2), dtype=np.float32),
                    descriptors=None,
                    vector=fallback_vec,
                    fallback_vector=fallback_vec,
                    part_vectors=fallback_parts,
                    part_visibility=fallback_vis,
                    fallback_part_vectors=fallback_parts,
                    fallback_part_visibility=fallback_vis,
                    visibility_coverage=0.0,
                    n_keypoints=0,
                )
            )
            print(f"[warn] {species}: feature extraction failed for image_id={row.get('image_id')}: {exc}")
        if idx % 100 == 0:
            print(f"[{species}] extracted {idx}/{len(rows)}")

    current_labels = labels_for_species(source_submissions["current_best"], species_rows)
    if args.smoke or args.max_images_per_species:
        keep_ids = {it.image_id for it in items}
        current_labels = {k: v for k, v in current_labels.items() if k in keep_ids}
    alt_votes = cluster_pair_votes(source_submissions, species_rows[species_rows["image_id"].isin(current_labels)], config)
    alt_votes = {k: v for k, v in alt_votes.items() if k[0] in current_labels and k[1] in current_labels}
    pairs = shortlist_pairs(items, current_labels, alt_votes, config, args.top_k_scale, args.pair_budget_per_species)
    item_by_id = {it.image_id: it for it in items}

    score_rows = []
    for idx, (a_id, b_id) in enumerate(sorted(pairs), start=1):
        a = item_by_id.get(a_id)
        b = item_by_id.get(b_id)
        if a is None or b is None:
            continue
        orient_ok, orient_rule = orientation_compatible(species, a.orientation, b.orientation, allow_opposite_lynx=True)
        scores = local_match_score(a, b, config, detector_name, config.ratio_test)
        if species == "LynxID2025" and orient_rule == "lynx_opposite_flank":
            scores["local_score"] *= 0.55
        same_current = current_labels.get(a_id) == current_labels.get(b_id)
        row = {
            "species": species,
            "image_id_a": a_id,
            "image_id_b": b_id,
            "orientation_a": a.orientation,
            "orientation_b": b.orientation,
            "orientation_rule": orient_rule,
            "same_current_cluster": bool(same_current),
            "alt_votes": int(alt_votes.get((a_id, b_id), 0)),
            "detector": detector_name,
            "kp_a": int(a.n_keypoints),
            "kp_b": int(b.n_keypoints),
            **scores,
        }
        row["accept_conservative"] = accept_edge(row, config.conservative)
        row["accept_balanced"] = accept_edge(row, config.balanced)
        row["accept_swing"] = accept_edge(row, config.swing)
        score_rows.append(row)
        if idx % 5000 == 0:
            print(f"[{species}] scored {idx}/{len(pairs)} pairs")

    pair_scores = pd.DataFrame(score_rows)
    info = {
        "species": species,
        "n_images": len(items),
        "feature_failures": failures,
        "detector": detector_name,
        "n_pairs_scored": int(len(pair_scores)),
        "mean_keypoints": float(np.mean([it.n_keypoints for it in items])) if items else 0.0,
        "median_keypoints": float(np.median([it.n_keypoints for it in items])) if items else 0.0,
        "mean_visibility_coverage": float(np.mean([it.visibility_coverage for it in items])) if items else 0.0,
        "median_visibility_coverage": float(np.median([it.visibility_coverage for it in items])) if items else 0.0,
        "resolved_sam_or_mask_views": int(sum(it.view_source != "original" for it in items)),
    }
    if not pair_scores.empty:
        for col in ["accept_conservative", "accept_balanced", "accept_swing"]:
            info[col] = int(pair_scores[col].sum())
        info["max_inliers"] = int(pair_scores["inliers"].max())
        info["max_local_score"] = float(pair_scores["local_score"].max())
        info["median_part_overlap"] = float(pair_scores["part_overlap"].median())
        info["max_part_score"] = float(pair_scores["part_score"].max())
    return items, pair_scores, info


def summarize_submission(submission: pd.DataFrame, test_rows: pd.DataFrame, current: pd.DataFrame, variant: str) -> list[dict]:
    current_map = dict(zip(current["image_id"].astype(int), current["cluster"].astype(str)))
    sub_map = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    rows = []
    for species in SPECIES_ORDER:
        ids = test_rows.loc[test_rows["species_id"].eq(species), "image_id"].astype(int).tolist()
        labels = [sub_map[i] for i in ids]
        counts = pd.Series(labels).value_counts()
        changed = sum(1 for i in ids if current_map.get(i) != sub_map.get(i))
        rows.append(
            {
                "variant": variant,
                "species": species,
                "n_images": len(ids),
                "n_clusters": int(counts.shape[0]),
                "max_cluster_size": int(counts.max()) if not counts.empty else 0,
                "singletons": int((counts == 1).sum()) if not counts.empty else 0,
                "rows_changed_vs_current": int(changed),
            }
        )
    return rows


def relabel_components(image_to_component: dict[int, object], species: str, variant: str) -> dict[int, str]:
    component_order: dict[object, int] = {}
    labels: dict[int, str] = {}
    for image_id in sorted(image_to_component):
        comp = image_to_component[image_id]
        if comp not in component_order:
            component_order[comp] = len(component_order)
        labels[image_id] = f"cluster_{species}_{variant}_{component_order[comp]}"
    return labels


def merge_variant_for_species(
    species: str,
    species_ids: list[int],
    current_labels: dict[int, str],
    pair_scores: pd.DataFrame,
    profile: str,
) -> dict[int, str]:
    config = SPECIES_CONFIGS[species]
    clusters = sorted({current_labels[i] for i in species_ids})
    uf = UnionFind(clusters)
    if pair_scores.empty:
        return {i: current_labels[i] for i in species_ids}
    accept_col = f"accept_{profile}"
    for row in pair_scores[pair_scores[accept_col]].itertuples(index=False):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        ca = current_labels.get(a)
        cb = current_labels.get(b)
        if ca is None or cb is None:
            continue
        if ca != cb:
            uf.union(ca, cb)
    return {i: str(uf.find(current_labels[i])) for i in species_ids}


def splitmerge_variant_for_species(
    species: str,
    species_ids: list[int],
    current_labels: dict[int, str],
    pair_scores: pd.DataFrame,
) -> dict[int, str]:
    config = SPECIES_CONFIGS[species]
    if species not in {"LynxID2025", "TexasHornedLizards"}:
        return merge_variant_for_species(species, species_ids, current_labels, pair_scores, "swing")

    uf = UnionFind(species_ids)
    groups: dict[str, list[int]] = {}
    for image_id in species_ids:
        groups.setdefault(current_labels[image_id], []).append(image_id)

    for members in groups.values():
        if len(members) <= config.preserve_clusters_up_to:
            anchor = members[0]
            for other in members[1:]:
                uf.union(anchor, other)

    if not pair_scores.empty:
        accepted = pair_scores[pair_scores["accept_swing"]].copy()
        for row in accepted.itertuples(index=False):
            a = int(row.image_id_a)
            b = int(row.image_id_b)
            if a not in current_labels or b not in current_labels:
                continue
            ca = current_labels[a]
            cb = current_labels[b]
            same_current = ca == cb
            if same_current or bool(row.accept_balanced) or int(row.alt_votes) > 0:
                uf.union(a, b)

    comp_by_id = {i: uf.find(i) for i in species_ids}
    return relabel_components(comp_by_id, species, "v7_swing_splitmerge")


def build_submission_variants(
    test_rows: pd.DataFrame,
    source_submissions: dict[str, pd.DataFrame],
    pair_scores_by_species: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    current = source_submissions["current_best"].copy()
    variants: dict[str, pd.DataFrame] = {}
    current_base = current.copy()
    variants["current_best_passthrough"] = current_base

    species_rows_by_name = {
        species: test_rows[test_rows["species_id"].eq(species)].sort_values("image_id").copy()
        for species in SPECIES_ORDER
    }
    current_labels_by_species = {
        species: labels_for_species(current, rows) for species, rows in species_rows_by_name.items()
    }

    for profile in ["conservative", "balanced", "swing"]:
        sub = current.copy()
        label_updates: dict[int, str] = {}
        for species, rows in species_rows_by_name.items():
            ids = rows["image_id"].astype(int).tolist()
            pair_scores = pair_scores_by_species.get(species, pd.DataFrame())
            # SeaTurtle is already the strongest local branch; keep it guarded in all but swing.
            effective_profile = "conservative" if species == "SeaTurtleID2022" and profile != "swing" else profile
            labels = merge_variant_for_species(
                species,
                ids,
                current_labels_by_species[species],
                pair_scores,
                effective_profile,
            )
            label_updates.update(labels)
        original_map = dict(zip(sub["image_id"].astype(int), sub["cluster"].astype(str)))
        sub["cluster"] = sub["image_id"].astype(int).map(lambda i: label_updates.get(i, original_map[i]))
        variants[f"current_plus_{profile}_pattern_merges"] = sub

    split_sub = current.copy()
    split_updates: dict[int, str] = {}
    for species, rows in species_rows_by_name.items():
        ids = rows["image_id"].astype(int).tolist()
        labels = splitmerge_variant_for_species(
            species,
            ids,
            current_labels_by_species[species],
            pair_scores_by_species.get(species, pd.DataFrame()),
        )
        split_updates.update(labels)
    split_original_map = dict(zip(split_sub["image_id"].astype(int), split_sub["cluster"].astype(str)))
    split_sub["cluster"] = split_sub["image_id"].astype(int).map(lambda i: split_updates.get(i, split_original_map[i]))
    variants["swing_split_large_lynx_texas"] = split_sub
    return variants


def update_experiment_log(output_root: Path, summary: dict, candidate_report: pd.DataFrame, best_path: Path) -> None:
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
        "status": "implemented_smoke_tested_ready_for_kaggle",
        "goal": "Final-swing species-specific local fingerprint ensemble over current 0.29758 hybrid.",
        "output_root": str(output_root),
        "recommended_first_submission": str(best_path),
        "summary": summary,
        "candidate_report_preview": candidate_report.to_dict("records")[:20],
    }
    if isinstance(log, list):
        log = [run for run in log if not (isinstance(run, dict) and run.get("run_id") == VERSION)]
        log.append(entry)
    elif isinstance(log, dict):
        runs = log.setdefault("runs", [])
        log["runs"] = [run for run in runs if not (isinstance(run, dict) and run.get("run_id") == VERSION)]
        log["runs"].append(entry)
    else:
        return
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    args = parse_args()
    if args.smoke:
        args.max_images_per_species = args.max_images_per_species or 42
        args.pair_budget_per_species = min(args.pair_budget_per_species, 2500)
        args.top_k_scale = min(args.top_k_scale, 0.55)

    data_root = find_data_root(args.data_root)
    sam_manifest = find_sam_manifest(args.sam_manifest)
    output_root = Path(args.output_root) if args.output_root else Path.cwd() / f"animalclef_{VERSION}"
    reports_dir = output_root / "reports"
    submissions_dir = output_root / "submissions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    test_rows, manifest_info = prepare_test_metadata(data_root, sam_manifest)
    selected_species = [s for s in args.species if s in SPECIES_CONFIGS]
    test_rows = test_rows[test_rows["species_id"].isin(SPECIES_ORDER)].copy()
    source_paths = find_submission_sources(args, data_root)
    source_submissions = {name: load_submission(path) for name, path in source_paths.items()}

    print(f"VERSION={VERSION}")
    print(f"data_root={data_root}")
    print(f"sam_manifest={sam_manifest}")
    print(f"output_root={output_root}")
    print("submission sources:")
    for name, path in source_paths.items():
        print(f"  {name}: {path}")

    pair_scores_by_species: dict[str, pd.DataFrame] = {}
    item_infos: list[dict] = []
    for species in selected_species:
        species_rows = test_rows[test_rows["species_id"].eq(species)].copy()
        items, pair_scores, info = evaluate_pairs_for_species(species, species_rows, source_submissions, args, reports_dir)
        pair_scores_by_species[species] = pair_scores
        item_infos.append(info)
        item_manifest = pd.DataFrame(
            [
                {
                    "species": it.species,
                    "image_id": it.image_id,
                    "row_idx": it.row_idx,
                    "orientation": it.orientation,
                    "view_path": it.view_path,
                    "view_source": it.view_source,
                    "quality": it.quality,
                    "visibility_coverage": it.visibility_coverage,
                    "visible_parts": int(it.part_visibility.sum()),
                    "fallback_visible_parts": int(it.fallback_part_visibility.sum()),
                    "n_keypoints": it.n_keypoints,
                }
                for it in items
            ]
        )
        item_manifest.to_csv(reports_dir / f"{VERSION}_{species}_item_manifest.csv", index=False)
        pair_scores.to_csv(reports_dir / f"{VERSION}_{species}_pair_scores.csv", index=False)

    nonempty_pair_scores = [df for df in pair_scores_by_species.values() if not df.empty]
    all_pair_scores = pd.concat(nonempty_pair_scores, ignore_index=True) if nonempty_pair_scores else pd.DataFrame()
    all_pair_scores.to_csv(reports_dir / f"{VERSION}_all_pair_scores.csv", index=False)

    variants = build_submission_variants(test_rows, source_submissions, pair_scores_by_species)
    current = source_submissions["current_best"]
    report_rows = []
    written_paths = {}
    for variant, sub in variants.items():
        out_path = submissions_dir / f"submission_{VERSION}_{variant}.csv"
        sub.to_csv(out_path, index=False)
        written_paths[variant] = str(out_path)
        report_rows.extend(summarize_submission(sub, test_rows, current, variant))

    candidate_report = pd.DataFrame(report_rows)
    candidate_report.to_csv(reports_dir / f"{VERSION}_candidate_report.csv", index=False)

    recommended = submissions_dir / f"submission_{VERSION}_current_plus_conservative_pattern_merges.csv"
    summary = {
        "version": VERSION,
        "data_root": str(data_root),
        "sam_manifest": str(sam_manifest) if sam_manifest else None,
        "manifest_info": manifest_info,
        "selected_species": selected_species,
        "source_paths": {k: str(v) for k, v in source_paths.items()},
        "item_infos": item_infos,
        "written_submissions": written_paths,
        "recommended_first_submission": str(recommended),
        "notes": [
            "Conservative/balanced/swing merge variants preserve the current 0.29758 hybrid and only merge clusters with local pattern evidence.",
            "Occlusion-aware visibility grids ignore SAM-removed or hand-corrupted body zones instead of treating missing parts as mismatches.",
            "swing_split_large_lynx_texas is the high-risk/high-upside file; it can repair oversized Lynx/Texas clusters but may over-split.",
            "SeaTurtle uses guarded rules because p06 is already very strong locally.",
        ],
    }
    (reports_dir / f"{VERSION}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    update_experiment_log(output_root, summary, candidate_report, recommended)

    print("candidate report:")
    print(candidate_report.to_string(index=False))
    print(f"wrote {recommended}")


if __name__ == "__main__":
    main()
