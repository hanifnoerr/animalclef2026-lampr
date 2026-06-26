
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

VERSION = "sam_yolo_speciesfusion_shapeaware_shortlabels_v20260427"
SEED = 20260427
IMG_SIZE = 512
SPECIES_ORDER = [
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
]
REID_SPECIES = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
TEXAS = "TexasHornedLizards"
NEUTRAL_GREY_RGB = np.array([128, 128, 128], dtype=np.uint8)
BLACK_RGB = np.array([0, 0, 0], dtype=np.uint8)
BACKGROUND_RGB = NEUTRAL_GREY_RGB


@dataclass
class FeatureItem:
    image_id: int
    species: str
    identity: str
    cluster: str
    view_path: str
    source_path: str
    orientation: str
    body_kind: str
    low_light: bool
    keypoints: list
    desc: np.ndarray | None
    vec: np.ndarray
    quality: float


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

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume the SAM+YOLO fused view manifest and create species-specific "
            "AnimalCLEF submission candidates from independent train-calibrated "
            "pattern graphs. Existing submissions are optional diagnostics only."
        )
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--view-manifest", type=str, default=None)
    parser.add_argument("--reference-submission", type=str, default=None, help="Optional diagnostics only; not auto-discovered.")
    parser.add_argument("--base-submission", type=str, default=None, help="Deprecated diagnostics alias for --reference-submission.")
    parser.add_argument("--output-root", type=str, default="/kaggle/working/animalclef_sam_yolo_speciesfusion_shapeaware_shortlabels_v20260427")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="Global square canvas size after aspect-ratio padding.")
    parser.add_argument("--max-side", type=int, default=760)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--pair-budget-per-species", type=int, default=85000)
    parser.add_argument("--train-images-per-species", type=int, default=850)
    parser.add_argument("--train-pairs-per-species", type=int, default=3200)
    parser.add_argument("--texas-pair-budget", type=int, default=18000)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=20)
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


def find_view_manifest(user_value: str | None) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    direct = [
        Path("/kaggle/working/animalclef_sam_yolo_views_v20260427/reports/view_manifest_sam_yolo_fused_v20260427.csv"),
        Path("/kaggle/input/animalclef2026-sam-yolo-view-factory-v20260427/animalclef_sam_yolo_views_v20260427/reports/view_manifest_sam_yolo_fused_v20260427.csv"),
        Path("/kaggle/input/animalclef2026-sam-yolo-view-factory-v20260427/reports/view_manifest_sam_yolo_fused_v20260427.csv"),
        Path.cwd() / "animalclef_sam_yolo_views_v20260427" / "reports" / "view_manifest_sam_yolo_fused_v20260427.csv",
        Path.cwd().parent / "animalclef_sam_yolo_views_v20260427" / "reports" / "view_manifest_sam_yolo_fused_v20260427.csv",
    ]
    for p in direct:
        if p.exists():
            return p.resolve()
    for base in [Path("/kaggle/input"), Path.cwd(), Path.cwd().parent, Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026")]:
        if not base.exists():
            continue
        try:
            matches = list(base.rglob("view_manifest_sam_yolo_fused_v20260427.csv"))
        except Exception:
            matches = []
        if matches:
            matches.sort(key=lambda x: len(str(x)))
            return matches[0].resolve()
    raise FileNotFoundError("Could not locate view_manifest_sam_yolo_fused_v20260427.csv.")


def find_reference_submission(user_value: str | None, data_root: Path) -> Path | None:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    return None


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns.")
    df = df[["image_id", "cluster"]].copy()
    df["image_id"] = df["image_id"].astype(int)
    df["cluster"] = df["cluster"].astype(str)
    return df


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    a = set(sub["image_id"].astype(int))
    b = set(sample["image_id"].astype(int))
    if a != b or len(sub) != len(sample):
        raise ValueError(f"Bad submission: len={len(sub)} expected={len(sample)} missing={sorted(b-a)[:5]} extra={sorted(a-b)[:5]}")
    if sub["cluster"].isna().any():
        raise ValueError("Submission contains null clusters.")


def remap_path(path_value: object, manifest_root: Path) -> str:
    if path_value is None or pd.isna(path_value):
        return ""
    s = str(path_value).strip()
    if not s:
        return ""
    p = Path(s)
    if p.exists():
        return str(p.resolve())
    normalized = s.replace("\\", "/")
    markers = ["animalclef_sam_yolo_views_v20260427/", "views/", "yolo_pred_mask/"]
    base_root = manifest_root.parent.parent if manifest_root.parent.name == "reports" else manifest_root.parent
    for marker in markers:
        if marker not in normalized:
            continue
        rel = normalized.split(marker, 1)[1]
        if marker != "animalclef_sam_yolo_views_v20260427/":
            rel = marker + rel
        candidate = base_root / Path(rel)
        if candidate.exists():
            return str(candidate.resolve())
    return s


def prepare_view_manifest(view_manifest: Path, data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(view_manifest)
    metadata = pd.read_csv(data_root / "metadata.csv")
    if "species_id" not in metadata.columns:
        metadata["species_id"] = metadata["dataset"].astype(str)
    metadata = metadata[metadata["species_id"].isin(SPECIES_ORDER)].copy()
    keep_meta = [c for c in ["image_id", "identity", "orientation", "path", "split", "dataset", "species_id"] if c in metadata.columns]
    if "identity" in df.columns:
        df = df.drop(columns=["identity"], errors="ignore")
    if "orientation" in df.columns:
        df = df.drop(columns=["orientation"], errors="ignore")
    if "split" in df.columns:
        df = df.drop(columns=["split"], errors="ignore")
    df = df.merge(metadata[keep_meta], on="image_id", how="left", suffixes=("", "_meta"))
    if "dataset" in df.columns:
        df["species_id"] = df["species_id"].fillna(df["dataset"])
    df["species_id"] = df["species_id"].astype(str)
    inferred_split = pd.Series(
        np.where(df["path"].astype(str).str.contains("/test/"), "test", "train"),
        index=df.index,
    )
    df["split"] = df["split"].fillna(inferred_split).astype(str)
    df["identity"] = df["identity"].fillna("").astype(str)
    df["orientation"] = df["orientation"].fillna("unknown").astype(str)
    if "original_path" not in df.columns or df["original_path"].isna().any():
        df["original_path"] = df["path"].map(lambda p: str(data_root / str(p)))
    for col in [
        "original_path",
        "view_original_path",
        "view_sam_clean_path",
        "view_yolo_crop_path",
        "view_sam_yolo_union_path",
        "view_sam_yolo_intersection_path",
        "view_species_final_path",
        "fused_mask_path",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(lambda x: remap_path(x, view_manifest))
    # Reused Kaggle outputs often preserve /kaggle/input/... original image
    # paths. When this notebook is run locally, or when Kaggle mounts the
    # competition dataset under a different alias, rebuild original_path from
    # metadata instead of letting Texas full-canvas masking fail.
    if "path" in df.columns:
        rebuilt_original = df["path"].map(lambda p: str(data_root / str(p)))
        missing_original = df["original_path"].fillna("").astype(str).map(lambda p: (not p) or (not Path(p).exists()))
        df.loc[missing_original, "original_path"] = rebuilt_original[missing_original]
    present = set(df["image_id"].astype(int))
    missing = metadata[~metadata["image_id"].astype(int).isin(present)].copy()
    if not missing.empty:
        # Smoke/local partial manifests should still produce a valid submission.
        # For missing rows, fall back to original images and singleton clustering.
        extra = missing[keep_meta].copy()
        extra["original_path"] = extra["path"].map(lambda p: str(data_root / str(p)))
        for col in [
            "view_original_path",
            "view_sam_clean_path",
            "view_yolo_crop_path",
            "view_sam_yolo_union_path",
            "view_sam_yolo_intersection_path",
            "view_species_final_path",
        ]:
            extra[col] = extra["original_path"]
        extra["fused_mask_path"] = ""
        extra["fusion_decision"] = "original_missing_fused_manifest"
        extra["final_view_kind"] = "original_fallback_missing_manifest"
        extra["sam_ok"] = False
        extra["yolo_ok"] = False
        extra["sam_yolo_iou"] = 0.0
        df = pd.concat([df, extra], ignore_index=True, sort=False)
    return df[df["species_id"].isin(SPECIES_ORDER)].copy()


def image_path_for_species(row: dict, species: str) -> str:
    if species == "LynxID2025":
        # Lynx camera-trap frames are often already subject-dominant and dark.
        # Use the original frame as the signal source; the fused mask is only
        # a subject guide for preprocessing.
        return str(row.get("original_path") or row.get("view_original_path") or row.get("view_species_final_path"))
    if species == "SalamanderID2025":
        # Salamander identity lives in the black/yellow pattern and the user
        # observed orientation is informative. Use SAM-clean output, but do not
        # rotate or flip it later.
        return str(row.get("view_sam_clean_path") or row.get("view_species_final_path") or row.get("view_sam_yolo_union_path"))
    if species == "SeaTurtleID2022":
        return str(row.get("view_sam_yolo_intersection_path") or row.get("view_species_final_path") or row.get("view_sam_clean_path"))
    # Texas uses the final SAM+YOLO view; the astro-dot template below enforces
    # no oval and no vertical flip.
    return str(row.get("view_species_final_path") or row.get("view_sam_yolo_union_path") or row.get("view_sam_clean_path") or row.get("original_path"))


def read_rgb(path: str, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BILINEAR)
    return np.asarray(img)


def read_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    if path and Path(path).exists():
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != shape:
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return np.where(mask > 16, 255, 0).astype(np.uint8)
    return np.full(shape, 255, dtype=np.uint8)


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray, pad_frac: float) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return rgb, mask
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = int(round(max(x1 - x0 + 1, y1 - y0 + 1) * pad_frac))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return rgb[y0 : y1 + 1, x0 : x1 + 1].copy(), mask[y0 : y1 + 1, x0 : x1 + 1].copy()


def square_pad_resize(
    rgb: np.ndarray,
    mask: np.ndarray,
    img_size: int,
    pad_rgb: np.ndarray = BLACK_RGB,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    side = max(h, w, 1)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:] = pad_rgb.reshape(1, 1, 3)
    mask_canvas = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = rgb
    mask_canvas[y0 : y0 + h, x0 : x0 + w] = mask
    out = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(mask_canvas, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return out, out_mask


def canonical_orientation(value: object) -> str:
    s = str(value or "unknown").strip().lower()
    if not s or s in {"nan", "na", "none"}:
        return "unknown"
    return s


def rotate_by_orientation(rgb: np.ndarray, species: str, orientation: str) -> np.ndarray:
    """Use metadata orientation where it is semantically reliable.

    Salamander images are often randomly rotated. The 2025 solutions explicitly
    normalized salamander orientation; doing it here lets head-only and full-body
    crops compare in a common frame. Lynx/SeaTurtle orientation is used as a
    matching rule instead of physically rotating the image.
    """
    # Final correction: do not physically rotate or flip any species here.
    # Salamander original orientation is treated as a pattern clue, not noise.
    return rgb


def enhance_lynx_low_light(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Adaptive low-light enhancement for dark Lynx camera-trap frames."""
    if rgb.size == 0:
        return rgb, False, 0.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    median_l = float(np.median(l))
    p90_l = float(np.percentile(l, 90))
    low_light = median_l < 72.0 or p90_l < 118.0
    if not low_light:
        # Still apply a mild CLAHE to make dark spots less dependent on exposure.
        enhanced = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
        lab[:, :, 0] = np.where(mask > 0, enhanced, l).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), False, median_l

    l_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    l_eq = np.where(mask > 0, l_eq, l).astype(np.uint8)
    rgb_eq = cv2.cvtColor(np.dstack([l_eq, lab[:, :, 1], lab[:, :, 2]]).astype(np.uint8), cv2.COLOR_LAB2RGB)
    # Gamma < 1 brightens the midtones without completely flattening spots.
    gamma = float(np.clip(0.58 + median_l / 180.0, 0.55, 0.92))
    lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    rgb_gamma = cv2.LUT(rgb_eq, lut)
    rgb_gamma = np.where(mask[:, :, None] > 0, rgb_gamma, rgb).astype(np.uint8)
    return rgb_gamma, True, median_l


def preprocess_rgb_for_species(
    rgb: np.ndarray,
    mask: np.ndarray,
    species: str,
    orientation: str,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    rgb = rotate_by_orientation(rgb, species, orientation)
    if species == "LynxID2025":
        rgb, low_light, light_level = enhance_lynx_low_light(rgb, mask)
        rgb, mask = crop_to_mask(rgb, mask, 0.10)
        rgb, mask = square_pad_resize(rgb, mask, img_size, BLACK_RGB)
        return rgb, mask, bool(low_light), float(light_level)
    if species == "SalamanderID2025":
        rgb = np.where(mask[:, :, None] > 0, rgb, NEUTRAL_GREY_RGB.reshape(1, 1, 3)).astype(np.uint8)
        rgb, mask = crop_to_mask(rgb, mask, 0.04)
        rgb, mask = square_pad_resize(rgb, mask, img_size, NEUTRAL_GREY_RGB)
        return rgb, mask, False, 0.0
    if species == TEXAS:
        rgb = np.where(mask[:, :, None] > 0, rgb, NEUTRAL_GREY_RGB.reshape(1, 1, 3)).astype(np.uint8)
        rgb, mask = crop_to_mask(rgb, mask, 0.04)
        rgb, mask = square_pad_resize(rgb, mask, img_size, NEUTRAL_GREY_RGB)
        return rgb, mask, False, 0.0
    # Sea Turtle keeps the existing clean-view route, with only the global
    # square-pad/resize normalization applied.
    rgb, mask = crop_to_mask(rgb, mask, 0.06)
    rgb, mask = square_pad_resize(rgb, mask, img_size, NEUTRAL_GREY_RGB)
    return rgb, mask, False, 0.0


def estimate_body_kind(rgb: np.ndarray, species: str) -> str:
    if species != "SalamanderID2025" or rgb.size == 0:
        return "body"
    diff = np.abs(rgb.astype(np.int16) - BACKGROUND_RGB[None, None, :].astype(np.int16)).sum(axis=2)
    mask = (diff > 28).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ys, xs = np.where(mask > 0)
    if len(xs) < 32:
        return "partial_or_head"
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    elongation = max(w, h) / max(1.0, min(w, h))
    coverage = float(mask.mean() / 255.0)
    if elongation < 1.55 or coverage < 0.10:
        return "partial_or_head"
    return "full_body"


def orientation_compatible(a: "FeatureItem", b: "FeatureItem") -> bool:
    if a.species != b.species:
        return False
    oa = canonical_orientation(a.orientation)
    ob = canonical_orientation(b.orientation)
    if "unknown" in {oa, ob}:
        return True
    if a.species == "LynxID2025":
        sides = {"left", "right"}
        if oa in sides and ob in sides and oa != ob:
            return False
        if {oa, ob} == {"front", "back"}:
            return False
    if a.species == "SeaTurtleID2022":
        lefts = {"left", "topleft"}
        rights = {"right", "topright"}
        if (oa in lefts and ob in rights) or (oa in rights and ob in lefts):
            return False
    return True


def pair_context_factor(a: "FeatureItem", b: "FeatureItem") -> tuple[float, str]:
    oa = canonical_orientation(a.orientation)
    ob = canonical_orientation(b.orientation)
    if not orientation_compatible(a, b):
        return 0.0, "orientation_block"
    factor = 1.0
    relation = "orientation_unknown_or_mixed"
    if oa == ob and oa != "unknown":
        relation = "same_orientation"
        if a.species in {"LynxID2025", "SeaTurtleID2022"}:
            factor *= 1.05
    elif a.species == "LynxID2025":
        relation = "weak_orientation_match"
        factor *= 0.92
    if a.species == "SalamanderID2025":
        if a.body_kind == b.body_kind:
            relation += "_same_body_kind"
            factor *= 1.04
        elif "partial_or_head" in {a.body_kind, b.body_kind}:
            relation += "_head_full_mixed"
            factor *= 0.88
    return float(factor), relation


def pattern_gray(rgb: np.ndarray, species: str) -> np.ndarray:
    if rgb.size == 0:
        return np.zeros((64, 64), dtype=np.uint8)
    if species == "SalamanderID2025":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        yellow = lab[:, :, 2].astype(np.float32)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        gray = 0.46 * yellow + 0.34 * sat + 0.20 * (255.0 - val)
    elif species == "LynxID2025":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l = lab[:, :, 0].astype(np.float32)
        inv = 255.0 - l
        blackhat = cv2.morphologyEx(l.astype(np.uint8), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32)
        gray = 0.72 * inv + 0.28 * blackhat
    elif species == "SeaTurtleID2022":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        gray = 0.65 * lab[:, :, 0].astype(np.float32) + 0.35 * lab[:, :, 1].astype(np.float32)
    else:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l = lab[:, :, 0].astype(np.float32)
        gray = 255.0 - l
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return vec
    return vec / n


def compute_vec(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small = small.reshape(-1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).reshape(-1)
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    return normalize(np.concatenate([small, hist * 4.0]).astype(np.float32))


def make_detector():
    try:
        return "sift", cv2.SIFT_create(nfeatures=900, contrastThreshold=0.018, edgeThreshold=12)
    except Exception:
        return "orb", cv2.ORB_create(nfeatures=1200, fastThreshold=7)


def root_sift(desc: np.ndarray | None, detector_name: str) -> np.ndarray | None:
    if desc is None or len(desc) == 0:
        return None
    if detector_name != "sift":
        return desc.astype(np.uint8)
    desc = desc.astype(np.float32)
    desc /= np.maximum(1e-7, desc.sum(axis=1, keepdims=True))
    return np.sqrt(desc)


def feature_item(row: dict, species: str, labels: dict[int, str], detector_name: str, detector, max_side: int, img_size: int) -> FeatureItem:
    image_id = int(row["image_id"])
    orientation = canonical_orientation(row.get("orientation", "unknown"))
    path = image_path_for_species(row, species)
    if not path or not Path(path).exists():
        path = str(row.get("original_path", ""))
    try:
        rgb = read_rgb(path, max_side=max_side)
    except Exception:
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    mask_path = str(row.get("fused_mask_path", "") or row.get("sam_mask_path", ""))
    mask = read_mask(mask_path, rgb.shape[:2])
    rgb, mask, low_light, light_level = preprocess_rgb_for_species(rgb, mask, species, orientation, img_size)
    body_kind = estimate_body_kind(rgb, species)
    gray = pattern_gray(rgb, species)
    kps, desc = detector.detectAndCompute(gray, None)
    if kps is None:
        kps = []
    desc = root_sift(desc, detector_name)
    vec = compute_vec(rgb, gray)
    quality = min(1.0, max(0.20, len(kps) / 320.0))
    if species == "LynxID2025" and low_light:
        quality *= 0.96
    if species == "SalamanderID2025" and body_kind == "partial_or_head":
        quality *= 0.92
    return FeatureItem(
        image_id=image_id,
        species=species,
        identity=str(row.get("identity", "")),
        cluster=str(labels.get(image_id, "")),
        view_path=path,
        source_path=str(row.get("original_path", path)),
        orientation=orientation,
        body_kind=body_kind,
        low_light=bool(low_light),
        keypoints=kps,
        desc=desc,
        vec=vec,
        quality=float(quality),
    )


def cosine_pairs(items: list[FeatureItem], top_k: int, budget: int) -> list[tuple[int, int, float]]:
    if len(items) < 2:
        return []
    ids = [it.image_id for it in items]
    mat = np.stack([it.vec for it in items]).astype(np.float32)
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)
    k = min(max(1, top_k), len(items) - 1)
    pairs: dict[tuple[int, int], float] = {}
    for i, image_id in enumerate(ids):
        idx = np.argpartition(-sim[i], kth=k - 1)[:k]
        for j in idx:
            if not orientation_compatible(items[i], items[int(j)]):
                continue
            a, b = sorted((int(image_id), int(ids[int(j)])))
            pairs[(a, b)] = max(float(sim[i, int(j)]), pairs.get((a, b), -999.0))
    out = [(a, b, s) for (a, b), s in pairs.items()]
    out.sort(key=lambda x: x[2], reverse=True)
    if budget and len(out) > budget:
        out = out[:budget]
    return out


def score_pair(a: FeatureItem, b: FeatureItem, detector_name: str) -> dict:
    vec_sim = float(np.dot(a.vec, b.vec))
    context_factor, context_relation = pair_context_factor(a, b)
    if context_factor <= 0.0:
        return {
            "vec_sim": vec_sim,
            "local_score": 0.0,
            "fused_score": 0.0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "context_factor": 0.0,
            "context_relation": context_relation,
            "orientation_a": a.orientation,
            "orientation_b": b.orientation,
            "body_kind_a": a.body_kind,
            "body_kind_b": b.body_kind,
            "low_light_a": bool(a.low_light),
            "low_light_b": bool(b.low_light),
        }
    if a.desc is None or b.desc is None or len(a.desc) < 4 or len(b.desc) < 4:
        return {
            "vec_sim": vec_sim,
            "local_score": 0.0,
            "fused_score": float(context_factor * 0.40 * max(0.0, vec_sim)),
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "context_factor": context_factor,
            "context_relation": context_relation,
            "orientation_a": a.orientation,
            "orientation_b": b.orientation,
            "body_kind_a": a.body_kind,
            "body_kind_b": b.body_kind,
            "low_light_a": bool(a.low_light),
            "low_light_b": bool(b.low_light),
        }
    norm = cv2.NORM_L2 if detector_name == "sift" else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    try:
        knn = matcher.knnMatch(a.desc, b.desc, k=2)
    except Exception:
        knn = []
    ratio = 0.80 if a.species == "SalamanderID2025" else 0.76
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * max(1e-8, n.distance):
            good.append(m)
    inliers = 0
    inlier_ratio = 0.0
    if len(good) >= 4:
        pts_a = np.float32([a.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts_b = np.float32([b.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 2)
        try:
            _, mask = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=1200)
            if mask is not None:
                inliers = int(mask.ravel().sum())
        except Exception:
            inliers = 0
    if good:
        inlier_ratio = float(inliers / max(1, len(good)))
    local_score = (
        0.46 * min(1.0, inliers / 18.0)
        + 0.28 * min(1.0, len(good) / 70.0)
        + 0.26 * inlier_ratio
    )
    local_score *= min(a.quality, b.quality)
    if a.species == "LynxID2025":
        fused = 0.56 * local_score + 0.44 * max(0.0, vec_sim)
    elif a.species == "SalamanderID2025":
        if "partial_or_head" in {a.body_kind, b.body_kind}:
            fused = 0.47 * local_score + 0.53 * max(0.0, vec_sim)
        else:
            fused = 0.58 * local_score + 0.42 * max(0.0, vec_sim)
    else:
        fused = 0.48 * local_score + 0.52 * max(0.0, vec_sim)
    fused *= context_factor
    return {
        "vec_sim": vec_sim,
        "local_score": float(local_score),
        "fused_score": float(fused),
        "good_matches": int(len(good)),
        "inliers": int(inliers),
        "inlier_ratio": float(inlier_ratio),
        "context_factor": context_factor,
        "context_relation": context_relation,
        "orientation_a": a.orientation,
        "orientation_b": b.orientation,
        "body_kind_a": a.body_kind,
        "body_kind_b": b.body_kind,
        "low_light_a": bool(a.low_light),
        "low_light_b": bool(b.low_light),
    }


def sample_train_rows(rows: pd.DataFrame, max_images: int) -> pd.DataFrame:
    train = rows[(rows["split"].eq("train")) & rows["identity"].astype(str).ne("")].copy()
    if max_images <= 0 or len(train) <= max_images:
        return train
    # Keep identity diversity; no single individual should dominate calibration.
    parts = []
    for _, group in train.groupby("identity", sort=False):
        parts.append(group.head(6))
    sampled = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
    return sampled.head(max_images).copy()


def calibration_pairs(items: list[FeatureItem], top_k: int, budget: int) -> list[tuple[int, int, float]]:
    by_id = {it.image_id: it for it in items}
    positives = []
    groups: dict[str, list[int]] = {}
    for it in items:
        if it.identity:
            groups.setdefault(it.identity, []).append(it.image_id)
    for members in groups.values():
        if len(members) < 2:
            continue
        combos = [
            (a, b)
            for a, b in itertools.combinations(sorted(members), 2)
            if orientation_compatible(by_id[a], by_id[b])
        ]
        positives.extend(combos[: min(len(combos), 12)])
    pairs = cosine_pairs(items, top_k=top_k, budget=max(budget, len(positives) * 3))
    out: dict[tuple[int, int], float] = {}
    for a, b in positives:
        out[(a, b)] = float(np.dot(by_id[a].vec, by_id[b].vec))
    for a, b, sim in pairs:
        out[(a, b)] = sim
        if len(out) >= budget:
            break
    return [(a, b, s) for (a, b), s in out.items()]


def score_pairs(items: list[FeatureItem], pairs: list[tuple[int, int, float]], detector_name: str) -> pd.DataFrame:
    by_id = {it.image_id: it for it in items}
    rows = []
    for idx, (a_id, b_id, prior_sim) in enumerate(pairs, start=1):
        a = by_id.get(int(a_id))
        b = by_id.get(int(b_id))
        if a is None or b is None:
            continue
        s = score_pair(a, b, detector_name)
        same_identity = bool(a.identity and a.identity == b.identity)
        same_cluster = bool(a.cluster and a.cluster == b.cluster)
        rows.append(
            {
                "species": a.species,
                "image_id_a": int(a_id),
                "image_id_b": int(b_id),
                "same_identity": same_identity,
                "same_base_cluster": same_cluster,
                "prior_vec_sim": float(prior_sim),
                "cluster_a": a.cluster,
                "cluster_b": b.cluster,
                **s,
            }
        )
        if idx % 10000 == 0:
            print(f"[{a.species}] scored {idx}/{len(pairs)} pairs")
    return pd.DataFrame(rows)


def threshold_for_precision(cal: pd.DataFrame, target_precision: float, species: str) -> dict:
    if cal.empty or "same_identity" not in cal.columns:
        return default_threshold(species, target_precision)
    y = cal["same_identity"].astype(bool).to_numpy()
    scores = cal["fused_score"].astype(float).to_numpy()
    if y.sum() == 0 or (~y).sum() == 0:
        return default_threshold(species, target_precision)
    order = np.argsort(-scores)
    y_sorted = y[order]
    s_sorted = scores[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    candidates = np.where((precision >= target_precision) & ((tp + fp) >= 3))[0]
    if len(candidates):
        idx = int(candidates[-1])
    else:
        # Bolder fallback: use the positive median but report that precision was not met.
        pos_scores = scores[y]
        idx = int(np.argmin(np.abs(s_sorted - float(np.percentile(pos_scores, 55)))))
    thr = float(s_sorted[idx])
    accepted = scores >= thr
    row = {
        "threshold": thr,
        "target_precision": float(target_precision),
        "cal_precision": float(((accepted & y).sum()) / max(1, int(accepted.sum()))),
        "cal_recall": float(((accepted & y).sum()) / max(1, int(y.sum()))),
        "cal_accepted": int(accepted.sum()),
        "cal_positive": int(y.sum()),
        "cal_negative": int((~y).sum()),
        "cal_auc_rank": auc_rank(y, scores),
    }
    return row


def default_threshold(species: str, target_precision: float) -> dict:
    base = {"LynxID2025": 0.62, "SalamanderID2025": 0.58, "SeaTurtleID2022": 0.66}.get(species, 0.62)
    return {
        "threshold": base,
        "target_precision": target_precision,
        "cal_precision": 0.0,
        "cal_recall": 0.0,
        "cal_accepted": 0,
        "cal_positive": 0,
        "cal_negative": 0,
        "cal_auc_rank": 0.5,
    }


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[y].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))


def labels_for_species(submission: pd.DataFrame, rows: pd.DataFrame) -> dict[int, str]:
    merged = rows[["image_id"]].merge(submission, on="image_id", how="left")
    return dict(zip(merged["image_id"].astype(int), merged["cluster"].astype(str)))


def relabel(ids: list[int], uf: UnionFind, species: str, variant: str) -> dict[int, str]:
    order: dict[int, int] = {}
    labels: dict[int, str] = {}
    for image_id in sorted(ids):
        root = uf.find(image_id)
        if root not in order:
            order[root] = len(order)
        labels[image_id] = f"cluster_{species}_{order[root]}"
    return labels


PROFILE = {
    "independent_strict": {
        "precision": {"LynxID2025": 0.82, "SalamanderID2025": 0.78, "SeaTurtleID2022": 0.84},
        "thr_scale": 0.98,
        "min_inliers": {"LynxID2025": 6, "SalamanderID2025": 5, "SeaTurtleID2022": 6},
        "max_edges": {"LynxID2025": 520, "SalamanderID2025": 150, "SeaTurtleID2022": 180},
    },
    "independent_balanced": {
        "precision": {"LynxID2025": 0.66, "SalamanderID2025": 0.62, "SeaTurtleID2022": 0.70},
        "thr_scale": 0.86,
        "min_inliers": {"LynxID2025": 4, "SalamanderID2025": 3, "SeaTurtleID2022": 4},
        "max_edges": {"LynxID2025": 820, "SalamanderID2025": 360, "SeaTurtleID2022": 340},
    },
    "independent_swing": {
        "precision": {"LynxID2025": 0.52, "SalamanderID2025": 0.50, "SeaTurtleID2022": 0.58},
        "thr_scale": 0.74,
        "min_inliers": {"LynxID2025": 3, "SalamanderID2025": 2, "SeaTurtleID2022": 3},
        "max_edges": {"LynxID2025": 900, "SalamanderID2025": 760, "SeaTurtleID2022": 430},
    },
    "independent_wild": {
        "precision": {"LynxID2025": 0.42, "SalamanderID2025": 0.40, "SeaTurtleID2022": 0.48},
        "thr_scale": 0.62,
        "min_inliers": {"LynxID2025": 2, "SalamanderID2025": 1, "SeaTurtleID2022": 2},
        "max_edges": {"LynxID2025": 930, "SalamanderID2025": 1300, "SeaTurtleID2022": 460},
    },
}


GRAPH_GUARD = {
    "independent_strict": {
        # Component caps are tied to train identity-size priors:
        # Lynx max/p99/p95 = 353/289/151, Salamander = 12/10/7,
        # SeaTurtle = 190/98/55. Strict uses roughly p95, wild uses max.
        "max_component": {"LynxID2025": 151, "SalamanderID2025": 7, "SeaTurtleID2022": 55},
        "max_degree": {"LynxID2025": 3, "SalamanderID2025": 2, "SeaTurtleID2022": 2},
        "mutual_rank": {"LynxID2025": 8, "SalamanderID2025": 3, "SeaTurtleID2022": 4},
    },
    "independent_balanced": {
        "max_component": {"LynxID2025": 289, "SalamanderID2025": 10, "SeaTurtleID2022": 98},
        "max_degree": {"LynxID2025": 4, "SalamanderID2025": 2, "SeaTurtleID2022": 3},
        "mutual_rank": {"LynxID2025": 10, "SalamanderID2025": 3, "SeaTurtleID2022": 5},
    },
    "independent_swing": {
        "max_component": {"LynxID2025": 353, "SalamanderID2025": 12, "SeaTurtleID2022": 190},
        "max_degree": {"LynxID2025": 5, "SalamanderID2025": 2, "SeaTurtleID2022": 3},
        "mutual_rank": {"LynxID2025": 12, "SalamanderID2025": 3, "SeaTurtleID2022": 6},
    },
    "independent_wild": {
        "max_component": {"LynxID2025": 353, "SalamanderID2025": 12, "SeaTurtleID2022": 190},
        "max_degree": {"LynxID2025": 6, "SalamanderID2025": 3, "SeaTurtleID2022": 4},
        "mutual_rank": {"LynxID2025": 15, "SalamanderID2025": 4, "SeaTurtleID2022": 7},
    },
}


def edge_rank_maps(pair_scores: pd.DataFrame) -> dict[tuple[int, int], int]:
    neighbors: dict[int, list[tuple[int, float]]] = {}
    for row in pair_scores.itertuples(index=False):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        s = float(row.fused_score)
        neighbors.setdefault(a, []).append((b, s))
        neighbors.setdefault(b, []).append((a, s))
    ranks: dict[tuple[int, int], int] = {}
    for node, vals in neighbors.items():
        vals.sort(key=lambda x: -x[1])
        for rank, (other, _) in enumerate(vals, start=1):
            ranks[(node, other)] = rank
    return ranks


def uf_members(uf: UnionFind, ids: list[int], root: int) -> list[int]:
    return [image_id for image_id in ids if uf.find(image_id) == root]


def salamander_edge_allowed(row, threshold: float, min_inliers: int) -> bool:
    a_kind = str(getattr(row, "body_kind_a", "body"))
    b_kind = str(getattr(row, "body_kind_b", "body"))
    if a_kind == b_kind:
        return True
    # Head-only to full-body matches can be real, but they should not be the
    # easy bridge that percolates all salamanders into one component.
    return (
        str(getattr(row, "orientation_a", "")) == str(getattr(row, "orientation_b", ""))
        and int(getattr(row, "inliers", 0)) >= max(8, min_inliers + 4)
        and float(getattr(row, "fused_score", 0.0)) >= threshold * 1.18
    )


def graph_guard_for(species: str, variant: str) -> dict[str, int]:
    guard = GRAPH_GUARD.get(variant, GRAPH_GUARD["independent_strict"])
    return {
        "max_component": int(guard["max_component"].get(species, 8)),
        "max_degree": int(guard["max_degree"].get(species, 2)),
        "mutual_rank": int(guard["mutual_rank"].get(species, 3)),
    }


def graph_merge_species(
    species: str,
    rows: pd.DataFrame,
    seed_labels: dict[int, str],
    pair_scores: pd.DataFrame,
    threshold: float,
    min_inliers: int,
    max_edges: int,
    variant: str,
    preserve_existing: bool = False,
) -> tuple[dict[int, str], list[dict]]:
    ids = rows["image_id"].astype(int).tolist()
    uf = UnionFind(ids)
    if preserve_existing:
        by_cluster: dict[str, list[int]] = {}
        for image_id in ids:
            by_cluster.setdefault(seed_labels[image_id], []).append(image_id)
        for members in by_cluster.values():
            anchor = members[0]
            for other in members[1:]:
                uf.union(anchor, other)
    accepted: list[dict] = []
    if pair_scores.empty:
        return relabel(ids, uf, species, variant), accepted
    cross = pair_scores[~pair_scores["same_base_cluster"].astype(bool)].copy()
    if cross.empty:
        return relabel(ids, uf, species, variant), accepted
    guard = graph_guard_for(species, variant)
    ranks = edge_rank_maps(cross)
    node_degree = {image_id: 0 for image_id in ids}
    cross = cross.sort_values(["fused_score", "inliers", "vec_sim"], ascending=[False, False, False])
    for row in cross.itertuples(index=False):
        if len(accepted) >= max_edges:
            break
        if float(row.fused_score) < threshold:
            continue
        if int(row.inliers) < min_inliers and float(row.local_score) < max(0.44, threshold * 0.74):
            continue
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        rank_a = int(ranks.get((a, b), 999))
        rank_b = int(ranks.get((b, a), 999))
        if rank_a > guard["mutual_rank"] or rank_b > guard["mutual_rank"]:
            continue
        if node_degree.get(a, 0) >= guard["max_degree"] or node_degree.get(b, 0) >= guard["max_degree"]:
            continue
        if species == "SalamanderID2025" and not salamander_edge_allowed(row, threshold, min_inliers):
            continue
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            continue
        if uf.size[ra] + uf.size[rb] > guard["max_component"]:
            continue
        if uf.union(a, b):
            node_degree[a] = node_degree.get(a, 0) + 1
            node_degree[b] = node_degree.get(b, 0) + 1
            accepted.append(
                {
                    "variant": variant,
                    "species": species,
                    "guard_max_component": int(guard["max_component"]),
                    "guard_max_degree": int(guard["max_degree"]),
                    "guard_mutual_rank": int(guard["mutual_rank"]),
                    "rank_a_to_b": rank_a,
                    "rank_b_to_a": rank_b,
                    "image_id_a": a,
                    "image_id_b": b,
                    "fused_score": float(row.fused_score),
                    "vec_sim": float(row.vec_sim),
                    "local_score": float(row.local_score),
                    "inliers": int(row.inliers),
                    "good_matches": int(row.good_matches),
                    "cluster_a": str(row.cluster_a),
                    "cluster_b": str(row.cluster_b),
                    "context_relation": str(getattr(row, "context_relation", "")),
                    "orientation_a": str(getattr(row, "orientation_a", "")),
                    "orientation_b": str(getattr(row, "orientation_b", "")),
                    "body_kind_a": str(getattr(row, "body_kind_a", "")),
                    "body_kind_b": str(getattr(row, "body_kind_b", "")),
                }
            )
    return relabel(ids, uf, species, variant), accepted


SHAPE_TARGETS = {
    # These are not seed labels. They are sanity targets for open-set cluster
    # shape, based on train identity multiplicities plus the best submitted
    # non-collapsed species profile. The old chooser only blocked huge clusters,
    # which let it select submissions that were mathematically over-split.
    "LynxID2025": {
        "clusters": 135,
        "min_clusters": 75,
        "max_clusters": 260,
        "max_cluster": 67,
        "min_max_cluster": 28,
        "max_max_cluster": 125,
        "singletons": 32,
        "max_singletons": 320,
    },
    "SalamanderID2025": {
        "clusters": 459,
        "min_clusters": 380,
        "max_clusters": 560,
        "max_cluster": 9,
        "min_max_cluster": 4,
        "max_max_cluster": 16,
        "singletons": 330,
        "max_singletons": 450,
    },
    "SeaTurtleID2022": {
        "clusters": 166,
        "min_clusters": 95,
        "max_clusters": 290,
        "max_cluster": 13,
        "min_max_cluster": 6,
        "max_max_cluster": 42,
        "singletons": 60,
        "max_singletons": 220,
    },
    "TexasHornedLizards": {
        "clusters": 74,
        "min_clusters": 35,
        "max_clusters": 155,
        "max_cluster": 26,
        "min_max_cluster": 8,
        "max_max_cluster": 55,
        "singletons": 26,
        "max_singletons": 145,
    },
}

VARIANT_BIAS = {
    "independent_swing": -0.04,
    "independent_wild": 0.12,
}


def log_ratio(value: float, target: float) -> float:
    return abs(math.log(max(1.0, value + 1.0) / max(1.0, target + 1.0)))


def species_shape_score(row) -> tuple[float, list[str]]:
    species = str(row.species)
    target = SHAPE_TARGETS.get(species)
    if target is None:
        return 0.0, []
    n_clusters = int(row.n_clusters)
    singletons = int(row.singletons)
    max_cluster = int(row.max_cluster)
    notes: list[str] = []
    if n_clusters <= 1 and int(row.n_images) > 1:
        notes.append("collapsed")
        return 999.0, notes

    score = 2.6 * log_ratio(n_clusters, target["clusters"])
    score += 1.4 * log_ratio(max_cluster, target["max_cluster"])
    score += 0.55 * log_ratio(singletons, target["singletons"])

    if n_clusters < target["min_clusters"]:
        score += 3.0 * log_ratio(n_clusters, target["min_clusters"])
        notes.append("too_few_clusters")
    if n_clusters > target["max_clusters"]:
        score += 3.0 * log_ratio(n_clusters, target["max_clusters"])
        notes.append("too_many_clusters")
    if max_cluster < target["min_max_cluster"]:
        score += 1.6 * log_ratio(max_cluster, target["min_max_cluster"])
        notes.append("max_cluster_too_small")
    if max_cluster > target["max_max_cluster"]:
        score += 12.0 * log_ratio(max_cluster, target["max_max_cluster"])
        notes.append("max_cluster_too_large")
    if singletons > target["max_singletons"]:
        score += 1.5 * log_ratio(singletons, target["max_singletons"])
        notes.append("too_many_singletons")
    return float(score), notes


def candidate_shape_scores(candidate_report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, part in candidate_report.groupby("variant", sort=False):
        total = float(VARIANT_BIAS.get(str(variant), 0.0))
        notes = []
        for row in part.itertuples(index=False):
            score, species_notes = species_shape_score(row)
            total += score
            if species_notes:
                notes.append(f"{row.species}:{'|'.join(species_notes)}")
        rows.append(
            {
                "variant": str(variant),
                "shape_score": float(total),
                "notes": ";".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["shape_score", "variant"], ascending=[True, True]).reset_index(drop=True)
    return out


def choose_recommended_submission(candidate_report: pd.DataFrame, written: dict[str, str]) -> tuple[str, Path, pd.DataFrame]:
    shape_scores = candidate_shape_scores(candidate_report)
    for row in shape_scores.itertuples(index=False):
        variant = str(row.variant)
        if variant in written:
            return variant, Path(written[variant]), shape_scores
    variant, path = next(iter(written.items()))
    return variant, Path(path), shape_scores


def compact_submission_labels(sub: pd.DataFrame, view_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize final labels to cluster_<SpeciesID>_<integer>.

    Kaggle only needs stable cluster identifiers. The algorithm's internal
    variant/version names are useful in reports, but long labels are risky and
    ugly in the actual upload file.
    """
    species_by_id = dict(zip(view_df["image_id"].astype(int), view_df["species_id"].astype(str)))
    work = sub.copy()
    work["image_id"] = work["image_id"].astype(int)
    work["_species_id"] = work["image_id"].map(species_by_id).fillna("AnimalCLEF").astype(str)
    compact = {}
    for species in SPECIES_ORDER:
        mask = work["_species_id"].eq(species)
        labels = work.loc[mask, "cluster"].astype(str)
        order = {old: f"cluster_{species}_{idx}" for idx, old in enumerate(sorted(labels.unique()))}
        compact.update(order)
    other = work.loc[~work["_species_id"].isin(SPECIES_ORDER), "cluster"].astype(str)
    for idx, old in enumerate(sorted(other.unique())):
        compact.setdefault(old, f"cluster_AnimalCLEF_{idx}")
    work["cluster"] = work["cluster"].astype(str).map(compact)
    return work[["image_id", "cluster"]]


def build_submission(sample: pd.DataFrame, view_df: pd.DataFrame, updates: dict[int, str], variant: str, out_path: Path) -> pd.DataFrame:
    sub = sample[["image_id"]].copy()
    species_by_id = dict(zip(view_df["image_id"].astype(int), view_df["species_id"].astype(str)))
    def label(image_id: int) -> str:
        if image_id in updates:
            return updates[image_id]
        species = species_by_id.get(image_id, "AnimalCLEF")
        return f"cluster_{species}_singleton_{image_id}"

    sub["cluster"] = sub["image_id"].astype(int).map(label)
    sub = compact_submission_labels(sub, view_df)
    validate_submission(sub, sample)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return sub


def write_species_fragments(sub: pd.DataFrame, view_df: pd.DataFrame, variant: str, fragments_dir: Path) -> None:
    """Write one replaceable cluster fragment per species.

    This keeps the final notebook usable as a one-shot submission generator,
    while also giving us a clean contract for parallel species notebooks:
    produce image_id/cluster rows for one species, then a later merger can
    splice that fragment without touching the other species.
    """
    species_by_id = dict(zip(view_df["image_id"].astype(int), view_df["species_id"].astype(str)))
    work = sub.copy()
    work["image_id"] = work["image_id"].astype(int)
    work["species_id"] = work["image_id"].map(species_by_id).fillna("AnimalCLEF").astype(str)
    fragments_dir.mkdir(parents=True, exist_ok=True)
    for species in SPECIES_ORDER:
        frag = work[work["species_id"].eq(species)][["image_id", "cluster", "species_id"]].copy()
        frag.insert(2, "variant", variant)
        frag["source_version"] = VERSION
        frag.to_csv(fragments_dir / f"fragment_{VERSION}_{variant}_{species}.csv", index=False)


def summarize_submission(sub: pd.DataFrame, reference: pd.DataFrame | None, view_df: pd.DataFrame, variant: str) -> list[dict]:
    sub_map = dict(zip(sub["image_id"].astype(int), sub["cluster"].astype(str)))
    ref_map = dict(zip(reference["image_id"].astype(int), reference["cluster"].astype(str))) if reference is not None else {}
    rows = []
    for species in SPECIES_ORDER:
        ids = view_df.loc[(view_df["split"].eq("test")) & (view_df["species_id"].eq(species)), "image_id"].astype(int).tolist()
        labels = [sub_map[i] for i in ids]
        counts = pd.Series(labels).value_counts()
        changed = sum(1 for i in ids if ref_map and sub_map[i] != ref_map.get(i))
        rows.append(
            {
                "variant": variant,
                "species": species,
                "n_images": int(len(ids)),
                "n_clusters": int(counts.shape[0]),
                "singletons": int((counts == 1).sum()),
                "max_cluster": int(counts.max()) if len(counts) else 0,
                "membership_changed_vs_optional_reference": int(changed) if ref_map else -1,
            }
        )
    return rows


def import_texas_helpers():
    try:
        import texas_astrodot_2025reuse_v20260426 as texas_base
        import texas_astrodot_nooval_v20260427 as texas_nooval

        # Make the Texas rule explicit at the final fusion layer too. The
        # field photos are ventral views with the head already at the top, so
        # the base helper must never vertically flip the belly-dot template.
        if hasattr(texas_nooval, "align_vertical_no_flip"):
            texas_base.align_vertical = texas_nooval.align_vertical_no_flip
        if hasattr(texas_nooval, "texas_belly_template_no_oval"):
            texas_base.texas_belly_template = texas_nooval.texas_belly_template_no_oval
        if hasattr(texas_nooval, "texas_pair_score_no_flip"):
            texas_base.texas_pair_score = texas_nooval.texas_pair_score_no_flip
        return texas_base, texas_nooval
    except Exception as exc:
        print(f"[Texas] helper import failed: {exc}")
        return None, None


def estimate_non_grey_foreground(rgb: np.ndarray) -> np.ndarray:
    diff_grey = np.abs(rgb.astype(np.int16) - NEUTRAL_GREY_RGB.reshape(1, 1, 3).astype(np.int16)).sum(axis=2)
    diff_black = np.abs(rgb.astype(np.int16) - BLACK_RGB.reshape(1, 1, 3).astype(np.int16)).sum(axis=2)
    mask = ((diff_grey > 24) & (diff_black > 18)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    coverage = float(mask.mean() / 255.0)
    if coverage < 0.02 or coverage > 0.94:
        mask = np.full(mask.shape[:2], 255, dtype=np.uint8)
    return mask


def write_neutral_grey_view(rgb: np.ndarray, mask: np.ndarray, out_img: Path, out_mask: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    prepared = np.where(mask[:, :, None] > 0, rgb, NEUTRAL_GREY_RGB.reshape(1, 1, 3)).astype(np.uint8)
    cv2.imwrite(str(out_img), cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_mask), mask)


def texas_labels_from_views(
    rows: pd.DataFrame,
    base_labels: dict[int, str],
    args: argparse.Namespace,
    reports_dir: Path,
    viz_dir: Path | None,
) -> dict[str, dict[int, str]]:
    texas_base, texas_nooval = import_texas_helpers()
    if texas_base is None or texas_nooval is None or rows.empty:
        return {}
    helper_args = argparse.Namespace(max_side=args.max_side, texas_canvas_w=224, texas_canvas_h=320)
    prepared_dir = reports_dir.parent / "prepared_views" / "TexasHornedLizards"
    items = []
    for idx, row in enumerate(rows.to_dict("records"), start=1):
        rec = dict(row)
        original_path = str(rec.get("original_path", "") or "")
        final_path = str(
            rec.get("view_species_final_path")
            or rec.get("view_sam_yolo_union_path")
            or rec.get("view_sam_clean_path")
            or original_path
        )
        # Final correction: Texas uses the SAM+YOLO final/union crop as the
        # image source. No oval is applied and the no-flip monkey patch above
        # keeps the head-at-top acquisition protocol intact. A neutral-grey
        # background is written before the astro-dot template is extracted.
        rec["source_path"] = original_path
        rec["view_path"] = final_path
        rec["mask_path"] = ""
        rec["view_source"] = "sam_yolo_final_neutral_grey_nooval_noflip"
        if final_path and Path(final_path).exists():
            try:
                rgb = read_rgb(final_path, args.max_side)
                mask = estimate_non_grey_foreground(rgb)
                out_img = prepared_dir / f"{TEXAS}_{int(row['image_id'])}_neutralgrey.jpg"
                out_mask = prepared_dir / f"{TEXAS}_{int(row['image_id'])}_neutralgrey_mask.png"
                write_neutral_grey_view(rgb, mask, out_img, out_mask)
                rec["view_path"] = str(out_img)
                rec["mask_path"] = str(out_mask)
            except Exception as exc:
                print(f"[Texas] neutral-grey prep failed image_id={row.get('image_id')}: {exc}")
        try:
            items.append(texas_nooval.texas_belly_template_no_oval(rec, base_labels[int(row["image_id"])], helper_args))
        except Exception as exc:
            print(f"[Texas] template failed image_id={row.get('image_id')}: {exc}")
        if idx % 75 == 0:
            print(f"[Texas no-oval YOLO-view] templates {idx}/{len(rows)}")
    if not items:
        return {}
    pair_scores = texas_base.score_all_texas_pairs(items, args.texas_pair_budget)
    pair_scores.to_csv(reports_dir / f"{VERSION}_Texas_view_nooval_pair_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "image_id": item.image_id,
                "view_path": item.view_path,
                "view_source": item.view_source,
                "quality": item.quality,
                "dot_points": len(item.dot_points),
                "mask_coverage": item.debug.get("mask_coverage", np.nan),
                "pca_angle": item.debug.get("pca_angle", np.nan),
                "rotate_angle": item.debug.get("rotate_angle", np.nan),
                "flipped_vertical": item.debug.get("flipped_vertical", False),
                "orientation_rule": item.debug.get("orientation_rule", "head_already_top_no_flip"),
                "mask_mode": item.debug.get("mask_mode", "full_aligned_crop_no_oval"),
            }
            for item in items
        ]
    ).to_csv(reports_dir / f"{VERSION}_Texas_template_report.csv", index=False)
    labels = {}
    for variant in ["merge_ultra", "splitmerge_guarded", "splitmerge_swing"]:
        try:
            labels[variant] = texas_independent_variant_labels(items, pair_scores, variant)
        except Exception as exc:
            print(f"[Texas] variant {variant} failed: {exc}")
    if viz_dir is not None:
        try:
            texas_base.save_texas_preview(items, viz_dir / f"{VERSION}_Texas_nooval_yoloview_preview.jpg", args.visual_limit)
            texas_base.save_pair_preview(items, pair_scores, viz_dir / f"{VERSION}_Texas_nooval_yoloview_top_pairs.jpg", max(6, args.visual_limit // 2), True)
        except Exception as exc:
            print(f"[Texas] visualization failed: {exc}")
    return labels


def texas_independent_variant_labels(items: list, pair_scores: pd.DataFrame, variant: str) -> dict[int, str]:
    """Texas clustering without any previous submission seed.

    Texas has no train identities, but the acquisition protocol is unusually
    standardized: ventral belly, head at top. This uses mutual high-rank dot
    matches as graph edges and caps component growth to avoid one large dot
    texture attractor.
    """
    ids = [int(it.image_id) for it in items]
    uf = UnionFind(ids)
    if pair_scores.empty:
        return relabel(ids, uf, TEXAS, f"{variant}_texas_independent")
    if variant == "merge_ultra":
        thr, rank_limit, max_comp, max_edges = 0.565, 2, 12, 95
        min_overlap, min_point, min_stack = 0.42, 0.46, 0.500
    elif variant == "splitmerge_guarded":
        thr, rank_limit, max_comp, max_edges = 0.540, 4, 24, 190
        min_overlap, min_point, min_stack = 0.35, 0.42, 0.490
    else:
        thr, rank_limit, max_comp, max_edges = 0.518, 6, 36, 235
        min_overlap, min_point, min_stack = 0.30, 0.39, 0.480

    cand = pair_scores[
        (pair_scores["score"].astype(float) >= thr)
        & (pair_scores["overlap"].astype(float) >= min_overlap)
        & (pair_scores["point_score"].astype(float) >= min_point)
        & (pair_scores["stack_gain"].astype(float) >= min_stack)
        & (pair_scores["transform"].astype(str).eq("identity"))
    ].copy()
    if cand.empty:
        return relabel(ids, uf, TEXAS, f"{variant}_texas_independent")

    neighbors: dict[int, list[tuple[int, float]]] = {i: [] for i in ids}
    for row in cand.itertuples(index=False):
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

    accepted = 0
    for row in cand.sort_values(["score", "overlap", "point_score"], ascending=False).itertuples(index=False):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        if ranks.get((a, b), 999) > rank_limit or ranks.get((b, a), 999) > rank_limit:
            continue
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            continue
        if uf.size[ra] + uf.size[rb] > max_comp:
            continue
        if uf.union(a, b):
            accepted += 1
            if accepted >= max_edges:
                break
    return relabel(ids, uf, TEXAS, f"{variant}_texas_independent")


def thumb(path: str, size: tuple[int, int]) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", size, (20, 20, 20))
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def save_edge_preview(view_df: pd.DataFrame, accepted: pd.DataFrame, out_path: Path, limit: int) -> None:
    if accepted.empty:
        return
    row_by_id = {int(r["image_id"]): r for r in view_df.to_dict("records")}
    edges = accepted.sort_values("fused_score", ascending=False).head(limit)
    tile = (230, 180)
    label_h = 36
    canvas = Image.new("RGB", (tile[0] * 2, len(edges) * (tile[1] + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for ridx, edge in enumerate(edges.to_dict("records")):
        y = ridx * (tile[1] + label_h)
        text = (
            f"{edge['variant']} {edge['species']} {edge['image_id_a']} vs {edge['image_id_b']} "
            f"s={float(edge['fused_score']):.3f} inl={int(edge.get('inliers', 0))}"
        )
        draw.text((5, y + 4), text[:90], fill=(255, 240, 140))
        for c, image_id in enumerate([int(edge["image_id_a"]), int(edge["image_id_b"])]):
            row = row_by_id.get(image_id, {})
            p = str(row.get("view_species_final_path") or row.get("view_sam_yolo_union_path") or row.get("original_path") or "")
            canvas.paste(thumb(p, tile), (c * tile[0], y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.train_images_per_species = min(args.train_images_per_species, 80)
        args.train_pairs_per_species = min(args.train_pairs_per_species, 240)
        args.pair_budget_per_species = min(args.pair_budget_per_species, 900)
        args.texas_pair_budget = min(args.texas_pair_budget, 900)
        args.top_k = min(args.top_k, 16)
    random.seed(SEED)
    np.random.seed(SEED)

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    submissions_dir = output_root / "submissions"
    fragments_dir = output_root / "species_fragments"
    viz_dir = output_root / "visualizations" if args.save_visualizations else None
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir.mkdir(parents=True, exist_ok=True)
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_data_root(args.data_root)
    view_manifest = find_view_manifest(args.view_manifest)
    view_df = prepare_view_manifest(view_manifest, data_root)
    sample = pd.read_csv(data_root / "sample_submission.csv")
    reference_arg = args.reference_submission or args.base_submission
    reference_path = find_reference_submission(reference_arg, data_root)
    reference = load_submission(reference_path) if reference_path else None
    if reference is not None:
        validate_submission(reference, sample)

    print(f"VERSION={VERSION}")
    print(f"data_root={data_root}")
    print(f"view_manifest={view_manifest}")
    print(f"reference_submission={reference_path}")
    print(f"output_root={output_root}")

    detector_name, detector = make_detector()
    print(f"detector={detector_name}")

    test_rows_by_species = {
        species: view_df[(view_df["split"].eq("test")) & (view_df["species_id"].eq(species))].sort_values("image_id").copy()
        for species in SPECIES_ORDER
    }
    source_rows = []
    cal_thresholds: dict[str, dict[str, dict]] = {}
    test_pair_scores: dict[str, pd.DataFrame] = {}
    item_cache: dict[str, list[FeatureItem]] = {}
    feature_rows: list[dict] = []

    for species in REID_SPECIES:
        test_rows = test_rows_by_species[species]
        if args.smoke:
            test_rows = test_rows.head(60).copy()
        seed_labels = {int(r["image_id"]): f"seed_{species}_{int(r['image_id'])}" for r in test_rows.to_dict("records")}
        print(f"\n[{species}] extracting test features: {len(test_rows)}")
        test_items = [
            feature_item(row, species, seed_labels, detector_name, detector, args.max_side, args.img_size)
            for row in test_rows.to_dict("records")
        ]
        item_cache[species] = test_items
        feature_rows.extend(
            {
                "species": it.species,
                "split": "test",
                "image_id": it.image_id,
                "orientation": it.orientation,
                "body_kind": it.body_kind,
                "low_light": bool(it.low_light),
                "n_keypoints": int(len(it.keypoints)),
                "quality": float(it.quality),
            }
            for it in test_items
        )
        test_pairs = cosine_pairs(test_items, args.top_k, args.pair_budget_per_species)
        test_scores = score_pairs(test_items, test_pairs, detector_name)
        test_scores.to_csv(reports_dir / f"{VERSION}_{species}_test_pair_scores.csv", index=False)
        test_pair_scores[species] = test_scores

        train_rows = sample_train_rows(view_df[view_df["species_id"].eq(species)].copy(), args.train_images_per_species)
        print(f"[{species}] extracting train calibration features: {len(train_rows)}")
        train_labels = {int(r["image_id"]): str(r.get("identity", "")) for r in train_rows.to_dict("records")}
        train_items = [
            feature_item(row, species, train_labels, detector_name, detector, args.max_side, args.img_size)
            for row in train_rows.to_dict("records")
        ]
        feature_rows.extend(
            {
                "species": it.species,
                "split": "train_calibration",
                "image_id": it.image_id,
                "orientation": it.orientation,
                "body_kind": it.body_kind,
                "low_light": bool(it.low_light),
                "n_keypoints": int(len(it.keypoints)),
                "quality": float(it.quality),
            }
            for it in train_items
        )
        train_pairs = calibration_pairs(train_items, args.top_k, args.train_pairs_per_species)
        cal_scores = score_pairs(train_items, train_pairs, detector_name)
        cal_scores.to_csv(reports_dir / f"{VERSION}_{species}_train_calibration_pair_scores.csv", index=False)
        cal_thresholds[species] = {}
        for profile, cfg in PROFILE.items():
            target = cfg["precision"][species]
            cal_thresholds[species][profile] = threshold_for_precision(cal_scores, target, species)
            cal_thresholds[species][profile]["threshold"] *= float(cfg["thr_scale"])
            cal_thresholds[species][profile]["profile"] = profile
            cal_thresholds[species][profile]["species"] = species
        source_rows.extend(cal_thresholds[species].values())

    pd.DataFrame(source_rows).to_csv(reports_dir / f"{VERSION}_calibrated_thresholds.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(reports_dir / f"{VERSION}_feature_diagnostics.csv", index=False)

    texas_rows = test_rows_by_species[TEXAS].copy()
    if args.smoke:
        texas_rows = texas_rows.head(60).copy()
    texas_seed_labels = {int(r["image_id"]): f"seed_{TEXAS}_{int(r['image_id'])}" for r in texas_rows.to_dict("records")}
    print("[Texas] independent belly-dot graph; no past submission seed labels")
    texas_variants = texas_labels_from_views(texas_rows, texas_seed_labels, args, reports_dir, viz_dir)

    variants: dict[str, dict] = {
        "independent_swing": {
            "profiles": {
                "LynxID2025": "independent_swing",
                "SalamanderID2025": "independent_swing",
                "SeaTurtleID2022": "independent_balanced",
            },
            "texas": "splitmerge_swing",
        },
        "independent_wild": {
            "profiles": {
                "LynxID2025": "independent_wild",
                "SalamanderID2025": "independent_wild",
                "SeaTurtleID2022": "independent_swing",
            },
            "texas": "splitmerge_swing",
        },
    }

    candidate_rows: list[dict] = []
    accepted_all: list[dict] = []
    written: dict[str, str] = {}
    for variant, spec in variants.items():
        updates: dict[int, str] = {}
        for species in REID_SPECIES:
            rows = test_rows_by_species[species]
            if args.smoke:
                rows = rows.head(60).copy()
            seed_labels = {int(r["image_id"]): f"seed_{species}_{int(r['image_id'])}" for r in rows.to_dict("records")}
            profile = spec["profiles"].get(species)
            if profile is None:
                updates.update({image_id: f"cluster_{species}_singleton_{image_id}" for image_id in seed_labels})
                continue
            threshold = float(cal_thresholds[species][profile]["threshold"])
            cfg = PROFILE[profile]
            labels, accepted = graph_merge_species(
                species,
                rows,
                seed_labels,
                test_pair_scores.get(species, pd.DataFrame()),
                threshold=threshold,
                min_inliers=int(cfg["min_inliers"][species]),
                max_edges=int(cfg["max_edges"][species]),
                variant=variant,
                preserve_existing=False,
            )
            updates.update(labels)
            accepted_all.extend(accepted)
        texas_choice = spec.get("texas")
        if texas_choice and texas_choice in texas_variants:
            updates.update(texas_variants[texas_choice])
        elif not texas_rows.empty:
            updates.update({image_id: f"cluster_{TEXAS}_singleton_{image_id}" for image_id in texas_seed_labels})
        out_path = submissions_dir / f"submission_{VERSION}_{variant}.csv"
        sub = build_submission(sample, view_df, updates, variant, out_path)
        write_species_fragments(sub, view_df, variant, fragments_dir)
        written[variant] = str(out_path)
        candidate_rows.extend(summarize_submission(sub, reference, view_df, variant))
        print(f"wrote {out_path}")

    candidate_report = pd.DataFrame(candidate_rows)
    candidate_report.to_csv(reports_dir / f"{VERSION}_candidate_report.csv", index=False)
    shape_score_report = candidate_shape_scores(candidate_report)
    shape_score_report.to_csv(reports_dir / f"{VERSION}_candidate_shape_scores.csv", index=False)
    accepted_report = pd.DataFrame(accepted_all)
    accepted_report.to_csv(reports_dir / f"{VERSION}_accepted_edges.csv", index=False)
    if viz_dir is not None and not accepted_report.empty:
        save_edge_preview(view_df, accepted_report, viz_dir / f"{VERSION}_accepted_edge_preview.jpg", args.visual_limit)

    # Convenience output: pick the candidate whose per-species cluster shape is
    # closest to a sane non-collapsed regime. This avoids both giant-component
    # failure and the quieter failure where a submission is mostly singleton
    # clusters.
    recommended_variant, recommended, shape_score_report = choose_recommended_submission(candidate_report, written)
    shape_score_report.to_csv(reports_dir / f"{VERSION}_candidate_shape_scores.csv", index=False)
    if recommended.exists():
        recommended_df = pd.read_csv(recommended)
        recommended_df.to_csv(output_root / "submission.csv", index=False)
        recommended_df.to_csv(Path.cwd() / "submission.csv", index=False)

    summary = {
        "version": VERSION,
        "data_root": str(data_root),
        "view_manifest": str(view_manifest),
        "reference_submission_optional": str(reference_path) if reference_path else None,
        "detector": detector_name,
        "written_submissions": written,
        "recommended_variant": recommended_variant,
        "recommended_first_review": str(recommended),
        "kaggle_top_level_submission": str(Path.cwd() / "submission.csv"),
        "reports": {
            "candidate_report": str(reports_dir / f"{VERSION}_candidate_report.csv"),
            "candidate_shape_scores": str(reports_dir / f"{VERSION}_candidate_shape_scores.csv"),
            "thresholds": str(reports_dir / f"{VERSION}_calibrated_thresholds.csv"),
            "feature_diagnostics": str(reports_dir / f"{VERSION}_feature_diagnostics.csv"),
            "accepted_edges": str(reports_dir / f"{VERSION}_accepted_edges.csv"),
            "texas_template_report": str(reports_dir / f"{VERSION}_Texas_template_report.csv"),
        },
        "species_fragments": str(fragments_dir),
        "notes": [
            "Main candidates are independent clusters built from singleton seeds.",
            "No previous submission is auto-discovered or used as a clustering base.",
            f"Global image prep pads to a species-specific square background and resizes to {args.img_size}x{args.img_size}: Lynx uses black padding; Salamander, SeaTurtle, and Texas use neutral-grey padding.",
            "Lynx uses the original image as signal source; the SAM fused mask only gates subject-only CLAHE/brightness adjustment.",
            "Salamander uses SAM-clean output with neutral-grey background and strictly no rotation or flip preprocessing.",
            "SeaTurtle keeps the existing clean-view route, with only the global square-pad/resize normalization added.",
            "Texas uses SAM+YOLO final/union output with neutral-grey background, no oval crop, and no vertical flip.",
            "ReID graph merges now require mutual-rank agreement, per-image degree caps, and per-species component caps.",
            "Lynx/Salamander/SeaTurtle component caps are train-prior tiers: p95, p99, and observed train max.",
            "submission.csv is copied from the candidate with the best species-shape sanity score, not merely the first non-collapsed file.",
            "Texas orientation rule is head_already_top_no_flip; neither template creation nor pair scoring uses vertical flipping.",
            "Per-species fragment CSVs are written so expensive species notebooks can be merged later.",
        ],
    }
    (reports_dir / f"{VERSION}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nCandidate report:")
    print(candidate_report.to_string(index=False))
    print("\nCandidate shape scores:")
    print(shape_score_report.to_string(index=False))
    if not accepted_report.empty:
        print("\nAccepted edge counts:")
        print(accepted_report.groupby(["variant", "species"]).size().reset_index(name="n_edges").to_string(index=False))
    print(f"\nRecommended first review/upload candidate ({recommended_variant}): {recommended}")
    print(f"Convenience submission.csv: {output_root / 'submission.csv'}")


if __name__ == "__main__":
    main()
