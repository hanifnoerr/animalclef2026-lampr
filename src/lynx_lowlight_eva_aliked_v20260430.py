#!/usr/bin/env python3
"""
AnimalCLEF2026 Lynx low-light EVA/CLIP + ALIKED verifier v20260430.

This is a guarded Lynx-only postprocessor. It starts from the current best
submission partition, keeps Salamander/SeaTurtle/Texas unchanged, and rewrites
only Lynx clusters using:

* original Lynx camera-trap images
* SAM mask-guided subject-only low-light enhancement
* EVA02/CLIP global embeddings with 5-crop TTA
* ALIKED local keypoint descriptors with geometric verification
* Lynx train identity calibration

The notebook using this script is intended as an incremental branch, not a
full replacement of the current best pipeline.
"""


import argparse
import itertools
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageOps
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

VERSION = "lynx_lowlight_eva_aliked_v20260430"
LYNX = "LynxID2025"
SEED = 20260430
BLACK = np.array([0, 0, 0], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sam-manifest", type=str, default=None)
    parser.add_argument("--current-best", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=f"/kaggle/working/animalclef_{VERSION}")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=22)
    parser.add_argument("--train-pair-limit", type=int, default=9000)
    parser.add_argument("--test-pair-limit", type=int, default=26000)
    parser.add_argument("--max-aliked-keypoints", type=int, default=512)
    parser.add_argument("--aliked-match-threshold", type=float, default=0.72)
    parser.add_argument("--eva-batch-size", type=int, default=32)
    parser.add_argument("--eva-model", type=str, default="EVA02-B-16")
    parser.add_argument("--eva-pretrained", type=str, default="merged2b_s8b_b131k")
    parser.add_argument("--disable-eva", action="store_true")
    parser.add_argument("--disable-aliked", action="store_true")
    parser.add_argument("--allow-orb-fallback", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def install_if_missing(import_name: str, pip_name: str) -> bool:
    try:
        __import__(import_name)
        return True
    except Exception:
        pass
    try:
        print(f"[deps] installing {pip_name}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)
        __import__(import_name)
        return True
    except Exception as exc:
        print(f"[deps] {pip_name} unavailable: {exc}")
        return False


def candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for value in [
        os.environ.get("DATA_ROOT"),
        "/kaggle/input/animal-clef-2026",
        "/kaggle/input/competitions/animal-clef-2026",
        "/kaggle/input",
        "/kaggle/working",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/animal-clef-2026",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/current_wildfusion_graph_v20260423",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026",
        ".",
    ]:
        if value:
            p = Path(value)
            if p.exists():
                roots.append(p)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def find_data_root(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if (p / "metadata.csv").exists() and (p / "sample_submission.csv").exists():
            return p.resolve()
        raise FileNotFoundError(f"--data-root does not contain metadata/sample_submission: {p}")
    for root in candidate_roots():
        if (root / "metadata.csv").exists() and (root / "sample_submission.csv").exists():
            return root.resolve()
    for root in candidate_roots():
        try:
            for meta in root.rglob("metadata.csv"):
                if (meta.parent / "sample_submission.csv").exists():
                    return meta.parent.resolve()
        except Exception:
            continue
    raise FileNotFoundError("Could not locate AnimalCLEF2026 data root.")


def path_preference(path: Path, names: list[str]) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    penalty = 0
    if "local_smoke" in text or "__pycache__" in text:
        penalty += 100
    if "final-texas-boundary-split-only" in text or "texas_boundary_splitonly" in text:
        penalty -= 20
    if "current_wildfusion_graph" in text:
        penalty -= 12
    if "/kaggle/input/" in text:
        penalty -= 10
    for i, name in enumerate(names):
        if path.name == name:
            penalty += i
            break
    if path.name == "submission.csv":
        penalty += 20
    return penalty, len(text), text


def find_first_file(names: list[str], arg: str | None = None) -> Path:
    if arg:
        p = Path(arg)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Path does not exist: {p}")
    hits: list[Path] = []
    for root in candidate_roots():
        for name in names:
            direct = root / name
            if direct.exists():
                hits.append(direct)
            try:
                hits.extend(root.rglob(name))
            except Exception:
                pass
    hits = sorted({p.resolve() for p in hits if p.exists()}, key=lambda p: path_preference(p, names))
    if not hits:
        raise FileNotFoundError(f"Could not find any of: {names}")
    return hits[0]


def find_current_best(arg: str | None) -> Path:
    names = [
        "submission_final_texas_boundary_splitonly_balanced_from_032583_v20260430.csv",
        "submission_metadata_prior_fusion_texas_meta_balanced_from_032583_v20260430.csv",
        "submission_032583_salamander_p80_cluster_cap_v20260429.csv",
        "submission_032368_gamble_salamander_p80_cap_v20260429.csv",
        "submission.csv",
    ]
    return find_first_file(names, arg)


def find_sam_manifest(arg: str | None) -> Path | None:
    names = ["view_manifest_sam3_all_species.csv"]
    if arg:
        p = Path(arg)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"--sam-manifest does not exist: {p}")
    hits: list[Path] = []
    for root in candidate_roots():
        try:
            hits.extend(root.rglob(names[0]))
        except Exception:
            pass
    def manifest_rank(p: Path) -> tuple[int, int, str]:
        text = str(p).replace("\\", "/").lower()
        penalty = 0
        if "fake" in text or "local_smoke" in text:
            penalty += 100
        if "/kaggle/input/" in text:
            penalty -= 20
        if "animalclef_sam3_views_cache" in text:
            penalty -= 5
        return penalty, len(text), text

    hits = sorted({p.resolve() for p in hits if p.exists()}, key=manifest_rank)
    return hits[0] if hits else None


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns.")
    df = df[["image_id", "cluster"]].copy()
    df["image_id"] = df["image_id"].astype(int)
    df["cluster"] = df["cluster"].astype(str)
    if df["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id in {path}")
    return df


def prepare_metadata(data_root: Path, sam_manifest: Path | None) -> pd.DataFrame:
    metadata = pd.read_csv(data_root / "metadata.csv").copy()
    if "dataset" not in metadata.columns:
        metadata["dataset"] = metadata["path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[1]
    if "split" not in metadata.columns:
        metadata["split"] = np.where(
            metadata["path"].astype(str).str.replace("\\", "/", regex=False).str.contains("/test/"),
            "test",
            "train",
        )
    metadata["image_id"] = metadata["image_id"].astype(int)
    metadata["abs_path"] = metadata["path"].apply(lambda p: str((data_root / str(p)).resolve()))

    if sam_manifest is not None:
        manifest = pd.read_csv(sam_manifest).copy()
        if "image_id" in manifest.columns:
            manifest["image_id"] = manifest["image_id"].astype(int)
            keep_cols = [
                c
                for c in ["image_id", "mask_path", "mask_full_path", "loose_path", "mask_ok"]
                if c in manifest.columns
            ]
            manifest = manifest[keep_cols].drop_duplicates("image_id")
            export_root = sam_manifest.parent.parent
            for col in ["mask_path", "mask_full_path", "loose_path"]:
                if col in manifest.columns:
                    manifest[col] = manifest[col].apply(lambda x: resolve_export_path(x, export_root))
            metadata = metadata.merge(manifest, on="image_id", how="left")
    for col in ["mask_path", "mask_full_path", "loose_path"]:
        if col not in metadata.columns:
            metadata[col] = ""
    if "mask_ok" not in metadata.columns:
        metadata["mask_ok"] = False
    return metadata


def resolve_export_path(value: object, export_root: Path) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    if not text:
        return ""
    p = Path(text)
    if p.exists():
        return str(p.resolve())
    norm = text.replace("\\", "/")
    marker = "animalclef_sam3_views_cache/"
    if marker in norm:
        rel = norm.split(marker, 1)[1]
        q = export_root / rel
        if q.exists():
            return str(q.resolve())
    for anchor in ["mask_png/", "mask_full_canvas/", "mask_loose_square/"]:
        if anchor in norm:
            rel = norm.split(anchor, 1)[1]
            q = export_root / anchor.strip("/") / rel
            if q.exists():
                return str(q.resolve())
    return text


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(sub.columns) != ["image_id", "cluster"]:
        raise ValueError("Submission columns must be exactly image_id, cluster.")
    if len(sub) != len(sample):
        raise ValueError(f"Submission has {len(sub)} rows; expected {len(sample)}.")
    if sub["cluster"].isna().any():
        raise ValueError("Submission contains null cluster labels.")
    if sub["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Submission image_id order does not match sample_submission.csv.")
    max_len = int(sub["cluster"].astype(str).str.len().max())
    if max_len > 64:
        raise ValueError(f"Cluster label too long: max length {max_len}.")


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0 or mask.max() == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xmin(xs)), int(ymin(ys)), int(xmax(xs)), int(ymax(ys))


def xmin(values: np.ndarray) -> int:
    return int(values.min())


def xmax(values: np.ndarray) -> int:
    return int(values.max())


def ymin(values: np.ndarray) -> int:
    return int(values.min())


def ymax(values: np.ndarray) -> int:
    return int(values.max())


def read_mask(row: pd.Series, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask_path = str(row.get("mask_path", "") or "")
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return (mask > 0).astype(np.uint8)
    # Fallback for missing SAM: use central/full image, but keep this explicit in reports.
    return np.ones((h, w), dtype=np.uint8)


def pad_square_resize(arr: np.ndarray, size: int, fill: np.ndarray = BLACK) -> Image.Image:
    h, w = arr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:] = fill[None, None, :]
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = arr
    img = Image.fromarray(canvas, mode="RGB")
    return img.resize((size, size), Image.Resampling.LANCZOS)


def enhance_subject_only(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if rgb.size == 0:
        return rgb
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() < 64:
        mask_u8 = np.ones(rgb.shape[:2], dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)

    subject_lum = float((l_chan[mask_u8 > 0] / 255.0).mean()) if mask_u8.sum() else float(l_chan.mean() / 255.0)
    gamma = 0.62 if subject_lum < 0.10 else 0.72 if subject_lum < 0.16 else 0.90 if subject_lum < 0.25 else 1.08
    lut = np.array([np.clip((i / 255.0) ** gamma * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    l_gamma = cv2.LUT(l_eq, lut)
    lab2 = lab.copy()
    lab2[:, :, 0] = l_gamma
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    soft = cv2.GaussianBlur(mask_u8.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
    soft = np.clip(soft[..., None], 0.0, 1.0)
    out = rgb.astype(np.float32) * (1.0 - soft) + enhanced.astype(np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def load_lynx_view(row: pd.Series, img_size: int, enhanced: bool = True) -> Image.Image:
    img = Image.open(row["abs_path"]).convert("RGB")
    rgb = np.array(img)
    mask = read_mask(row, rgb.shape[:2])
    if enhanced:
        rgb = enhance_subject_only(rgb, mask)
    return pad_square_resize(rgb, img_size, BLACK)


def brightness_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="brightness"):
        img = Image.open(row.abs_path).convert("L")
        img.thumbnail((256, 256))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        out.append(
            {
                "image_id": int(row.image_id),
                "lum_mean": float(arr.mean()),
                "lum_std": float(arr.std()),
                "dark_frac": float((arr < 0.12).mean()),
                "bright_frac": float((arr > 0.88).mean()),
            }
        )
    return pd.DataFrame(out)


def make_five_crops(img: Image.Image) -> list[Image.Image]:
    img = img.convert("RGB")
    w, h = img.size
    if w != h:
        side = min(w, h)
    else:
        side = w
    crop = max(16, int(round(side * 0.86)))
    coords = [
        (0, 0),
        (side - crop, 0),
        (0, side - crop),
        (side - crop, side - crop),
        ((side - crop) // 2, (side - crop) // 2),
    ]
    crops = []
    for x, y in coords:
        crops.append(img.crop((x, y, x + crop, y + crop)))
    return crops


class SimpleGlobalExtractor:
    """CPU fallback for local smoke tests only."""

    def __init__(self, img_size: int):
        self.img_size = img_size
        self.name = "simple_color_fallback"

    def extract(self, rows: pd.DataFrame) -> np.ndarray:
        feats = []
        for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="simple features"):
            img = load_lynx_view(pd.Series(row._asdict()), self.img_size, enhanced=True)
            arr = np.asarray(img.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1)
            hist = hist / max(hist.sum(), 1.0)
            small = cv2.resize(arr, (8, 8)).reshape(-1)
            feat = np.concatenate([hist, small.astype(np.float32)])
            feats.append(feat)
        return normalize_rows(np.vstack(feats).astype(np.float32))


class EVAGlobalExtractor:
    def __init__(self, model_name: str, pretrained: str, batch_size: int, img_size: int):
        ok = install_if_missing("open_clip", "open_clip_torch")
        if not ok:
            raise ImportError("open_clip_torch unavailable")
        import torch
        import open_clip

        self.torch = torch
        self.open_clip = open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.img_size = img_size
        self.name = f"{model_name}:{pretrained}"
        candidates = [(model_name, pretrained), ("ViT-L-14", "laion2b_s32b_b82k"), ("ViT-B-16", "laion2b_s34b_b88k")]
        last_exc: Exception | None = None
        for name, weights in candidates:
            try:
                print(f"[EVA/CLIP] loading {name} / {weights}")
                model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=weights)
                self.model = model.to(self.device).eval()
                self.preprocess = preprocess
                self.name = f"{name}:{weights}"
                return
            except Exception as exc:
                print(f"[EVA/CLIP] failed {name}/{weights}: {exc}")
                last_exc = exc
        raise RuntimeError(f"Could not load any EVA/CLIP model: {last_exc}")

    def extract(self, rows: pd.DataFrame) -> np.ndarray:
        torch = self.torch
        feats: list[np.ndarray] = []
        batch = []
        batch_owner = []
        with torch.no_grad():
            for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"{self.name}"):
                img = load_lynx_view(pd.Series(row._asdict()), self.img_size, enhanced=True)
                crops = make_five_crops(img)
                for crop in crops:
                    batch.append(self.preprocess(crop))
                    batch_owner.append(int(row.image_id))
                if len(batch) >= self.batch_size:
                    feats.extend(self._flush(batch, batch_owner))
                    batch.clear()
                    batch_owner.clear()
            if batch:
                feats.extend(self._flush(batch, batch_owner))

        # _flush returns one crop feature per crop. Average every 5 crops in row order.
        arr = np.vstack(feats).astype(np.float32)
        if arr.shape[0] != len(rows) * 5:
            # Defensive fallback; should not happen.
            return normalize_rows(arr[: len(rows)])
        arr = arr.reshape(len(rows), 5, -1).mean(axis=1)
        return normalize_rows(arr)

    def _flush(self, batch: list, batch_owner: list[int]) -> list[np.ndarray]:
        torch = self.torch
        x = torch.stack(batch).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        return [v for v in feat.detach().cpu().float().numpy()]


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-7)


@dataclass
class LocalFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray


class ALIKEDExtractor:
    def __init__(self, img_size: int, max_keypoints: int):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size
        self.max_keypoints = max_keypoints
        self.backend = ""

        # Newer Kornia exposes ALIKED directly, but Kaggle's preinstalled
        # Kornia can be older. Prefer Kornia if present; otherwise use the
        # official LightGlue package, which also provides ALIKED.
        try:
            import kornia.feature as KF

            if hasattr(KF, "ALIKED"):
                try:
                    self.model = KF.ALIKED.from_pretrained(
                        "aliked-n16",
                        max_num_keypoints=max_keypoints,
                        detection_threshold=0.18,
                    ).to(self.device).eval()
                except TypeError:
                    self.model = KF.ALIKED.from_pretrained("aliked-n16").to(self.device).eval()
                self.backend = "kornia"
                print(f"[ALIKED] backend=kornia device={self.device}, max_keypoints={max_keypoints}")
                return
            print("[ALIKED] installed kornia has no KF.ALIKED; trying LightGlue backend")
        except Exception as exc:
            print(f"[ALIKED] kornia backend unavailable: {exc}; trying LightGlue backend")

        try:
            try:
                from lightglue import ALIKED as LG_ALIKED
            except Exception:
                try:
                    print("[deps] installing lightglue")
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightglue"], check=True)
                except Exception:
                    print("[deps] pip lightglue failed, trying GitHub install")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/cvg/LightGlue.git"],
                        check=True,
                    )
                from lightglue import ALIKED as LG_ALIKED

            self.model = LG_ALIKED(max_num_keypoints=max_keypoints, detection_threshold=0.18).to(self.device).eval()
            self.backend = "lightglue"
            print(f"[ALIKED] backend=lightglue device={self.device}, max_keypoints={max_keypoints}")
        except Exception as exc:
            raise ImportError(f"Could not initialize ALIKED through Kornia or LightGlue: {exc}") from exc

    def extract(self, rows: pd.DataFrame) -> dict[int, LocalFeatures]:
        out: dict[int, LocalFeatures] = {}
        torch = self.torch
        with torch.no_grad():
            for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="ALIKED features"):
                img = load_lynx_view(pd.Series(row._asdict()), self.img_size, enhanced=True)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
                try:
                    raw = self.model.extract(tensor) if self.backend == "lightglue" else self.model(tensor)
                    features = raw[0] if isinstance(raw, (list, tuple)) else raw
                    kpts = feature_value(features, ["keypoints", "kpts"])
                    desc = feature_value(features, ["descriptors", "desc"])
                except Exception as exc:
                    print(f"[ALIKED] failed image_id={row.image_id}: {exc}")
                    kpts = np.zeros((0, 2), dtype=np.float32)
                    desc = np.zeros((0, 128), dtype=np.float32)
                kpts_np = squeeze_keypoints(tensor_to_numpy(kpts))
                desc_np = squeeze_descriptors(tensor_to_numpy(desc), len(kpts_np))
                if len(kpts_np) > self.max_keypoints:
                    kpts_np = kpts_np[: self.max_keypoints]
                    desc_np = desc_np[: self.max_keypoints]
                out[int(row.image_id)] = LocalFeatures(kpts_np, desc_np)
        return out


def tensor_to_numpy(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def squeeze_keypoints(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    if arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def squeeze_descriptors(arr: np.ndarray, n_keypoints: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or n_keypoints <= 0:
        return np.zeros((0, 128), dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != n_keypoints and arr.ndim == 2 and arr.shape[1] == n_keypoints:
        arr = arr.T
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] != n_keypoints:
        n = min(arr.shape[0], n_keypoints)
        arr = arr[:n]
    return arr.astype(np.float32, copy=False)


def feature_value(features, names: list[str]):
    if isinstance(features, dict):
        for name in names:
            if name in features:
                return features[name]
    for name in names:
        if hasattr(features, name):
            return getattr(features, name)
    return None


def load_local_feature_cache(path: Path) -> dict[int, LocalFeatures] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            raw = pickle.load(f)
        out = {
            int(k): LocalFeatures(np.asarray(v["keypoints"], dtype=np.float32), np.asarray(v["descriptors"], dtype=np.float32))
            for k, v in raw.items()
        }
        print(f"[cache] loaded local features {path}")
        return out
    except Exception as exc:
        print(f"[cache] could not load local features {path}: {exc}")
        return None


def save_local_feature_cache(path: Path, local_features: dict[int, LocalFeatures]) -> None:
    raw = {
        int(k): {"keypoints": v.keypoints.astype(np.float32), "descriptors": v.descriptors.astype(np.float32)}
        for k, v in local_features.items()
    }
    with path.open("wb") as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] wrote local features {path}")


def match_aliked_features(
    fa: LocalFeatures,
    fb: LocalFeatures,
    sim_threshold: float = 0.72,
) -> dict[str, float]:
    if fa.descriptors.size == 0 or fb.descriptors.size == 0 or len(fa.keypoints) < 4 or len(fb.keypoints) < 4:
        return {"local_score": 0.0, "matches": 0, "inliers": 0, "mean_conf": 0.0}
    da = normalize_rows(fa.descriptors)
    db = normalize_rows(fb.descriptors)
    sim = da @ db.T
    nn_ab = sim.argmax(axis=1)
    conf_ab = sim[np.arange(sim.shape[0]), nn_ab]
    nn_ba = sim.argmax(axis=0)
    mutual_mask = np.array([nn_ba[j] == i for i, j in enumerate(nn_ab)], dtype=bool)
    keep = mutual_mask & (conf_ab >= sim_threshold)
    idx_a = np.where(keep)[0]
    idx_b = nn_ab[idx_a]
    matches = int(len(idx_a))
    mean_conf = float(conf_ab[idx_a].mean()) if matches else 0.0
    inliers = 0
    if matches >= 4:
        pts_a = fa.keypoints[idx_a].astype(np.float32)
        pts_b = fb.keypoints[idx_b].astype(np.float32)
        try:
            _, mask_h = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 14.0, maxIters=1000)
            inliers_h = int(mask_h.sum()) if mask_h is not None else 0
        except Exception:
            inliers_h = 0
        try:
            _, mask_a = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=14.0)
            inliers_a = int(mask_a.sum()) if mask_a is not None else 0
        except Exception:
            inliers_a = 0
        inliers = max(inliers_h, inliers_a)
    local_score = 0.58 * min(inliers / 18.0, 1.0) + 0.27 * min(matches / 48.0, 1.0) + 0.15 * np.clip(mean_conf, 0, 1)
    return {
        "local_score": float(local_score),
        "matches": matches,
        "inliers": int(inliers),
        "mean_conf": float(mean_conf),
    }


def orb_pair_score(row_a: pd.Series, row_b: pd.Series, img_size: int) -> dict[str, float]:
    img_a = np.asarray(load_lynx_view(row_a, img_size, enhanced=True).convert("L"))
    img_b = np.asarray(load_lynx_view(row_b, img_size, enhanced=True).convert("L"))
    orb = cv2.ORB_create(nfeatures=800, fastThreshold=7)
    kpa, desa = orb.detectAndCompute(img_a, None)
    kpb, desb = orb.detectAndCompute(img_b, None)
    if desa is None or desb is None or not kpa or not kpb:
        return {"local_score": 0.0, "matches": 0, "inliers": 0, "mean_conf": 0.0}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desa, desb), key=lambda m: m.distance)
    good = matches[: min(len(matches), 80)]
    mean_conf = float(np.mean([1.0 - min(m.distance, 96) / 96.0 for m in good])) if good else 0.0
    inliers = 0
    if len(good) >= 4:
        pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
        pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
        try:
            _, mask = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=14.0)
            inliers = int(mask.sum()) if mask is not None else 0
        except Exception:
            inliers = 0
    local_score = 0.56 * min(inliers / 18.0, 1.0) + 0.29 * min(len(good) / 56.0, 1.0) + 0.15 * mean_conf
    return {"local_score": float(local_score), "matches": int(len(good)), "inliers": int(inliers), "mean_conf": mean_conf}


def sample_train_pairs(train: pd.DataFrame, features: np.ndarray, limit: int) -> list[tuple[int, int, int]]:
    ids = train["image_id"].astype(int).to_numpy()
    labels = train["identity"].astype(str).to_numpy()
    by_label: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    pairs: set[tuple[int, int, int]] = set()
    rng = np.random.default_rng(SEED)

    # Positives: mix random and hard low-similarity within identity.
    sim = features @ features.T
    for label, idxs in by_label.items():
        if len(idxs) < 2:
            continue
        combos = list(itertools.combinations(idxs, 2))
        if len(combos) > 80:
            hard = sorted(combos, key=lambda ab: sim[ab[0], ab[1]])[:30]
            rand_pick = [combos[i] for i in rng.choice(len(combos), size=50, replace=False)]
            combos = hard + rand_pick
        for a, b in combos:
            x, y = sorted((int(ids[a]), int(ids[b])))
            pairs.add((x, y, 1))

    # Negatives: global hard negatives plus some random.
    order = np.argsort(-sim, axis=1)
    for i in range(len(ids)):
        added = 0
        for j in order[i, 1:80]:
            if labels[i] != labels[j]:
                x, y = sorted((int(ids[i]), int(ids[j])))
                pairs.add((x, y, 0))
                added += 1
                if added >= 8:
                    break
    while len([p for p in pairs if p[2] == 0]) < min(limit // 2, 6000):
        a, b = rng.integers(0, len(ids), size=2)
        if a == b or labels[a] == labels[b]:
            continue
        x, y = sorted((int(ids[a]), int(ids[b])))
        pairs.add((x, y, 0))

    positives = [p for p in pairs if p[2] == 1]
    negatives = [p for p in pairs if p[2] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = max(100, limit // 2)
    selected = positives[:half] + negatives[: max(half, limit - min(len(positives), half))]
    rng.shuffle(selected)
    return selected[:limit]


def test_candidate_pairs(test: pd.DataFrame, features: np.ndarray, base: pd.DataFrame, top_k: int, limit: int) -> list[tuple[int, int, str]]:
    ids = test["image_id"].astype(int).to_numpy()
    id_to_cluster = base.set_index("image_id")["cluster"].astype(str).to_dict()
    sim = features @ features.T
    pairs: set[tuple[int, int, str]] = set()

    # All current-cluster pairs are needed for split verification.
    by_cluster: dict[str, list[int]] = {}
    for idx, image_id in enumerate(ids):
        by_cluster.setdefault(id_to_cluster[int(image_id)], []).append(idx)
    for cluster, idxs in by_cluster.items():
        for a, b in itertools.combinations(idxs, 2):
            x, y = sorted((int(ids[a]), int(ids[b])))
            pairs.add((x, y, "within_current"))

    # Global nearest cross-cluster candidates for strict merge rescue.
    order = np.argsort(-sim, axis=1)
    for i in range(len(ids)):
        added = 0
        for j in order[i, 1 : top_k * 5]:
            if id_to_cluster[int(ids[i])] == id_to_cluster[int(ids[j])]:
                continue
            x, y = sorted((int(ids[i]), int(ids[j])))
            pairs.add((x, y, "topk_cross"))
            added += 1
            if added >= top_k:
                break

    out = list(pairs)
    # Preserve all within-current pairs; trim cross-cluster if needed.
    within = [p for p in out if p[2] == "within_current"]
    cross = [p for p in out if p[2] != "within_current"]
    id_to_idx = {int(image_id): i for i, image_id in enumerate(ids)}
    cross = sorted(cross, key=lambda p: -sim[id_to_idx[p[0]], id_to_idx[p[1]]])
    selected = within + cross[: max(0, limit - len(within))]
    return selected


def orientation_relation(oa: str, ob: str) -> str:
    oa = str(oa).lower()
    ob = str(ob).lower()
    flank = {"left", "right"}
    if oa == ob and oa in flank:
        return "same_flank"
    if oa in flank and ob in flank and oa != ob:
        return "opposite_flank"
    if oa == ob and oa in {"front", "back"}:
        return "same_axial"
    if "unknown" in {oa, ob} or not oa or not ob:
        return "unknown"
    return "mixed"


def score_pairs(
    pairs: list[tuple[int, int, int | str]],
    rows: pd.DataFrame,
    features: np.ndarray,
    feature_rows: pd.DataFrame,
    brightness: pd.DataFrame,
    local_features: dict[int, LocalFeatures] | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows_by_id = rows.set_index("image_id", drop=False)
    feat_index = {int(v): i for i, v in enumerate(feature_rows["image_id"].astype(int).tolist())}
    bright = brightness.set_index("image_id").to_dict("index")
    records = []
    for a, b, tag in tqdm(pairs, desc="pair scores"):
        a = int(a)
        b = int(b)
        ia = feat_index[a]
        ib = feat_index[b]
        row_a = rows_by_id.loc[a]
        row_b = rows_by_id.loc[b]
        eva_sim = float(np.dot(features[ia], features[ib]))
        if local_features is not None and not args.disable_aliked:
            loc = match_aliked_features(local_features[a], local_features[b], args.aliked_match_threshold)
        elif args.allow_orb_fallback:
            loc = orb_pair_score(row_a, row_b, args.img_size)
        else:
            loc = {"local_score": 0.0, "matches": 0, "inliers": 0, "mean_conf": 0.0}
        ba = bright.get(a, {})
        bb = bright.get(b, {})
        lum_gap = abs(float(ba.get("lum_mean", 0.0)) - float(bb.get("lum_mean", 0.0)))
        rel = orientation_relation(row_a.get("orientation", ""), row_b.get("orientation", ""))
        rec = {
            "image_id_a": a,
            "image_id_b": b,
            "pair_tag": tag,
            "eva_sim": eva_sim,
            "eva_sim01": float((eva_sim + 1.0) * 0.5),
            "local_score": float(loc["local_score"]),
            "matches": int(loc["matches"]),
            "inliers": int(loc["inliers"]),
            "mean_conf": float(loc["mean_conf"]),
            "lum_gap": float(lum_gap),
            "orientation_a": str(row_a.get("orientation", "")),
            "orientation_b": str(row_b.get("orientation", "")),
            "orientation_relation": rel,
        }
        if isinstance(tag, (int, np.integer)):
            rec["label"] = int(tag)
        records.append(rec)
    return pd.DataFrame(records)


def feature_matrix_for_classifier(df: pd.DataFrame) -> np.ndarray:
    rel = df["orientation_relation"].astype(str)
    arr = np.column_stack(
        [
            df["eva_sim01"].astype(float).to_numpy(),
            df["local_score"].astype(float).to_numpy(),
            np.minimum(df["matches"].astype(float).to_numpy() / 60.0, 1.0),
            np.minimum(df["inliers"].astype(float).to_numpy() / 22.0, 1.0),
            df["mean_conf"].astype(float).to_numpy(),
            np.minimum(df["lum_gap"].astype(float).to_numpy() / 0.20, 1.0),
            rel.eq("same_flank").astype(float).to_numpy(),
            rel.eq("opposite_flank").astype(float).to_numpy(),
            rel.eq("same_axial").astype(float).to_numpy(),
            rel.eq("mixed").astype(float).to_numpy(),
            rel.eq("unknown").astype(float).to_numpy(),
        ]
    )
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def train_calibrator(train_pairs: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    x = feature_matrix_for_classifier(train_pairs)
    y = train_pairs["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 50:
        raise ValueError("Need both positive and negative train pairs for calibration.")
    try:
        x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=SEED,
        )
        clf.fit(x_tr, y_tr)
    except Exception as exc:
        print(f"[calibration] HGB failed, using logistic regression: {exc}")
        x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(x_tr, y_tr)

    pred = clf.predict_proba(x_va)[:, 1]
    stats = {
        "train_pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()),
        "valid_auc": float(roc_auc_score(y_va, pred)) if len(np.unique(y_va)) > 1 else 0.5,
        "valid_ap": float(average_precision_score(y_va, pred)) if len(np.unique(y_va)) > 1 else float(y_va.mean()),
    }
    return clf, stats


class UF:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}
        self.size = {int(v): 1 for v in values}

    def find(self, value: int) -> int:
        value = int(value)
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


def compact_labels(groups: dict[int, int], species: str) -> dict[int, str]:
    by_root: dict[int, list[int]] = {}
    for image_id, root in groups.items():
        by_root.setdefault(int(root), []).append(int(image_id))
    labels: dict[int, str] = {}
    for idx, members in enumerate(sorted(by_root.values(), key=lambda xs: (min(xs), len(xs)))):
        label = f"cluster_{species}_{idx}"
        for image_id in members:
            labels[image_id] = label
    return labels


def build_lynx_variant(
    base: pd.DataFrame,
    lynx_test: pd.DataFrame,
    pair_scores: pd.DataFrame,
    profile: str,
) -> tuple[dict[int, str], pd.DataFrame]:
    ids = lynx_test["image_id"].astype(int).tolist()
    cluster_by_id = base.set_index("image_id")["cluster"].astype(str).to_dict()
    within = pair_scores[pair_scores["same_current_cluster"]].copy()
    cross = pair_scores[~pair_scores["same_current_cluster"]].copy()

    # Split phase: never create a split unless the calibrated graph gives a
    # clear reason. The safe profile only isolates very weak outliers.
    split_thr = {"split_safe": 0.42, "split_balanced": 0.50, "splitmerge_strict": 0.48}[profile]
    min_cluster = {"split_safe": 12, "split_balanced": 8, "splitmerge_strict": 8}[profile]
    uf = UF(ids)
    actions = []

    for cluster, members_df in base[base["image_id"].isin(ids)].groupby("cluster"):
        members = members_df["image_id"].astype(int).tolist()
        if len(members) < min_cluster:
            for a, b in itertools.combinations(members, 2):
                uf.union(a, b)
            continue
        sub = within[
            within["image_id_a"].isin(members)
            & within["image_id_b"].isin(members)
            & (within["match_prob"].astype(float) >= split_thr)
        ]
        local_uf = UF(members)
        for row in sub.itertuples(index=False):
            local_uf.union(int(row.image_id_a), int(row.image_id_b))
        comps: dict[int, list[int]] = {}
        for image_id in members:
            comps.setdefault(local_uf.find(image_id), []).append(image_id)
        comps_sorted = sorted(comps.values(), key=len, reverse=True)
        if len(comps_sorted) <= 1:
            for a, b in itertools.combinations(members, 2):
                uf.union(a, b)
            continue
        # Safe mode: split only isolated low-confidence singleton/duo outliers.
        if profile == "split_safe":
            major = set(comps_sorted[0])
            for a, b in itertools.combinations(major, 2):
                uf.union(a, b)
            for comp in comps_sorted[1:]:
                max_to_major = within[
                    (
                        within["image_id_a"].isin(comp)
                        & within["image_id_b"].isin(major)
                    )
                    | (
                        within["image_id_b"].isin(comp)
                        & within["image_id_a"].isin(major)
                    )
                ]["match_prob"].max()
                max_to_major = float(max_to_major) if not pd.isna(max_to_major) else 0.0
                if len(comp) <= 2 and max_to_major < 0.34:
                    for a, b in itertools.combinations(comp, 2):
                        uf.union(a, b)
                    actions.append(
                        {
                            "profile": profile,
                            "action": "split_outlier",
                            "source_cluster": cluster,
                            "component_size": len(comp),
                            "members": " ".join(map(str, sorted(comp))),
                            "max_prob_to_major": max_to_major,
                        }
                    )
                else:
                    for a in comp:
                        for b in major:
                            uf.union(a, b)
                    for a, b in itertools.combinations(comp, 2):
                        uf.union(a, b)
        else:
            for comp in comps_sorted:
                for a, b in itertools.combinations(comp, 2):
                    uf.union(a, b)
            actions.append(
                {
                    "profile": profile,
                    "action": "split_component",
                    "source_cluster": cluster,
                    "component_sizes": " ".join(map(str, [len(c) for c in comps_sorted])),
                }
            )

    # Merge phase: only for splitmerge_strict, high probability, orientation-aware.
    if profile == "splitmerge_strict":
        max_cluster_size = 84  # Lynx train p90 is ~84; current max 67.
        candidates = cross.sort_values(["match_prob", "local_score", "inliers"], ascending=False)
        for row in candidates.itertuples(index=False):
            a = int(row.image_id_a)
            b = int(row.image_id_b)
            if uf.find(a) == uf.find(b):
                continue
            rel = str(row.orientation_relation)
            prob = float(row.match_prob)
            local = float(row.local_score)
            inliers = int(row.inliers)
            eva = float(row.eva_sim01)
            if rel == "same_flank":
                ok = prob >= 0.88 and (local >= 0.42 or inliers >= 10)
            elif rel == "opposite_flank":
                ok = prob >= 0.965 and eva >= 0.93 and local >= 0.28
            else:
                ok = prob >= 0.91 and (local >= 0.38 or eva >= 0.94)
            if not ok:
                continue
            size_after = uf.size[uf.find(a)] + uf.size[uf.find(b)]
            if size_after > max_cluster_size:
                continue
            before_a = uf.find(a)
            before_b = uf.find(b)
            uf.union(a, b)
            actions.append(
                {
                    "profile": profile,
                    "action": "strict_merge",
                    "image_id_a": a,
                    "image_id_b": b,
                    "source_cluster_a": cluster_by_id[a],
                    "source_cluster_b": cluster_by_id[b],
                    "root_a": before_a,
                    "root_b": before_b,
                    "match_prob": prob,
                    "local_score": local,
                    "inliers": inliers,
                    "eva_sim01": eva,
                    "orientation_relation": rel,
                    "size_after": size_after,
                }
            )

    groups = {image_id: uf.find(image_id) for image_id in ids}
    return compact_labels(groups, LYNX), pd.DataFrame(actions)


def apply_lynx_labels(base: pd.DataFrame, sample: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    out = base.copy()
    mask = out["image_id"].astype(int).isin(labels)
    out.loc[mask, "cluster"] = out.loc[mask, "image_id"].astype(int).map(labels)
    out = sample[["image_id"]].merge(out, on="image_id", how="left")
    if out["cluster"].isna().any():
        missing = out[out["cluster"].isna()]["image_id"].head().tolist()
        raise ValueError(f"Missing labels for sample image ids: {missing}")
    return out[["image_id", "cluster"]]


def shape_report(sub: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    tmp = sub.merge(test_rows[["image_id", "dataset"]], on="image_id", how="left")
    rows = []
    for species, g in tmp.groupby("dataset"):
        sizes = g.groupby("cluster").size()
        rows.append(
            {
                "species": species,
                "n_images": int(len(g)),
                "n_clusters": int(sizes.size),
                "max_cluster_size": int(sizes.max()),
                "singletons": int((sizes == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_visualizations(
    out_dir: Path,
    lynx_test: pd.DataFrame,
    pair_scores: pd.DataFrame,
    max_pairs: int = 12,
    img_size: int = 512,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = lynx_test.set_index("image_id", drop=False)
    pairs = pair_scores.sort_values(["match_prob", "local_score"], ascending=False).head(max_pairs)
    for idx, row in enumerate(pairs.itertuples(index=False)):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        img_a0 = ImageOps.contain(Image.open(rows_by_id.loc[a, "abs_path"]).convert("RGB"), (img_size, img_size))
        img_b0 = ImageOps.contain(Image.open(rows_by_id.loc[b, "abs_path"]).convert("RGB"), (img_size, img_size))
        img_a1 = load_lynx_view(rows_by_id.loc[a], img_size, enhanced=True)
        img_b1 = load_lynx_view(rows_by_id.loc[b], img_size, enhanced=True)
        tile_w = img_size
        tile_h = img_size
        canvas = Image.new("RGB", (tile_w * 4, tile_h + 44), (240, 240, 240))
        for j, img in enumerate([img_a0, img_a1, img_b0, img_b1]):
            x = j * tile_w
            bg = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
            bg.paste(img, ((tile_w - img.width) // 2, (tile_h - img.height) // 2))
            canvas.paste(bg, (x, 0))
        draw = ImageDraw.Draw(canvas)
        text = (
            f"{a} vs {b} prob={row.match_prob:.3f} local={row.local_score:.3f} "
            f"inliers={row.inliers} rel={row.orientation_relation}"
        )
        draw.rectangle((0, tile_h, tile_w * 4, tile_h + 44), fill=(245, 245, 245))
        draw.text((10, tile_h + 14), text, fill=(20, 20, 20))
        canvas.save(out_dir / f"lynx_pair_{idx:02d}_{a}_{b}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    seed_everything()
    if args.smoke:
        args.train_pair_limit = min(args.train_pair_limit, 600)
        args.test_pair_limit = min(args.test_pair_limit, 1200)
        args.top_k = min(args.top_k, 8)
        args.disable_eva = True
        args.disable_aliked = True

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    submissions_dir = output_root / "submissions"
    viz_dir = output_root / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_data_root(args.data_root)
    current_best_path = find_current_best(args.current_best)
    sam_manifest = find_sam_manifest(args.sam_manifest)
    print(f"[paths] data_root={data_root}")
    print(f"[paths] current_best={current_best_path}")
    print(f"[paths] sam_manifest={sam_manifest}")

    sample = pd.read_csv(data_root / "sample_submission.csv")
    sample["image_id"] = sample["image_id"].astype(int)
    metadata = prepare_metadata(data_root, sam_manifest)
    test_rows = metadata[metadata["split"].eq("test")].copy()
    lynx_train = metadata[(metadata["dataset"].eq(LYNX)) & (metadata["split"].eq("train"))].copy()
    lynx_test = metadata[(metadata["dataset"].eq(LYNX)) & (metadata["split"].eq("test"))].copy()
    if args.smoke:
        lynx_train = lynx_train.head(240).copy()
        lynx_test = lynx_test.head(120).copy()
    all_lynx = pd.concat([lynx_train, lynx_test], ignore_index=True)
    base = load_submission(current_best_path)
    validate_submission(sample[["image_id"]].merge(base, on="image_id", how="left"), sample)

    print(f"[data] lynx_train={len(lynx_train)} lynx_test={len(lynx_test)}")
    print(f"[data] mask coverage all lynx={float(all_lynx['mask_ok'].fillna(False).astype(bool).mean()):.4f}")

    t0 = time.time()
    feature_path = reports_dir / f"{VERSION}_global_features.npy"
    extractor_name = "cache"
    if feature_path.exists():
        cached = np.load(feature_path)
        if cached.shape[0] == len(all_lynx):
            global_features = normalize_rows(cached)
            print(f"[cache] loaded global features {feature_path} shape={global_features.shape}")
        else:
            print(f"[cache] ignoring global feature cache with wrong row count: {cached.shape[0]} != {len(all_lynx)}")
            global_features = None
    else:
        global_features = None
    if global_features is None:
        if args.disable_eva:
            global_extractor = SimpleGlobalExtractor(args.img_size)
        else:
            global_extractor = EVAGlobalExtractor(args.eva_model, args.eva_pretrained, args.eva_batch_size, args.img_size)
        extractor_name = global_extractor.name
        global_features = global_extractor.extract(all_lynx)
        np.save(feature_path, global_features)
        print(f"[features] wrote {feature_path} shape={global_features.shape} extractor={global_extractor.name}")

    brightness_path = reports_dir / f"{VERSION}_brightness.csv"
    if brightness_path.exists():
        brightness = pd.read_csv(brightness_path)
        print(f"[cache] loaded brightness {brightness_path}")
    else:
        brightness = brightness_metrics(all_lynx)
        brightness.to_csv(brightness_path, index=False)

    local_features: dict[int, LocalFeatures] | None = None
    if not args.disable_aliked:
        local_cache_path = reports_dir / f"{VERSION}_aliked_features.pkl"
        local_features = load_local_feature_cache(local_cache_path)
    if local_features is None and not args.disable_aliked:
        try:
            aliked = ALIKEDExtractor(args.img_size, args.max_aliked_keypoints)
            local_features = aliked.extract(all_lynx)
            save_local_feature_cache(local_cache_path, local_features)
        except Exception as exc:
            if not args.allow_orb_fallback:
                raise
            print(f"[ALIKED] unavailable, ORB fallback enabled: {exc}")
            local_features = None

    train_features = global_features[: len(lynx_train)]
    test_features = global_features[len(lynx_train) :]
    train_pairs_idx = sample_train_pairs(lynx_train, train_features, args.train_pair_limit)
    train_pairs = score_pairs(
        train_pairs_idx,
        all_lynx,
        global_features,
        all_lynx,
        brightness,
        local_features,
        args,
    )
    train_pairs.to_csv(reports_dir / f"{VERSION}_train_pairs.csv", index=False)
    clf, cal_stats = train_calibrator(train_pairs)
    print(f"[calibration] {json.dumps(cal_stats, indent=2)}")

    candidate_idx = test_candidate_pairs(lynx_test, test_features, base, args.top_k, args.test_pair_limit)
    test_pairs = score_pairs(
        candidate_idx,
        all_lynx,
        global_features,
        all_lynx,
        brightness,
        local_features,
        args,
    )
    base_cluster = base.set_index("image_id")["cluster"].astype(str).to_dict()
    test_pairs["cluster_a"] = test_pairs["image_id_a"].astype(int).map(base_cluster)
    test_pairs["cluster_b"] = test_pairs["image_id_b"].astype(int).map(base_cluster)
    test_pairs["same_current_cluster"] = test_pairs["cluster_a"].eq(test_pairs["cluster_b"])
    test_pairs["match_prob"] = clf.predict_proba(feature_matrix_for_classifier(test_pairs))[:, 1]
    test_pairs = test_pairs.sort_values(["match_prob", "local_score", "inliers"], ascending=False).reset_index(drop=True)
    test_pairs.to_csv(reports_dir / f"{VERSION}_test_pair_scores.csv", index=False)

    written: dict[str, str] = {}
    action_frames = []
    for profile in ["split_safe", "split_balanced", "splitmerge_strict"]:
        labels, actions = build_lynx_variant(base, lynx_test, test_pairs, profile)
        sub = apply_lynx_labels(base, sample, labels)
        validate_submission(sub, sample)
        name = f"submission_{VERSION}_{profile}.csv"
        sub.to_csv(submissions_dir / name, index=False)
        # Also write top-level safest candidate as submission.csv.
        if profile == "split_safe":
            sub.to_csv(output_root / "submission.csv", index=False)
        report = shape_report(sub, test_rows)
        report["profile"] = profile
        report.to_csv(reports_dir / f"{VERSION}_{profile}_shape_report.csv", index=False)
        if len(actions):
            actions.to_csv(reports_dir / f"{VERSION}_{profile}_actions.csv", index=False)
            action_frames.append(actions)
        written[profile] = str(submissions_dir / name)
        print(f"\n[{profile}]")
        print(report.to_string(index=False))

    action_all = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    action_all.to_csv(reports_dir / f"{VERSION}_all_actions.csv", index=False)
    if args.save_visualizations:
        save_visualizations(viz_dir, lynx_test, test_pairs, img_size=args.img_size)

    base_report = shape_report(sample[["image_id"]].merge(base, on="image_id", how="left"), test_rows)
    base_report.to_csv(reports_dir / f"{VERSION}_base_shape_report.csv", index=False)
    run_report = {
        "version": VERSION,
        "data_root": str(data_root),
        "current_best": str(current_best_path),
        "sam_manifest": str(sam_manifest) if sam_manifest else None,
        "global_extractor": extractor_name,
        "aliked_enabled": bool(local_features is not None and not args.disable_aliked),
        "calibration": cal_stats,
        "n_train_pairs": int(len(train_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "written": written,
        "recommended_first": str(output_root / "submission.csv"),
        "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
        "notes": [
            "Lynx-only postprocessor; other species copied from current best input.",
            "Uses original images with SAM-guided subject-only enhancement, not white-background SAM crops.",
            "Default submission.csv is split_safe. Inspect reports before submitting split_balanced or splitmerge_strict.",
        ],
    }
    (reports_dir / f"{VERSION}_run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    print("\n[done]")
    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
