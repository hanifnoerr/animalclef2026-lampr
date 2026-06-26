#!/usr/bin/env python3
"""
AnimalCLEF2026 SeaTurtle EVA02/CLIP + ALIKED merge-only branch.

This is a SeaTurtle-only incremental postprocessor over the current best
0.34411 partition. It keeps Lynx, Salamander, and Texas unchanged, then tries
to recover over-split SeaTurtle identities using the 2025-winning global/local
recipe:

* original SeaTurtle images, no SAM foreground replacement
* original and horizontal-flip views because test orientation metadata is absent
* EVA02/CLIP 5-crop global descriptors
* ALIKED local feature matching with geometric verification
* train-identity calibrated pair probabilities
* merge-only graph postprocessing with train-distribution max-size caps

The script expects the 0.34411 Salamander EVA/ALIKED output as the base. If it
is not attached, pass --current-best explicitly.
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

VERSION = "seaturtle_eva_aliked_mergeonly_v20260430"
SALAMANDER = "SeaTurtleID2022"
LYNX = "LynxID2025"
SEED = 20260430
PAD = np.array([128, 128, 128], dtype=np.uint8)
TURTLE_VIEWS = ["orig", "hflip"]


CURRENT_BEST_NAMES = [
    "submission_034411_salamander_eva_aliked_mergeonly_v20260430.csv",
    "submission_salamander_eva_aliked_mergeonly_v20260430_ultra_p90.csv",
    "submission_salamander_eva_aliked_mergeonly_v20260430_strict_p90.csv",
    "submission_salamander_eva_aliked_mergeonly_v20260430_p95_gamble.csv",
    "submission_032684_lynx_aliked_prob098_mergeonly_v20260430.csv",
    "submission_lb032903_lynx_aliked_prob098_mergeonly_v20260430.csv",
    "submission.csv",
]
BASE_032684_NAMES = [
    "submission_final_texas_boundary_splitonly_balanced_from_032583_v20260430.csv",
    "submission.csv",
]
LYNX_PAIR_SCORE_NAME = "lynx_lowlight_eva_aliked_v20260430_test_pair_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--current-best", type=str, default=None)
    parser.add_argument("--base-032684", type=str, default=None)
    parser.add_argument("--lynx-pair-scores", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=f"/kaggle/working/animalclef_{VERSION}")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--train-pair-limit", type=int, default=7000)
    parser.add_argument("--test-pair-limit", type=int, default=14000)
    parser.add_argument("--max-aliked-keypoints", type=int, default=512)
    parser.add_argument("--aliked-match-threshold", type=float, default=0.72)
    parser.add_argument("--eva-batch-size", type=int, default=32)
    parser.add_argument("--eva-model", type=str, default="EVA02-B-16")
    parser.add_argument("--eva-pretrained", type=str, default="merged2b_s8b_b131k")
    parser.add_argument("--disable-eva", action="store_true")
    parser.add_argument("--disable-aliked", action="store_true")
    parser.add_argument("--cache-aliked", action="store_true", help="Write ALIKED feature pickles. Off by default for SeaTurtle because the train set is large.")
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
        "/kaggle/input/competitions/animal-clef-2026",
        "/kaggle/input/animal-clef-2026",
        "/kaggle/input",
        "/kaggle/working",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/animal-clef-2026",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/current_wildfusion_graph_v20260423",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/kaggle_output_salamander_eva02_clip_aliked_mergeonly_v315549470",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/kaggle_output_lynx_lowlight_eva_aliked_v315501274",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026",
        ".",
    ]:
        if value:
            p = Path(value)
            if p.exists():
                roots.append(p)
    out = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def find_data_root(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if (p / "metadata.csv").exists() and (p / "sample_submission.csv").exists():
            return p.resolve()
        raise FileNotFoundError(f"--data-root is not an AnimalCLEF root: {p}")
    for root in candidate_roots():
        if (root / "metadata.csv").exists() and (root / "sample_submission.csv").exists():
            return root.resolve()
    for root in candidate_roots():
        try:
            for meta in root.rglob("metadata.csv"):
                if (meta.parent / "sample_submission.csv").exists():
                    return meta.parent.resolve()
        except Exception:
            pass
    raise FileNotFoundError("Could not locate AnimalCLEF2026 data root.")


def rank_current_best(path: Path) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    penalty = 0
    if path.name == "submission.csv" and not any(
        token in text
        for token in [
            "034411",
            "0-34411",
            "0_34411",
            "salamander-eva02-clip-aliked-merge-only",
            "salamander_eva_aliked_mergeonly",
            "lb032903",
            "0-32903",
            "0_32903",
            "lynx_aliked_reproducer",
            "lynxaliked-merge",
        ]
    ):
        penalty += 1000
    if "local_smoke" in text or "__pycache__" in text:
        penalty += 100
    if "034411" in path.name.lower() or "salamander_eva_aliked_mergeonly" in text or "salamander-eva02-clip-aliked-merge-only" in text:
        penalty -= 140
    if "032684_lynx_aliked" in path.name.lower() or "lb032903" in text:
        penalty -= 80
    if "animalclef_lb032903_lynx_aliked_reproducer" in text:
        penalty -= 70
    if "current_wildfusion_graph" in text:
        penalty -= 20
    if "/kaggle/input/" in text:
        penalty -= 10
    if path.name == "submission.csv":
        penalty += 10
    return penalty, len(text), text


def rank_base_032684(path: Path) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    penalty = 0
    if "local_smoke" in text or "__pycache__" in text:
        penalty += 100
    if "final-texas-boundary-split-only" in text or "animalclef_texas_boundary_splitonly_final" in text:
        penalty -= 60
    if "submission_final_texas_boundary_splitonly" in path.name.lower():
        penalty -= 40
    if "/kaggle/input/" in text:
        penalty -= 10
    if path.name == "submission.csv":
        penalty += 8
    return penalty, len(text), text


def rank_pair_path(path: Path) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    penalty = 0
    if "local_smoke" in text or "__pycache__" in text:
        penalty += 100
    if "lynx-low-light-eva-clip-aliked" in text or "animalclef_lynx_lowlight_eva_aliked" in text:
        penalty -= 60
    if "/reports/" in text:
        penalty -= 10
    if "/kaggle/input/" in text:
        penalty -= 10
    return penalty, len(text), text


def find_optional_file(names: Iterable[str], arg: str | None, ranker) -> Path | None:
    if arg:
        p = Path(arg)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Path does not exist: {p}")
    hits = []
    for root in candidate_roots():
        for name in names:
            direct = root / name
            if direct.exists():
                hits.append(direct)
            try:
                hits.extend(root.rglob(name))
            except Exception:
                pass
    hits = sorted({p.resolve() for p in hits if p.exists()}, key=ranker)
    return hits[0] if hits else None


def find_file(names: Iterable[str], arg: str | None, ranker) -> Path:
    hit = find_optional_file(names, arg, ranker)
    if hit is None:
        raise FileNotFoundError(f"Could not find any of: {list(names)}")
    return hit


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns.")
    df = df[["image_id", "cluster"]].copy()
    df["image_id"] = df["image_id"].astype(int)
    df["cluster"] = df["cluster"].astype(str)
    if df["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id values in {path}")
    return df


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(sub.columns) != ["image_id", "cluster"]:
        raise ValueError("Submission columns must be exactly image_id, cluster.")
    if len(sub) != len(sample):
        raise ValueError(f"Submission rows {len(sub)} != sample rows {len(sample)}")
    if sub["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Submission image order does not match sample_submission.csv")
    if sub["cluster"].isna().any():
        raise ValueError("Submission contains null cluster labels.")
    max_len = int(sub["cluster"].astype(str).str.len().max())
    if max_len > 64:
        raise ValueError(f"Cluster labels too long: max length {max_len}")


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


def compact_species_labels(sub: pd.DataFrame, sample: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    sample_ids = set(sample["image_id"].astype(int))
    for species, group in test_rows[test_rows["image_id"].isin(sample_ids)].groupby("dataset", sort=True):
        ids = set(group["image_id"].astype(int))
        part = sub[sub["image_id"].astype(int).isin(ids)].copy()
        groups = []
        for _, members in part.groupby("cluster"):
            groups.append(sorted(members["image_id"].astype(int).tolist()))
        labels = {}
        for idx, members in enumerate(sorted(groups, key=lambda xs: (min(xs), len(xs), max(xs)))):
            for image_id in members:
                labels[image_id] = f"cluster_{species}_{idx}"
        parts.append(pd.DataFrame({"image_id": sorted(ids), "cluster": [labels[i] for i in sorted(ids)]}))
    final = pd.concat(parts, ignore_index=True)
    final = sample[["image_id"]].merge(final, on="image_id", how="left")
    return final[["image_id", "cluster"]]


def build_032903_from_components(
    base_032684: pd.DataFrame,
    lynx_pairs: pd.DataFrame,
    sample: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> pd.DataFrame:
    lynx_ids = (
        test_rows[test_rows["dataset"].eq(LYNX) & test_rows["image_id"].isin(sample["image_id"])]
        ["image_id"]
        .astype(int)
        .tolist()
    )
    current = base_032684[base_032684["image_id"].astype(int).isin(lynx_ids)].copy()
    uf = UF(lynx_ids)
    for _, group in current.groupby("cluster"):
        ids = group["image_id"].astype(int).tolist()
        for a, b in itertools.combinations(ids, 2):
            uf.union(a, b)

    same_current = bool_series(lynx_pairs["same_current_cluster"])
    accepted = lynx_pairs[
        (~same_current)
        & (~lynx_pairs["orientation_relation"].astype(str).eq("opposite_flank"))
        & (lynx_pairs["local_score"].astype(float) >= 0.96)
        & (lynx_pairs["inliers"].astype(float) >= 50)
        & (lynx_pairs["match_prob"].astype(float) >= 0.98)
    ].copy()
    accepted = accepted.sort_values(["local_score", "inliers", "match_prob"], ascending=False)
    for row in accepted.itertuples(index=False):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        if a not in uf.parent or b not in uf.parent or uf.find(a) == uf.find(b):
            continue
        size_after = uf.size[uf.find(a)] + uf.size[uf.find(b)]
        if size_after <= 84:
            uf.union(a, b)

    groups = {}
    for image_id in lynx_ids:
        groups.setdefault(uf.find(image_id), []).append(image_id)
    labels = {}
    for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
        for image_id in members:
            labels[image_id] = f"cluster_{LYNX}_{idx}"
    out = base_032684.copy()
    mask = out["image_id"].astype(int).isin(labels)
    out.loc[mask, "cluster"] = out.loc[mask, "image_id"].astype(int).map(labels)
    return compact_species_labels(out, sample, test_rows)


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def prepare_metadata(data_root: Path) -> pd.DataFrame:
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
    return metadata


def orientation_normalize(img: Image.Image, orientation: str) -> Image.Image:
    orientation = str(orientation).lower()
    if orientation == "right":
        return img.rotate(-90, expand=True)
    if orientation == "left":
        return img.rotate(90, expand=True)
    if orientation == "bottom":
        return img.rotate(180, expand=True)
    return img


def pad_square_resize(img: Image.Image, size: int, fill: np.ndarray = PAD) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:] = fill[None, None, :]
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = rgb
    return Image.fromarray(canvas).resize((size, size), Image.Resampling.LANCZOS)


def load_salamander_view(row: pd.Series, img_size: int, view: str) -> Image.Image:
    img = Image.open(row["abs_path"]).convert("RGB")
    if view == "hflip":
        img = ImageOps.mirror(img)
    elif view == "center":
        w, h = img.size
        crop = int(round(min(w, h) * 0.88))
        left = max(0, (w - crop) // 2)
        top = max(0, (h - crop) // 2)
        img = img.crop((left, top, left + crop, top + crop))
    return pad_square_resize(img, img_size, PAD)


def make_five_crops(img: Image.Image) -> list[Image.Image]:
    img = img.convert("RGB")
    side = min(img.size)
    crop = max(16, int(round(side * 0.86)))
    coords = [
        (0, 0),
        (side - crop, 0),
        (0, side - crop),
        (side - crop, side - crop),
        ((side - crop) // 2, (side - crop) // 2),
    ]
    return [img.crop((x, y, x + crop, y + crop)) for x, y in coords]


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-7)


class SimpleGlobalExtractor:
    def __init__(self, img_size: int, view: str):
        self.img_size = img_size
        self.view = view
        self.name = f"simple_color_fallback:{view}"

    def extract(self, rows: pd.DataFrame) -> np.ndarray:
        feats = []
        for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"simple {self.view}"):
            img = load_salamander_view(pd.Series(row._asdict()), self.img_size, self.view)
            arr = np.asarray(img.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).reshape(-1)
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
            small = cv2.resize(arr, (8, 8)).reshape(-1)
            feat = np.concatenate([hist_h, hist_s, hist_v, small]).astype(np.float32)
            feat /= max(float(np.linalg.norm(feat)), 1e-7)
            feats.append(feat)
        return normalize_rows(np.vstack(feats))


class EVAGlobalExtractor:
    def __init__(self, model_name: str, pretrained: str, batch_size: int, img_size: int, view: str):
        ok = install_if_missing("open_clip", "open_clip_torch")
        if not ok:
            raise ImportError("open_clip_torch unavailable")
        import torch
        import open_clip

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.img_size = img_size
        self.view = view
        candidates = [(model_name, pretrained), ("ViT-L-14", "laion2b_s32b_b82k"), ("ViT-B-16", "laion2b_s34b_b88k")]
        last_exc = None
        for name, weights in candidates:
            try:
                print(f"[EVA/CLIP] loading {name} / {weights} for {view}")
                model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=weights)
                self.model = model.to(self.device).eval()
                self.preprocess = preprocess
                self.name = f"{name}:{weights}:{view}"
                return
            except Exception as exc:
                print(f"[EVA/CLIP] failed {name}/{weights}: {exc}")
                last_exc = exc
        raise RuntimeError(f"Could not load EVA/CLIP model: {last_exc}")

    def extract(self, rows: pd.DataFrame) -> np.ndarray:
        torch = self.torch
        crop_feats = []
        batch = []
        with torch.no_grad():
            for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=self.name):
                img = load_salamander_view(pd.Series(row._asdict()), self.img_size, self.view)
                for crop in make_five_crops(img):
                    batch.append(self.preprocess(crop))
                if len(batch) >= self.batch_size:
                    crop_feats.extend(self._flush(batch))
                    batch.clear()
            if batch:
                crop_feats.extend(self._flush(batch))
        arr = np.vstack(crop_feats).astype(np.float32)
        arr = arr.reshape(len(rows), 5, -1).mean(axis=1)
        return normalize_rows(arr)

    def _flush(self, batch: list) -> list[np.ndarray]:
        torch = self.torch
        x = torch.stack(batch).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        return [v for v in feat.detach().cpu().float().numpy()]


@dataclass
class LocalFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray


class ALIKEDExtractor:
    def __init__(self, img_size: int, max_keypoints: int, view: str):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size
        self.max_keypoints = max_keypoints
        self.view = view
        self.backend = ""

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
                print(f"[ALIKED] backend=kornia view={view} device={self.device}")
                return
            print("[ALIKED] kornia has no ALIKED; trying LightGlue")
        except Exception as exc:
            print(f"[ALIKED] kornia unavailable: {exc}; trying LightGlue")

        try:
            try:
                from lightglue import ALIKED as LG_ALIKED
            except Exception:
                try:
                    print("[deps] installing lightglue")
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightglue"], check=True)
                except Exception:
                    print("[deps] pip lightglue failed; trying GitHub")
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/cvg/LightGlue.git"], check=True)
                from lightglue import ALIKED as LG_ALIKED
            self.model = LG_ALIKED(max_num_keypoints=max_keypoints, detection_threshold=0.18).to(self.device).eval()
            self.backend = "lightglue"
            print(f"[ALIKED] backend=lightglue view={view} device={self.device}")
        except Exception as exc:
            raise ImportError(f"Could not initialize ALIKED: {exc}") from exc

    def extract(self, rows: pd.DataFrame) -> dict[int, LocalFeatures]:
        out = {}
        torch = self.torch
        with torch.no_grad():
            for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"ALIKED {self.view}"):
                img = load_salamander_view(pd.Series(row._asdict()), self.img_size, self.view)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
                try:
                    raw = self.model.extract(tensor) if self.backend == "lightglue" else self.model(tensor)
                    features = raw[0] if isinstance(raw, (list, tuple)) else raw
                    kpts = feature_value(features, ["keypoints", "kpts"])
                    desc = feature_value(features, ["descriptors", "desc"])
                    kpts_np = squeeze_keypoints(tensor_to_numpy(kpts))
                    desc_np = squeeze_descriptors(tensor_to_numpy(desc), len(kpts_np))
                except Exception as exc:
                    print(f"[ALIKED] failed image_id={row.image_id} view={self.view}: {exc}")
                    kpts_np = np.zeros((0, 2), dtype=np.float32)
                    desc_np = np.zeros((0, 128), dtype=np.float32)
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
        arr = arr[: min(arr.shape[0], n_keypoints)]
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


def match_local(fa: LocalFeatures, fb: LocalFeatures, sim_threshold: float) -> dict[str, float]:
    if fa.descriptors.size == 0 or fb.descriptors.size == 0 or len(fa.keypoints) < 4 or len(fb.keypoints) < 4:
        return {"local_score": 0.0, "matches": 0, "inliers": 0, "mean_conf": 0.0}
    da = normalize_rows(fa.descriptors)
    db = normalize_rows(fb.descriptors)
    sim = da @ db.T
    nn_ab = sim.argmax(axis=1)
    conf_ab = sim[np.arange(sim.shape[0]), nn_ab]
    nn_ba = sim.argmax(axis=0)
    mutual = np.array([nn_ba[j] == i for i, j in enumerate(nn_ab)], dtype=bool)
    keep = mutual & (conf_ab >= sim_threshold)
    idx_a = np.where(keep)[0]
    idx_b = nn_ab[idx_a]
    matches = int(len(idx_a))
    mean_conf = float(conf_ab[idx_a].mean()) if matches else 0.0
    inliers = 0
    if matches >= 4:
        pts_a = fa.keypoints[idx_a].astype(np.float32)
        pts_b = fb.keypoints[idx_b].astype(np.float32)
        try:
            _, mask_h = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 12.0, maxIters=1000)
            inliers_h = int(mask_h.sum()) if mask_h is not None else 0
        except Exception:
            inliers_h = 0
        try:
            _, mask_a = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=12.0)
            inliers_a = int(mask_a.sum()) if mask_a is not None else 0
        except Exception:
            inliers_a = 0
        inliers = max(inliers_h, inliers_a)
    local_score = 0.58 * min(inliers / 16.0, 1.0) + 0.27 * min(matches / 42.0, 1.0) + 0.15 * np.clip(mean_conf, 0, 1)
    return {"local_score": float(local_score), "matches": matches, "inliers": int(inliers), "mean_conf": float(mean_conf)}


def orb_pair_score(row_a: pd.Series, row_b: pd.Series, img_size: int, view: str) -> dict[str, float]:
    img_a = np.asarray(load_salamander_view(row_a, img_size, view).convert("L"))
    img_b = np.asarray(load_salamander_view(row_b, img_size, view).convert("L"))
    orb = cv2.ORB_create(nfeatures=800, fastThreshold=7)
    kpa, desa = orb.detectAndCompute(img_a, None)
    kpb, desb = orb.detectAndCompute(img_b, None)
    if desa is None or desb is None or not kpa or not kpb:
        return {"local_score": 0.0, "matches": 0, "inliers": 0, "mean_conf": 0.0}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desa, desb), key=lambda m: m.distance)[:80]
    mean_conf = float(np.mean([1.0 - min(m.distance, 96) / 96.0 for m in matches])) if matches else 0.0
    inliers = 0
    if len(matches) >= 4:
        pts_a = np.float32([kpa[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kpb[m.trainIdx].pt for m in matches])
        try:
            _, mask = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=12.0)
            inliers = int(mask.sum()) if mask is not None else 0
        except Exception:
            inliers = 0
    local_score = 0.58 * min(inliers / 16.0, 1.0) + 0.27 * min(len(matches) / 42.0, 1.0) + 0.15 * mean_conf
    return {"local_score": float(local_score), "matches": int(len(matches)), "inliers": int(inliers), "mean_conf": mean_conf}


def load_feature_cache(path: Path, n_rows: int) -> np.ndarray | None:
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == n_rows:
            print(f"[cache] loaded {path} {arr.shape}")
            return normalize_rows(arr)
        print(f"[cache] ignoring wrong row count {path}: {arr.shape[0]} != {n_rows}")
    return None


def load_local_cache(path: Path) -> dict[int, LocalFeatures] | None:
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


def save_local_cache(path: Path, features: dict[int, LocalFeatures]) -> None:
    raw = {int(k): {"keypoints": v.keypoints.astype(np.float32), "descriptors": v.descriptors.astype(np.float32)} for k, v in features.items()}
    with path.open("wb") as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] wrote local features {path}")


def extract_global_views(args: argparse.Namespace, rows: pd.DataFrame, reports_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for view in TURTLE_VIEWS:
        path = reports_dir / f"{VERSION}_global_{view}.npy"
        arr = load_feature_cache(path, len(rows))
        if arr is None:
            extractor = SimpleGlobalExtractor(args.img_size, view) if args.disable_eva else EVAGlobalExtractor(args.eva_model, args.eva_pretrained, args.eva_batch_size, args.img_size, view)
            arr = extractor.extract(rows)
            np.save(path, arr)
            print(f"[features] wrote {path} {arr.shape}")
        out[view] = arr
    return out


def extract_local_views(args: argparse.Namespace, rows: pd.DataFrame, reports_dir: Path) -> dict[str, dict[int, LocalFeatures]]:
    out = {}
    if args.disable_aliked:
        return out
    for view in TURTLE_VIEWS:
        path = reports_dir / f"{VERSION}_aliked_{view}.pkl"
        cached = load_local_cache(path) if args.cache_aliked else None
        if cached is not None:
            out[view] = cached
            continue
        extractor = ALIKEDExtractor(args.img_size, args.max_aliked_keypoints, view)
        feats = extractor.extract(rows)
        if args.cache_aliked:
            save_local_cache(path, feats)
        out[view] = feats
    return out


def sample_train_pairs(train: pd.DataFrame, features: np.ndarray, limit: int) -> list[tuple[int, int, int]]:
    ids = train["image_id"].astype(int).to_numpy()
    labels = train["identity"].astype(str).to_numpy()
    by_label = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    sim = features @ features.T
    rng = np.random.default_rng(SEED)
    pairs: set[tuple[int, int, int]] = set()
    for idxs in by_label.values():
        if len(idxs) < 2:
            continue
        combos = list(itertools.combinations(idxs, 2))
        if len(combos) > 50:
            hard = sorted(combos, key=lambda ab: sim[ab[0], ab[1]])[:20]
            rand = [combos[i] for i in rng.choice(len(combos), size=30, replace=False)]
            combos = hard + rand
        for a, b in combos:
            x, y = sorted((int(ids[a]), int(ids[b])))
            pairs.add((x, y, 1))
    order = np.argsort(-sim, axis=1)
    for i in range(len(ids)):
        added = 0
        for j in order[i, 1:80]:
            if labels[i] != labels[j]:
                x, y = sorted((int(ids[i]), int(ids[j])))
                pairs.add((x, y, 0))
                added += 1
                if added >= 10:
                    break
    while len([p for p in pairs if p[2] == 0]) < min(limit // 2, 5000):
        a, b = rng.integers(0, len(ids), size=2)
        if a == b or labels[a] == labels[b]:
            continue
        x, y = sorted((int(ids[a]), int(ids[b])))
        pairs.add((x, y, 0))
    positives = [p for p in pairs if p[2] == 1]
    negatives = [p for p in pairs if p[2] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected = positives[: limit // 2] + negatives[: limit - min(len(positives), limit // 2)]
    rng.shuffle(selected)
    return selected[:limit]


def test_candidate_pairs(test: pd.DataFrame, features: np.ndarray, base: pd.DataFrame, top_k: int, limit: int) -> list[tuple[int, int, str]]:
    ids = test["image_id"].astype(int).to_numpy()
    cluster_by_id = base.set_index("image_id")["cluster"].astype(str).to_dict()
    sim = features @ features.T
    order = np.argsort(-sim, axis=1)
    pairs: set[tuple[int, int, str]] = set()
    for i in range(len(ids)):
        added = 0
        for j in order[i, 1 : top_k * 5]:
            if cluster_by_id[int(ids[i])] == cluster_by_id[int(ids[j])]:
                continue
            x, y = sorted((int(ids[i]), int(ids[j])))
            pairs.add((x, y, "topk_cross"))
            added += 1
            if added >= top_k:
                break
    id_to_idx = {int(image_id): i for i, image_id in enumerate(ids)}
    return sorted(pairs, key=lambda p: -sim[id_to_idx[p[0]], id_to_idx[p[1]]])[:limit]


def score_pairs(
    pairs: list[tuple[int, int, int | str]],
    rows: pd.DataFrame,
    feature_rows: pd.DataFrame,
    global_features: dict[str, np.ndarray],
    local_features: dict[str, dict[int, LocalFeatures]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows_by_id = rows.set_index("image_id", drop=False)
    feat_idx = {int(v): i for i, v in enumerate(feature_rows["image_id"].astype(int).tolist())}
    records = []
    for a, b, tag in tqdm(pairs, desc="pair scores"):
        a = int(a)
        b = int(b)
        ia = feat_idx[a]
        ib = feat_idx[b]
        eva_scores = {
            "orig_orig": float(np.dot(global_features["orig"][ia], global_features["orig"][ib])),
            "hflip_hflip": float(np.dot(global_features["hflip"][ia], global_features["hflip"][ib])),
            "orig_hflip": float(np.dot(global_features["orig"][ia], global_features["hflip"][ib])),
            "hflip_orig": float(np.dot(global_features["hflip"][ia], global_features["orig"][ib])),
        }
        eva_best_view = max(eva_scores, key=lambda view: eva_scores[view])
        eva_best = eva_scores[eva_best_view]

        local_records = {}
        for view_a, view_b in [("orig", "orig"), ("hflip", "hflip"), ("orig", "hflip"), ("hflip", "orig")]:
            key = f"{view_a}_{view_b}"
            if view_a in local_features and view_b in local_features:
                local_records[key] = match_local(local_features[view_a][a], local_features[view_b][b], args.aliked_match_threshold)
            elif args.allow_orb_fallback:
                local_records[key] = orb_pair_score(rows_by_id.loc[a], rows_by_id.loc[b], args.img_size, view_b)
            else:
                local_records[key] = {"local_score": 0.0, "matches": 0, "inliers": 0, "mean_conf": 0.0}
        local_best_view = max(local_records, key=lambda view: local_records[view]["local_score"])
        local_best = local_records[local_best_view]
        rec = {
            "image_id_a": a,
            "image_id_b": b,
            "pair_tag": tag,
            "eva_orig": eva_scores["orig_orig"],
            "eva_hflip": eva_scores["hflip_hflip"],
            "eva_cross": max(eva_scores["orig_hflip"], eva_scores["hflip_orig"]),
            "eva_best": eva_best,
            "eva_best01": float((eva_best + 1.0) * 0.5),
            "eva_best_view": eva_best_view,
            "local_orig": local_records["orig_orig"]["local_score"],
            "local_hflip": local_records["hflip_hflip"]["local_score"],
            "local_cross": max(local_records["orig_hflip"]["local_score"], local_records["hflip_orig"]["local_score"]),
            "local_best": local_best["local_score"],
            "local_best_view": local_best_view,
            "matches": int(local_best["matches"]),
            "inliers": int(local_best["inliers"]),
            "mean_conf": float(local_best["mean_conf"]),
            "orientation_a": str(rows_by_id.loc[a].get("orientation", "")),
            "orientation_b": str(rows_by_id.loc[b].get("orientation", "")),
        }
        if isinstance(tag, (int, np.integer)):
            rec["label"] = int(tag)
        records.append(rec)
    return pd.DataFrame(records)


def classifier_features(df: pd.DataFrame) -> np.ndarray:
    arr = np.column_stack(
        [
            df["eva_best01"].astype(float).to_numpy(),
            ((df["eva_orig"].astype(float).to_numpy() + 1.0) * 0.5),
            ((df["eva_hflip"].astype(float).to_numpy() + 1.0) * 0.5),
            ((df["eva_cross"].astype(float).to_numpy() + 1.0) * 0.5),
            df["local_best"].astype(float).to_numpy(),
            df["local_orig"].astype(float).to_numpy(),
            df["local_hflip"].astype(float).to_numpy(),
            df["local_cross"].astype(float).to_numpy(),
            np.minimum(df["matches"].astype(float).to_numpy() / 60.0, 1.0),
            np.minimum(df["inliers"].astype(float).to_numpy() / 24.0, 1.0),
            df["mean_conf"].astype(float).to_numpy(),
            df["eva_best_view"].astype(str).str.contains("hflip", regex=False).astype(float).to_numpy(),
            df["local_best_view"].astype(str).str.contains("hflip", regex=False).astype(float).to_numpy(),
            df["eva_best_view"].astype(str).str.contains("orig_hflip|hflip_orig", regex=True).astype(float).to_numpy(),
            df["local_best_view"].astype(str).str.contains("orig_hflip|hflip_orig", regex=True).astype(float).to_numpy(),
            df["orientation_a"].astype(str).eq(df["orientation_b"].astype(str)).astype(float).to_numpy(),
        ]
    )
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def train_calibrator(train_pairs: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    x = classifier_features(train_pairs)
    y = train_pairs["label"].astype(int).to_numpy()
    x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)
    try:
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=SEED,
        )
        clf.fit(x_tr, y_tr)
    except Exception as exc:
        print(f"[calibration] HGB failed; using logistic regression: {exc}")
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


def select_rule(train_pairs: pd.DataFrame, target_precision: float, min_support: int) -> dict:
    candidates = []
    for prob_thr in [0.90, 0.94, 0.97, 0.98, 0.99]:
        for local_thr in [0.40, 0.55, 0.70, 0.82, 0.90, 0.95]:
            for inlier_thr in [4, 6, 8, 10, 14, 18, 24, 32]:
                for eva_thr in [0.86, 0.90, 0.94, 0.96, 0.98]:
                    mask = (
                        (train_pairs["match_prob"].astype(float) >= prob_thr)
                        & (train_pairs["local_best"].astype(float) >= local_thr)
                        & (train_pairs["inliers"].astype(float) >= inlier_thr)
                        & (train_pairs["eva_best01"].astype(float) >= eva_thr)
                    )
                    sub = train_pairs[mask]
                    if len(sub) < min_support:
                        continue
                    precision = float(sub["label"].mean())
                    positives = int(sub["label"].sum())
                    negatives = int((sub["label"] == 0).sum())
                    candidates.append(
                        {
                            "prob_thr": prob_thr,
                            "local_thr": local_thr,
                            "inlier_thr": inlier_thr,
                            "eva_thr": eva_thr,
                            "support": int(len(sub)),
                            "positives": positives,
                            "negatives": negatives,
                            "precision": precision,
                        }
                    )
    if not candidates:
        return {"prob_thr": 0.999, "local_thr": 0.999, "inlier_thr": 999, "eva_thr": 0.999, "support": 0, "positives": 0, "negatives": 0, "precision": 0.0}
    df = pd.DataFrame(candidates)
    good = df[df["precision"] >= target_precision].copy()
    if good.empty:
        good = df.sort_values(["precision", "positives", "support"], ascending=[False, False, False]).head(30)
    else:
        good = good.sort_values(["positives", "precision", "support"], ascending=[False, False, False])
    return good.iloc[0].to_dict()


def build_variant(
    name: str,
    base: pd.DataFrame,
    sample: pd.DataFrame,
    test_rows: pd.DataFrame,
    salamander_test: pd.DataFrame,
    test_pairs: pd.DataFrame,
    rule: dict,
    max_size: int,
    max_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    salamander_ids = salamander_test["image_id"].astype(int).tolist()
    current = base[base["image_id"].astype(int).isin(salamander_ids)].copy()
    uf = UF(salamander_ids)
    for _, group in current.groupby("cluster"):
        ids = group["image_id"].astype(int).tolist()
        for a, b in itertools.combinations(ids, 2):
            uf.union(a, b)

    cluster_by_id = current.set_index("image_id")["cluster"].astype(str).to_dict()
    candidates = test_pairs[
        (test_pairs["match_prob"].astype(float) >= float(rule["prob_thr"]))
        & (test_pairs["local_best"].astype(float) >= float(rule["local_thr"]))
        & (test_pairs["inliers"].astype(float) >= int(rule["inlier_thr"]))
        & (test_pairs["eva_best01"].astype(float) >= float(rule["eva_thr"]))
    ].copy()
    candidates = candidates.sort_values(["match_prob", "local_best", "inliers"], ascending=False)

    actions = []
    for row in candidates.itertuples(index=False):
        if len(actions) >= max_edges:
            break
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        if a not in uf.parent or b not in uf.parent or uf.find(a) == uf.find(b):
            continue
        size_after = uf.size[uf.find(a)] + uf.size[uf.find(b)]
        if size_after > max_size:
            continue
        uf.union(a, b)
        actions.append(
            {
                "profile": name,
                "image_id_a": a,
                "image_id_b": b,
                "cluster_a": cluster_by_id[a],
                "cluster_b": cluster_by_id[b],
                "match_prob": float(row.match_prob),
                "eva_best01": float(row.eva_best01),
                "eva_best_view": row.eva_best_view,
                "local_best": float(row.local_best),
                "local_best_view": row.local_best_view,
                "matches": int(row.matches),
                "inliers": int(row.inliers),
                "size_after": int(size_after),
            }
        )

    groups = {}
    for image_id in salamander_ids:
        groups.setdefault(uf.find(image_id), []).append(image_id)
    labels = {}
    for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
        for image_id in members:
            labels[image_id] = f"cluster_{SALAMANDER}_{idx}"

    sub = base.copy()
    mask = sub["image_id"].astype(int).isin(labels)
    sub.loc[mask, "cluster"] = sub.loc[mask, "image_id"].astype(int).map(labels)
    final = compact_species_labels(sub, sample, test_rows)
    shape = shape_report(final, test_rows)
    return final, pd.DataFrame(actions), shape


def shape_report(sub: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    tmp = sub.merge(test_rows[["image_id", "dataset"]], on="image_id", how="left")
    rows = []
    for species, group in tmp.groupby("dataset", sort=True):
        sizes = group.groupby("cluster").size()
        rows.append({"species": species, "n_images": int(len(group)), "n_clusters": int(sizes.size), "max_cluster_size": int(sizes.max()), "singletons": int((sizes == 1).sum())})
    return pd.DataFrame(rows)


def save_visualizations(out_dir: Path, salamander_test: pd.DataFrame, actions: pd.DataFrame, img_size: int, max_pairs: int = 16) -> None:
    if actions.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = salamander_test.set_index("image_id", drop=False)
    pairs = actions.head(max_pairs)
    for idx, row in enumerate(pairs.itertuples(index=False)):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        panels = []
        for image_id in [a, b]:
            r = rows_by_id.loc[image_id]
            original = ImageOps.contain(Image.open(r.abs_path).convert("RGB"), (img_size, img_size))
            hflip = load_salamander_view(r, img_size, "hflip")
            panels.extend([original, hflip])
        canvas = Image.new("RGB", (img_size * 4, img_size + 44), (238, 238, 238))
        for j, img in enumerate(panels):
            bg = Image.new("RGB", (img_size, img_size), tuple(int(x) for x in PAD))
            bg.paste(img, ((img_size - img.width) // 2, (img_size - img.height) // 2))
            canvas.paste(bg, (j * img_size, 0))
        draw = ImageDraw.Draw(canvas)
        text = f"{a} vs {b} prob={row.match_prob:.3f} local={row.local_best:.3f} inliers={row.inliers} view={row.local_best_view}"
        draw.rectangle((0, img_size, img_size * 4, img_size + 44), fill=(245, 245, 245))
        draw.text((10, img_size + 14), text, fill=(20, 20, 20))
        canvas.save(out_dir / f"seaturtle_merge_{idx:02d}_{a}_{b}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    seed_everything()
    if args.smoke:
        args.disable_eva = True
        args.disable_aliked = True
        args.train_pair_limit = min(args.train_pair_limit, 500)
        args.test_pair_limit = min(args.test_pair_limit, 800)
        args.top_k = min(args.top_k, 8)

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    submissions_dir = output_root / "submissions"
    viz_dir = output_root / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_data_root(args.data_root)
    metadata = prepare_metadata(data_root)
    test_rows = metadata[metadata["split"].eq("test")].copy()
    sample = pd.read_csv(data_root / "sample_submission.csv")[["image_id"]].copy()
    sample["image_id"] = sample["image_id"].astype(int)

    current_best_path = find_optional_file(CURRENT_BEST_NAMES, args.current_best, rank_current_best)
    if (
        current_best_path is not None
        and args.current_best is None
        and current_best_path.name == "submission.csv"
        and rank_current_best(current_best_path)[0] >= 900
    ):
        current_best_path = None
    if current_best_path is not None:
        current_best = sample.merge(load_submission(current_best_path), on="image_id", how="left")
        validate_submission(current_best, sample)
        base_source = str(current_best_path)
    else:
        base_path = find_file(BASE_032684_NAMES, args.base_032684, rank_base_032684)
        lynx_pair_path = find_file([LYNX_PAIR_SCORE_NAME], args.lynx_pair_scores, rank_pair_path)
        base_032684 = sample.merge(load_submission(base_path), on="image_id", how="left")
        validate_submission(base_032684, sample)
        current_best = build_032903_from_components(base_032684, pd.read_csv(lynx_pair_path), sample, test_rows)
        validate_submission(current_best, sample)
        base_source = f"rebuilt_032903_from base={base_path} lynx_pairs={lynx_pair_path}"

    salamander_train = metadata[(metadata["dataset"].eq(SALAMANDER)) & (metadata["split"].eq("train"))].copy()
    salamander_test = metadata[(metadata["dataset"].eq(SALAMANDER)) & (metadata["split"].eq("test"))].copy()
    if args.smoke:
        salamander_train = salamander_train.head(260).copy()
        salamander_test = salamander_test.head(120).copy()
    all_salamander = pd.concat([salamander_train, salamander_test], ignore_index=True)

    print(f"[paths] data_root={data_root}")
    print(f"[paths] current_best_source={base_source}")
    print(f"[data] seaturtle_train={len(salamander_train)} seaturtle_test={len(salamander_test)}")

    t0 = time.time()
    global_features = extract_global_views(args, all_salamander, reports_dir)
    local_features = extract_local_views(args, all_salamander, reports_dir)

    combined_for_sampling = normalize_rows(global_features["orig"] + global_features["hflip"])
    train_features = combined_for_sampling[: len(salamander_train)]
    test_features = combined_for_sampling[len(salamander_train) :]

    train_pair_idx = sample_train_pairs(salamander_train, train_features, args.train_pair_limit)
    train_pairs = score_pairs(train_pair_idx, all_salamander, all_salamander, global_features, local_features, args)
    train_pairs.to_csv(reports_dir / f"{VERSION}_train_pairs.csv", index=False)
    clf, cal_stats = train_calibrator(train_pairs)
    train_pairs["match_prob"] = clf.predict_proba(classifier_features(train_pairs))[:, 1]
    train_pairs.to_csv(reports_dir / f"{VERSION}_train_pairs_scored.csv", index=False)
    print(f"[calibration] {json.dumps(cal_stats, indent=2)}")

    test_pair_idx = test_candidate_pairs(salamander_test, test_features, current_best, args.top_k, args.test_pair_limit)
    test_pairs = score_pairs(test_pair_idx, all_salamander, all_salamander, global_features, local_features, args)
    cluster_map = current_best.set_index("image_id")["cluster"].astype(str).to_dict()
    test_pairs["cluster_a"] = test_pairs["image_id_a"].astype(int).map(cluster_map)
    test_pairs["cluster_b"] = test_pairs["image_id_b"].astype(int).map(cluster_map)
    test_pairs["same_current_cluster"] = test_pairs["cluster_a"].eq(test_pairs["cluster_b"])
    test_pairs["match_prob"] = clf.predict_proba(classifier_features(test_pairs))[:, 1]
    test_pairs = test_pairs.sort_values(["match_prob", "local_best", "inliers"], ascending=False).reset_index(drop=True)
    test_pairs.to_csv(reports_dir / f"{VERSION}_test_pair_scores.csv", index=False)

    rules = {
        "ultra_p50": select_rule(train_pairs, target_precision=0.997, min_support=16),
        "strict_p80": select_rule(train_pairs, target_precision=0.992, min_support=28),
        "p90_gamble": select_rule(train_pairs, target_precision=0.985, min_support=40),
    }
    for key, rule in rules.items():
        rule["profile"] = key
    pd.DataFrame(rules.values()).to_csv(reports_dir / f"{VERSION}_selected_rules.csv", index=False)

    profiles = [
        ("ultra_p50", 16, 10),
        ("strict_p80", 30, 22),
        ("p90_gamble", 42, 38),
    ]
    written = {}
    action_frames = []
    shape_frames = []
    for profile, max_size, max_edges in profiles:
        sub, actions, shape = build_variant(profile, current_best, sample, test_rows, salamander_test, test_pairs, rules[profile], max_size=max_size, max_edges=max_edges)
        validate_submission(sub, sample)
        out_name = f"submission_{VERSION}_{profile}.csv"
        sub.to_csv(submissions_dir / out_name, index=False)
        if profile == "ultra_p50":
            sub.to_csv(output_root / "submission.csv", index=False)
        actions.to_csv(reports_dir / f"{VERSION}_{profile}_actions.csv", index=False)
        shape["profile"] = profile
        shape.to_csv(reports_dir / f"{VERSION}_{profile}_shape_report.csv", index=False)
        actions["profile"] = profile
        action_frames.append(actions)
        shape_frames.append(shape)
        written[profile] = str(submissions_dir / out_name)
        print(f"\n[{profile}] rule={rules[profile]}")
        print("actions:", len(actions))
        print(shape.to_string(index=False))
        if args.save_visualizations and profile == "ultra_p50":
            save_visualizations(viz_dir, salamander_test, actions, args.img_size)

    all_actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    all_shapes = pd.concat(shape_frames, ignore_index=True) if shape_frames else pd.DataFrame()
    all_actions.to_csv(reports_dir / f"{VERSION}_all_actions.csv", index=False)
    all_shapes.to_csv(reports_dir / f"{VERSION}_all_shape_reports.csv", index=False)
    base_shape = shape_report(current_best, test_rows)
    base_shape.to_csv(reports_dir / f"{VERSION}_base_shape_report.csv", index=False)

    summary = {
        "version": VERSION,
        "base_public_lb": "0.34411",
        "data_root": str(data_root),
        "current_best_source": base_source,
        "calibration": cal_stats,
        "selected_rules": rules,
        "written": written,
        "recommended_first": str(output_root / "submission.csv"),
        "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
        "notes": [
            "SeaTurtle-only merge candidates; no splits.",
            "Two views are scored for every image: original and horizontal flip because test orientation is unavailable.",
            "Default submission.csv is ultra_p50. Inspect selected_rules/actions/shape reports before upload.",
        ],
    }
    (reports_dir / f"{VERSION}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[done]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
