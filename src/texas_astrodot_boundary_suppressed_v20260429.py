
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import texas_astrodot_2025reuse_v20260426 as base


VERSION = "texas_astrodot_boundary_suppressed_v20260429"
TEXAS = base.TEXAS

CURRENT_BEST_FILENAMES = [
    "submission_032583_salamander_p80_cluster_cap_v20260429.csv",
    "submission_032368_gamble_salamander_p80_cap_v20260429.csv",
    "submission.csv",
]


def find_current_best_boundary(user_value: str | None, data_root: Path) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.cwd(),
        Path.cwd().parent,
        data_root.parent,
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423\submissions"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423"),
    ]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for filename in CURRENT_BEST_FILENAMES:
            direct = root / filename
            if direct.exists():
                matches.append(direct)
            try:
                matches.extend(root.rglob(filename))
            except Exception:
                continue

    def preference(path: Path) -> tuple[int, int, str]:
        text = str(path).replace("\\", "/").lower()
        penalty = 0
        if "032583" in text or "032368_gamble" in text:
            penalty -= 40
        if "lb032583" in text or "salamander_p80" in text:
            penalty -= 20
        if text.endswith("/submission.csv"):
            penalty += 5
        if "local_smoke" in text:
            penalty += 50
        return penalty, -int(path.stat().st_size), text

    matches = sorted({p.resolve() for p in matches}, key=preference)
    if matches:
        return matches[0]
    return base.find_current_best(user_value, data_root)


def align_vertical_no_flip(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align the ventral lizard body but do not flip: field heads are already up."""
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
    return crop_rgb, crop_mask, {
        "pca_angle": float(angle),
        "rotate_angle": float(rotate_angle),
        "top_width": float(max(widths[:2])),
        "bottom_width": float(max(widths[2:])),
        "flipped_vertical": False,
        "orientation_rule": "head_already_top_no_flip",
    }


def interior_weight_from_mask(mask: np.ndarray) -> np.ndarray:
    """Softly suppress the body outline while preserving full body geometry.

    The old oval mask was too rigid; the no-oval mask kept useful peripheral
    dots but also allowed the animal contour to dominate. This distance-weighted
    formula keeps the real mask shape and scale but makes the contour band weak:

        W = 0.08 + 0.92 * clip((D - r0) / (r1 - r0), 0, 1) ** gamma

    where D is distance to the foreground boundary. Dots near the center keep
    full strength; the hard silhouette edge is downweighted, not cropped away.
    """
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    h, w = m.shape[:2]
    if int((m > 0).sum()) < 40:
        return np.ones((h, w), dtype=np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    short = float(max(1, min(h, w)))
    r0 = max(2.0, short * 0.020)
    r1 = max(r0 + 2.0, short * 0.115)
    core = np.clip((dist - r0) / max(1e-6, r1 - r0), 0.0, 1.0)
    weight = 0.08 + 0.92 * np.power(core, 0.72)
    weight[m == 0] = 0.0
    return weight.astype(np.float32)


def boundary_suppressed_dot_heat(
    belly_rgb: np.ndarray,
    belly_mask: np.ndarray,
    interior_weight: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(belly_rgb, cv2.COLOR_RGB2LAB)
    l_eq = base.clahe_u8(lab[:, :, 0])
    dark = (255.0 - l_eq.astype(np.float32)) / 255.0
    blackhat = cv2.morphologyEx(l_eq, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    raw = 0.58 * (blackhat.astype(np.float32) / 255.0) + 0.42 * dark
    valid = belly_mask > 0
    if int(valid.sum()) > 20:
        vals = raw[valid]
        lo = float(np.percentile(vals, 12))
        hi = float(np.percentile(vals, 98))
        raw = (raw - lo) / max(1e-6, hi - lo)
    raw = np.clip(raw, 0.0, 1.0)

    edge = cv2.morphologyEx(
        np.where(valid, 255, 0).astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(np.float32) / 255.0
    edge = cv2.GaussianBlur(edge, (9, 9), 0)

    heat = raw * interior_weight
    heat = np.clip(heat - 0.20 * edge, 0.0, 1.0)
    heat[~valid] = 0.0
    heat = cv2.GaussianBlur(heat, (3, 3), 0)
    return heat.astype(np.float32)


def weighted_dot_points_from_heat(heat: np.ndarray, mask: np.ndarray, interior_weight: np.ndarray) -> np.ndarray:
    valid = (mask > 0) & (interior_weight >= 0.16)
    if int(valid.sum()) < 40:
        valid = mask > 0
    vals = heat[valid]
    if len(vals) < 40:
        return np.zeros((0, 4), dtype=np.float32)
    thr = max(float(np.percentile(vals, 84)), float(vals.mean() + 0.55 * vals.std()))
    binary = np.zeros(heat.shape, dtype=np.uint8)
    binary[(heat >= thr) & valid] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    h, w = heat.shape[:2]
    total = float(h * w)
    pts: list[list[float]] = []
    min_area = max(2.0, total * 0.00004)
    max_area = total * 0.010
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bw = max(1.0, float(stats[idx, cv2.CC_STAT_WIDTH]))
        bh = max(1.0, float(stats[idx, cv2.CC_STAT_HEIGHT]))
        if max(bw / bh, bh / bw) > 4.8:
            continue
        cx, cy = centroids[idx]
        comp = labels == idx
        strength = float((heat[comp] * np.maximum(0.2, interior_weight[comp])).mean())
        pts.append([float(cx) / max(1, w - 1), float(cy) / max(1, h - 1), area / total, strength])
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    pts_arr = np.asarray(pts, dtype=np.float32)
    order = np.argsort(-pts_arr[:, 3])
    return pts_arr[order[:260]]


def texas_belly_template_boundary(row: dict, current_cluster: str, args: argparse.Namespace) -> base.TexasDotItem:
    """Full aligned crop, no oval, but contour-suppressed belly-dot signal."""
    rgb, mask, quality = base.read_rgb_mask(row, args.max_side)
    aligned_rgb, aligned_mask, debug = align_vertical_no_flip(rgb, mask)
    w = int(args.texas_canvas_w)
    h = int(args.texas_canvas_h)
    belly_rgb = cv2.resize(aligned_rgb, (w, h), interpolation=cv2.INTER_AREA)
    belly_mask_full = cv2.resize(aligned_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    belly_mask_full = base.clean_mask(belly_mask_full)

    coverage = float(belly_mask_full.mean() / 255.0)
    if coverage < 0.035:
        belly_mask_full = np.full((h, w), 255, dtype=np.uint8)
        quality *= 0.68
        coverage = 1.0
    elif coverage > 0.92:
        quality *= 0.94

    interior_weight = interior_weight_from_mask(belly_mask_full)
    effective_mask = np.where((belly_mask_full > 0) & (interior_weight >= 0.12), 255, 0).astype(np.uint8)
    if float(effective_mask.mean() / 255.0) < 0.025:
        effective_mask = belly_mask_full.copy()
        interior_weight = np.where(effective_mask > 0, 1.0, 0.0).astype(np.float32)
        quality *= 0.74

    heat = boundary_suppressed_dot_heat(belly_rgb, belly_mask_full, interior_weight)
    points = weighted_dot_points_from_heat(heat, effective_mask, interior_weight)
    small = cv2.resize(heat, (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    weight_small = cv2.resize(interior_weight, (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    mask_small = cv2.resize((effective_mask > 0).astype(np.float32), (32, 48), interpolation=cv2.INTER_AREA).reshape(-1)
    vector = base.normalize(np.concatenate([small * weight_small, mask_small * 0.10, weight_small * 0.08]).astype(np.float32))
    quality *= min(1.0, 0.70 + 0.30 * min(1.0, len(points) / 42.0))

    debug = dict(debug)
    debug["mask_mode"] = "full_aligned_crop_boundary_suppressed"
    debug["full_mask_coverage"] = coverage
    debug["effective_mask_coverage"] = float(effective_mask.mean() / 255.0)
    debug["mean_interior_weight"] = float(interior_weight[belly_mask_full > 0].mean()) if (belly_mask_full > 0).any() else 0.0

    return base.TexasDotItem(
        image_id=int(row["image_id"]),
        current_cluster=str(current_cluster),
        source_path=str(row.get("source_path", "")),
        view_path=str(row.get("view_path", row.get("source_path", ""))),
        view_source=str(row.get("view_source", "")),
        belly_rgb=belly_rgb,
        belly_mask=effective_mask,
        dot_heat=heat,
        dot_points=points,
        vector=vector,
        quality=float(quality),
        debug=debug,
    )


def texas_pair_score_boundary(a: base.TexasDotItem, b: base.TexasDotItem) -> dict:
    """Dot-centered score: point alignment and stack sharpness matter more than contour correlation."""
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
    common_mask = np.where(a.belly_mask > 0, 1.0, 0.0).astype(np.float32)
    base_dx, base_dy = base.phase_shift(a.dot_heat, b.dot_heat, common_mask)
    candidates = {(0, 0), (base_dx, base_dy)}
    for dx0, dy0 in [(base_dx, base_dy), (0, 0)]:
        for ddx in (-8, 0, 8):
            for ddy in (-8, 0, 8):
                candidates.add((int(np.clip(dx0 + ddx, -24, 24)), int(np.clip(dy0 + ddy, -24, 24))))
    for dx, dy in candidates:
        corr, overlap, stack_gain = base.masked_corr_and_stack(a.dot_heat, a.belly_mask, b.dot_heat, b.belly_mask, dx, dy)
        if overlap < 0.040:
            continue
        point_score = base.chamfer_dot_score(a.dot_points, b.dot_points, dx / max(1, w - 1), dy / max(1, h - 1))
        fused = 0.25 * corr + 0.42 * point_score + 0.25 * stack_gain + 0.08 * max(0.0, desc_cos)
        if min(len(a.dot_points), len(b.dot_points)) < 12:
            fused *= 0.82
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


def texas_variant_labels_boundary(items: list[base.TexasDotItem], pair_scores: pd.DataFrame, variant: str) -> dict[int, str]:
    ids = [it.image_id for it in items]
    by_cluster: dict[str, list[int]] = {}
    current_by_id = {it.image_id: it.current_cluster for it in items}
    for it in items:
        by_cluster.setdefault(it.current_cluster, []).append(it.image_id)

    if variant == "split_strict":
        keep_thr, merge_thr, merge_rank = 0.330, 9.9, 0
        max_cross_component = 999
    elif variant == "merge_ultra":
        keep_thr, merge_thr, merge_rank = 0.0, 0.575, 1
        max_cross_component = 32
    elif variant == "splitmerge_guarded":
        keep_thr, merge_thr, merge_rank = 0.305, 0.555, 1
        max_cross_component = 32
    else:
        keep_thr, merge_thr, merge_rank = 0.285, 0.540, 1
        max_cross_component = 32

    uf = base.UnionFind(ids)
    if variant == "merge_ultra":
        for members in by_cluster.values():
            anchor = members[0]
            for other in members[1:]:
                uf.union(anchor, other)
    else:
        for members in by_cluster.values():
            if len(members) <= 1:
                continue
            g = pair_scores[
                pair_scores["image_id_a"].isin(members)
                & pair_scores["image_id_b"].isin(members)
                & pair_scores["same_current_cluster"].astype(bool)
            ]
            for row in g[g["score"].astype(float) >= keep_thr].itertuples(index=False):
                # Boundary-safe evidence should include either actual point
                # alignment or a stacking gain; pure contour correlation is not enough.
                if float(row.point_score) < 0.18 and float(row.stack_gain) < 0.42:
                    continue
                uf.union(int(row.image_id_a), int(row.image_id_b))

    if merge_rank > 0:
        cross = pair_scores[~pair_scores["same_current_cluster"].astype(bool)].copy()
        if not cross.empty:
            cross = cross[
                (cross["score"].astype(float) >= merge_thr)
                & (cross["overlap"].astype(float) >= 0.105)
                & (cross["point_score"].astype(float) >= 0.42)
                & (cross["stack_gain"].astype(float) >= 0.46)
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
                        ra = uf.find(a)
                        rb = uf.find(b)
                        if ra != rb and uf.size[ra] + uf.size[rb] <= max_cross_component:
                            uf.union(ra, rb)

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
        new_label = f"cluster_TexasHornedLizards_boundary_{variant}_{comp_order[root]}"
        for image_id in members:
            labels[image_id] = new_label
    return labels


def main() -> None:
    base.VERSION = VERSION
    base.find_current_best = find_current_best_boundary
    base.align_vertical = align_vertical_no_flip
    base.texas_belly_template = texas_belly_template_boundary
    base.texas_pair_score = texas_pair_score_boundary
    base.texas_variant_labels = texas_variant_labels_boundary
    base.main()


if __name__ == "__main__":
    main()
