#!/usr/bin/env python3
"""
AnimalCLEF2026 LB 0.32903 Lynx ALIKED merge-only reproducer.

This notebook/script reproduces the submitted 0.32903 public-LB artifact
without loading the submitted CSV. It starts from the verified 0.32684
final-texas-boundary-split-only partition, reads the Lynx EVA02+ALIKED
test pair scores from the heavy Lynx notebook output, and applies the exact
ultra-strict merge-only rule that improved the leaderboard.

Rule:
    cross-current-cluster Lynx pair
    not opposite flank
    local_score >= 0.96
    inliers >= 50
    match_prob >= 0.98
    merged component size <= 84

Only Lynx labels change. Salamander, SeaTurtle, and Texas remain identical to
the 0.32684 base partition.
"""


import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


VERSION = "lynx_aliked_best_reproducer_v20260430"
LYNX = "LynxID2025"

BASE_CANDIDATES = [
    "submission_final_texas_boundary_splitonly_balanced_from_032583_v20260430.csv",
    "submission.csv",
]
PAIR_SCORE_NAME = "lynx_lowlight_eva_aliked_v20260430_test_pair_scores.csv"
SUBMITTED_NAME = "submission_032684_lynx_aliked_prob098_mergeonly_v20260430.csv"
SCORE_NAME = "submission_lb032903_lynx_aliked_prob098_mergeonly_v20260430.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-submission", type=str, default=None)
    parser.add_argument("--pair-scores", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=f"/kaggle/working/animalclef_{VERSION}")
    parser.add_argument("--reference-submission", type=str, default=None)
    return parser.parse_args()


def candidate_roots() -> list[Path]:
    roots = []
    for value in [
        os.environ.get("DATA_ROOT"),
        "/kaggle/input/competitions/animal-clef-2026",
        "/kaggle/input/animal-clef-2026",
        "/kaggle/input",
        "/kaggle/working",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/animal-clef-2026",
        "C:/Users/Hanif/Documents/kaggle/AnimalCLEF2026/current_wildfusion_graph_v20260423",
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


def rank_base_path(path: Path) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    penalty = 0
    if "local_smoke" in text or "__pycache__" in text:
        penalty += 100
    if "final-texas-boundary-split-only" in text:
        penalty -= 50
    if "animalclef_texas_boundary_splitonly_final" in text:
        penalty -= 40
    if "submission_final_texas_boundary_splitonly" in path.name.lower():
        penalty -= 30
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
    if "lynx-low-light-eva-clip-aliked" in text:
        penalty -= 50
    if "animalclef_lynx_lowlight_eva_aliked" in text:
        penalty -= 40
    if "/reports/" in text:
        penalty -= 10
    if "/kaggle/input/" in text:
        penalty -= 10
    return penalty, len(text), text


def find_file(names: Iterable[str], arg: str | None, ranker) -> Path:
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
    if not hits:
        raise FileNotFoundError(f"Could not find any of: {list(names)}")
    return hits[0]


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
        parts.append(
            pd.DataFrame(
                {
                    "image_id": sorted(ids),
                    "cluster": [labels[image_id] for image_id in sorted(ids)],
                }
            )
        )
    final = pd.concat(parts, ignore_index=True)
    final = sample[["image_id"]].merge(final, on="image_id", how="left")
    return final[["image_id", "cluster"]]


def shape_report(sub: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    tmp = sub.merge(test_rows[["image_id", "dataset"]], on="image_id", how="left")
    rows = []
    for species, group in tmp.groupby("dataset", sort=True):
        sizes = group.groupby("cluster").size()
        rows.append(
            {
                "species": species,
                "n_images": int(len(group)),
                "n_clusters": int(sizes.size),
                "max_cluster_size": int(sizes.max()),
                "singletons": int((sizes == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_best_submission(base: pd.DataFrame, sample: pd.DataFrame, metadata: pd.DataFrame, pair_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = metadata[metadata["split"].eq("test")].copy()
    lynx_ids = (
        test_rows[test_rows["dataset"].eq(LYNX) & test_rows["image_id"].isin(sample["image_id"])]
        ["image_id"]
        .astype(int)
        .tolist()
    )
    current = base[base["image_id"].astype(int).isin(lynx_ids)].copy()

    uf = UF(lynx_ids)
    for _, group in current.groupby("cluster"):
        ids = group["image_id"].astype(int).tolist()
        for a, b in itertools.combinations(ids, 2):
            uf.union(a, b)

    if pair_scores["same_current_cluster"].dtype != bool:
        same_current = pair_scores["same_current_cluster"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        same_current = pair_scores["same_current_cluster"]

    accepted = pair_scores[
        (~same_current)
        & (~pair_scores["orientation_relation"].astype(str).eq("opposite_flank"))
        & (pair_scores["local_score"].astype(float) >= 0.96)
        & (pair_scores["inliers"].astype(float) >= 50)
        & (pair_scores["match_prob"].astype(float) >= 0.98)
    ].copy()
    accepted = accepted.sort_values(["local_score", "inliers", "match_prob"], ascending=False)

    actions = []
    for row in accepted.itertuples(index=False):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        if a not in uf.parent or b not in uf.parent:
            continue
        if uf.find(a) == uf.find(b):
            continue
        size_after = uf.size[uf.find(a)] + uf.size[uf.find(b)]
        if size_after > 84:
            continue
        uf.union(a, b)
        actions.append(
            {
                "image_id_a": a,
                "image_id_b": b,
                "cluster_a": row.cluster_a,
                "cluster_b": row.cluster_b,
                "match_prob": float(row.match_prob),
                "eva_sim01": float(row.eva_sim01),
                "local_score": float(row.local_score),
                "matches": int(row.matches),
                "inliers": int(row.inliers),
                "orientation_relation": row.orientation_relation,
                "size_after": int(size_after),
            }
        )

    groups = {}
    for image_id in lynx_ids:
        groups.setdefault(uf.find(image_id), []).append(image_id)
    labels = {}
    for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
        for image_id in members:
            labels[image_id] = f"cluster_{LYNX}_{idx}"

    sub = base.copy()
    mask = sub["image_id"].astype(int).isin(labels)
    sub.loc[mask, "cluster"] = sub.loc[mask, "image_id"].astype(int).map(labels)
    final = compact_species_labels(sub, sample, test_rows)
    return final, pd.DataFrame(actions)


def main() -> None:
    args = parse_args()
    data_root = find_data_root(args.data_root)
    base_path = find_file(BASE_CANDIDATES, args.base_submission, rank_base_path)
    pair_path = find_file([PAIR_SCORE_NAME], args.pair_scores, rank_pair_path)

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    submissions_dir = output_root / "submissions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data_root / "metadata.csv").copy()
    if "dataset" not in metadata.columns:
        metadata["dataset"] = metadata["path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[1]
    if "split" not in metadata.columns:
        metadata["split"] = metadata["path"].astype(str).str.replace("\\", "/", regex=False).str.contains("/test/").map({True: "test", False: "train"})
    metadata["image_id"] = metadata["image_id"].astype(int)
    test_rows = metadata[metadata["split"].eq("test")].copy()

    sample = pd.read_csv(data_root / "sample_submission.csv")[["image_id"]].copy()
    sample["image_id"] = sample["image_id"].astype(int)
    base = load_submission(base_path)
    base_for_validation = sample.merge(base, on="image_id", how="left")
    validate_submission(base_for_validation, sample)
    pair_scores = pd.read_csv(pair_path)

    final, actions = build_best_submission(base_for_validation, sample, metadata, pair_scores)
    validate_submission(final, sample)

    top_level = output_root / "submission.csv"
    submitted_name_path = submissions_dir / SUBMITTED_NAME
    score_name_path = submissions_dir / SCORE_NAME
    final.to_csv(top_level, index=False)
    final.to_csv(submitted_name_path, index=False)
    final.to_csv(score_name_path, index=False)

    actions_path = reports_dir / f"{VERSION}_actions.csv"
    shape_path = reports_dir / f"{VERSION}_shape_report.csv"
    actions.to_csv(actions_path, index=False)
    shape = shape_report(final, test_rows)
    shape.to_csv(shape_path, index=False)

    base_shape = shape_report(base_for_validation, test_rows)
    base_shape.to_csv(reports_dir / f"{VERSION}_base_shape_report.csv", index=False)

    reference_match = None
    reference_sha = None
    if args.reference_submission:
        ref = Path(args.reference_submission)
        if ref.exists():
            ref_df = pd.read_csv(ref)[["image_id", "cluster"]]
            ref_df["image_id"] = ref_df["image_id"].astype(int)
            ref_df["cluster"] = ref_df["cluster"].astype(str)
            reference_match = bool(final.equals(ref_df))
            reference_sha = file_sha256(ref)

    summary = {
        "version": VERSION,
        "public_lb": "0.32903",
        "base_public_lb": "0.32684",
        "data_root": str(data_root),
        "base_submission": str(base_path),
        "pair_scores": str(pair_path),
        "top_level_submission": str(top_level),
        "submitted_name_copy": str(submitted_name_path),
        "score_name_copy": str(score_name_path),
        "n_actions": int(len(actions)),
        "actions": actions.to_dict("records"),
        "output_sha256": file_sha256(top_level),
        "reference_sha256": reference_sha,
        "matches_reference_submission": reference_match,
        "notes": [
            "This reproduces the 0.32903 artifact by recomputing labels from the 0.32684 base and the Lynx EVA02+ALIKED pair-score report.",
            "It does not load the submitted 0.32903 CSV unless --reference-submission is passed for validation.",
            "Only six Lynx cross-cluster merges are accepted; all non-Lynx labels are inherited from the 0.32684 base.",
        ],
    }
    summary_path = reports_dir / f"{VERSION}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Base shape:")
    print(base_shape.to_string(index=False))
    print("\nFinal shape:")
    print(shape.to_string(index=False))
    print("\nAccepted Lynx merges:")
    print(actions.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
