
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VERSION = "precision_edge_rescue_reproducer_v20260428"

SPECIES_ORDER = [
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
]

BASE_FILENAME = "submission_texas_astrodot_2025reuse_v20260426_splitmerge_swing.csv"
TEXAS_NOOVAL_FILENAME = "submission_texas_astrodot_nooval_v20260427_splitmerge_swing.csv"
TARGET_FILENAME = "submission_precision_edge_rescue_v5_midprecision_st_turtle_v20260427.csv"
FINAL_031811_FILENAME = "submission_031811_precisionedge_texas_nooval_v20260428.csv"
PAIR_PREFIX = "winner_reuse_wildfusion_gamble_v20260427"

PRECISION_RULES = {
    "SalamanderID2025": {
        "fused_min": 0.7070,
        "inliers_min": 8,
        "local_min": 0.625,
        "part_min": 0.90,
        "max_size": 9,
    },
    "SeaTurtleID2022": {
        "fused_min": 0.6780,
        "inliers_min": 22,
        "local_min": 0.740,
        "part_min": 0.86,
        "max_size": 13,
    },
}


class UnionFind:
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


def candidate_roots() -> list[Path]:
    roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\archive\generated_runs_v20260427"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423"),
    ]
    return [p for p in roots if p.exists()]


def path_preference(path: Path) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    penalty = 0
    if "local_smoke" in text or "/smoke" in text:
        penalty += 20
    if "__pycache__" in text:
        penalty += 20
    if "/kaggle/input/" in text or "kaggle_output" in text:
        penalty -= 5
    if "winner_reuse_texas_astro_stack" in text:
        penalty -= 2
    if "winner_reuse_wildfusion_gamble" in text:
        penalty -= 2
    if "texas_astro_stack_no_oval" in text or "texas-astro-stack-no-oval" in text:
        penalty -= 3
    return penalty, len(text), text


def find_data_root(user_value: str | None = None) -> Path:
    candidates = []
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
    for root in candidate_roots():
        try:
            for meta in root.rglob("metadata.csv"):
                if (meta.parent / "sample_submission.csv").exists():
                    return meta.parent.resolve()
        except Exception:
            continue
    raise FileNotFoundError("Could not locate AnimalCLEF2026 metadata.csv and sample_submission.csv.")


def find_file(filename: str, user_value: str | None = None, roots: Iterable[Path] | None = None) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    search_roots = list(roots or candidate_roots())
    matches: list[Path] = []
    for root in search_roots:
        direct = root / filename
        if direct.exists():
            matches.append(direct)
        try:
            matches.extend(root.rglob(filename))
        except Exception:
            continue
    matches = [p.resolve() for p in matches if p.exists()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Add the previous notebook output dataset containing it as a Kaggle input."
        )
    matches = sorted(set(matches), key=path_preference)
    return matches[0]


def find_optional_file(filename: str, user_value: str | None = None, roots: Iterable[Path] | None = None) -> Path | None:
    try:
        return find_file(filename, user_value=user_value, roots=roots)
    except FileNotFoundError:
        return None


def pair_score_path(species: str, user_root: str | None = None) -> Path:
    filename = f"{PAIR_PREFIX}_{species}_test_pair_scores.csv"
    roots = []
    if user_root:
        roots.append(Path(user_root))
    roots.extend(candidate_roots())
    return find_file(filename, roots=roots)


def prepare_metadata(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(data_root / "metadata.csv").copy()
    if "species_id" not in metadata.columns:
        if "dataset" in metadata.columns:
            metadata["species_id"] = metadata["dataset"].astype(str)
        else:
            metadata["species_id"] = metadata["path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[1]
    if "split" not in metadata.columns:
        metadata["split"] = np.where(
            metadata["path"].astype(str).str.contains("/test/|\\\\test\\\\", regex=True),
            "test",
            "train",
        )
    metadata["image_id"] = metadata["image_id"].astype(int)
    test_rows = metadata[metadata["split"].eq("test")].copy()
    sample = pd.read_csv(data_root / "sample_submission.csv")
    sample["image_id"] = sample["image_id"].astype(int)
    return metadata, test_rows, sample


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns.")
    df = df[["image_id", "cluster"]].copy()
    df["image_id"] = df["image_id"].astype(int)
    df["cluster"] = df["cluster"].astype(str)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"{path} contains duplicate image_id values, e.g. {dupes}")
    return df


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != ["image_id", "cluster"]:
        raise ValueError("Submission must contain exactly image_id, cluster columns.")
    if len(submission) != len(sample):
        raise ValueError(f"Submission row count {len(submission)} != sample row count {len(sample)}.")
    if submission["cluster"].isna().any():
        raise ValueError("Submission has missing cluster labels.")
    if submission["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Submission image_id order does not match sample_submission.csv.")


def relabel_compact(ids: list[int], uf: UnionFind, species: str) -> dict[int, str]:
    groups: dict[int, list[int]] = {}
    for image_id in ids:
        groups.setdefault(uf.find(image_id), []).append(image_id)
    out: dict[int, str] = {}
    for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
        label = f"cluster_{species}_{idx}"
        for image_id in members:
            out[int(image_id)] = label
    return out


def compact_labels_by_species(submission: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    species_map = dict(zip(test_rows["image_id"].astype(int), test_rows["species_id"].astype(str)))
    label_map = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    compact: dict[int, str] = {}
    for species in SPECIES_ORDER:
        ids = sorted([int(i) for i, sp in species_map.items() if sp == species])
        groups: dict[str, list[int]] = {}
        for image_id in ids:
            groups.setdefault(label_map[image_id], []).append(image_id)
        for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
            label = f"cluster_{species}_{idx}"
            for image_id in members:
                compact[image_id] = label
    out = submission.copy()
    out["cluster"] = out["image_id"].astype(int).map(compact)
    return out[["image_id", "cluster"]]


def apply_precision_edges(
    base: pd.DataFrame,
    test_rows: pd.DataFrame,
    species: str,
    pair_scores: pd.DataFrame,
) -> tuple[dict[int, str], list[dict], int]:
    ids = test_rows.loc[test_rows["species_id"].eq(species), "image_id"].astype(int).tolist()
    base_map = dict(zip(base["image_id"].astype(int), base["cluster"].astype(str)))

    uf = UnionFind(ids)
    cluster_to_members: dict[str, list[int]] = {}
    for image_id in ids:
        cluster_to_members.setdefault(base_map[image_id], []).append(image_id)
    for members in cluster_to_members.values():
        anchor = members[0]
        for other in members[1:]:
            uf.union(anchor, other)

    rule = PRECISION_RULES[species]
    df = pair_scores.copy()
    df = df[~df["same_current_cluster"].astype(bool)].copy()
    for col in ["fused_score", "local_score", "part_score", "inliers"]:
        if col not in df.columns:
            df[col] = 0
    df = df[
        (df["fused_score"].astype(float) >= float(rule["fused_min"]))
        & (df["local_score"].astype(float) >= float(rule["local_min"]))
        & (df["part_score"].astype(float) >= float(rule["part_min"]))
        & (df["inliers"].astype(int) >= int(rule["inliers_min"]))
    ].copy()
    df = df.sort_values(["fused_score", "local_score", "inliers"], ascending=[False, False, False])

    accepted: list[dict] = []
    for row in df.itertuples(index=False):
        a = int(row.image_id_a)
        b = int(row.image_id_b)
        if a not in uf.parent or b not in uf.parent:
            continue
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            continue
        size_after = int(uf.size[ra] + uf.size[rb])
        if size_after > int(rule["max_size"]):
            continue
        uf.union(a, b)
        accepted.append(
            {
                "variant": "v5_midprecision_st_turtle",
                "species": species,
                "image_id_a": a,
                "image_id_b": b,
                "fused_score": float(row.fused_score),
                "inliers": int(row.inliers),
                "inlier_ratio": float(getattr(row, "inlier_ratio", np.nan)),
                "mega_sim": float(getattr(row, "mega_sim", np.nan)),
                "local_score": float(row.local_score),
                "part_score": float(row.part_score),
                "alt_votes": int(getattr(row, "alt_votes", 0)),
                "orientation_rule": str(getattr(row, "orientation_rule", "")),
                "size_after": size_after,
            }
        )
    return relabel_compact(ids, uf, species), accepted, int(len(df))


def summarize(submission: pd.DataFrame, test_rows: pd.DataFrame, label: str) -> list[dict]:
    merged = test_rows[["image_id", "species_id"]].merge(submission, on="image_id", how="left")
    rows = []
    for species in SPECIES_ORDER:
        g = merged[merged["species_id"].eq(species)]
        counts = g["cluster"].value_counts()
        rows.append(
            {
                "submission": label,
                "species": species,
                "n_images": int(len(g)),
                "n_clusters": int(len(counts)),
                "singletons": int((counts == 1).sum()) if not counts.empty else 0,
                "max_cluster": int(counts.max()) if not counts.empty else 0,
                "top_cluster_sizes": " ".join(map(str, counts.head(12).astype(int).tolist())),
            }
        )
    return rows


def partition_set(submission: pd.DataFrame) -> set[tuple[int, ...]]:
    return set(
        submission.groupby("cluster")["image_id"]
        .apply(lambda s: tuple(sorted(map(int, s.tolist()))))
        .tolist()
    )


def maybe_validate_reference(final: pd.DataFrame, reference_path: str | None) -> dict:
    if not reference_path:
        return {"reference_found": False, "note": "No reference submission was provided; skipped optional validation."}
    ref = Path(reference_path)
    reference = load_submission(ref)
    return {
        "reference_found": True,
        "reference_path": str(ref),
        "byte_equal": bool(final.equals(reference)),
        "partition_equal": bool(partition_set(final) == partition_set(reference)),
        "n_clusters_final": int(final["cluster"].nunique()),
        "n_clusters_reference": int(reference["cluster"].nunique()),
    }


def splice_nooval_texas(
    precision: pd.DataFrame,
    texas_nooval: pd.DataFrame,
    test_rows: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    precision_map = dict(zip(precision["image_id"].astype(int), precision["cluster"].astype(str)))
    texas_map = dict(zip(texas_nooval["image_id"].astype(int), texas_nooval["cluster"].astype(str)))
    species_map = dict(zip(test_rows["image_id"].astype(int), test_rows["species_id"].astype(str)))
    missing_texas = [
        image_id
        for image_id, species in species_map.items()
        if species == "TexasHornedLizards" and image_id not in texas_map
    ]
    if missing_texas:
        raise ValueError(f"Texas no-oval submission is missing Texas image ids, e.g. {missing_texas[:8]}")
    out = sample[["image_id"]].copy()
    out["image_id"] = out["image_id"].astype(int)
    out["cluster"] = out["image_id"].map(
        lambda image_id: texas_map[int(image_id)]
        if species_map[int(image_id)] == "TexasHornedLizards"
        else precision_map[int(image_id)]
    )
    return compact_labels_by_species(out, test_rows)


def main(
    data_root: str | None = None,
    base_submission: str | None = None,
    texas_nooval_submission: str | None = None,
    pair_report_root: str | None = None,
    output_root: str | None = None,
    reference_submission: str | None = None,
    final_reference_submission: str | None = None,
) -> dict:
    data_path = find_data_root(data_root)
    metadata, test_rows, sample = prepare_metadata(data_path)
    base_path = find_file(BASE_FILENAME, base_submission)
    base = load_submission(base_path)
    validate_submission(base, sample)

    out_root = Path(output_root or Path.cwd() / VERSION).resolve()
    submissions_dir = out_root / "submissions"
    reports_dir = out_root / "reports"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    final = base.copy()
    base_map = dict(zip(base["image_id"].astype(int), base["cluster"].astype(str)))
    updates: dict[int, str] = {}
    all_edges: list[dict] = []
    rule_report: list[dict] = []
    pair_paths: dict[str, str] = {}

    for species in ["SalamanderID2025", "SeaTurtleID2022"]:
        path = pair_score_path(species, pair_report_root)
        pair_paths[species] = str(path)
        pair_scores = pd.read_csv(path)
        labels, accepted, candidates = apply_precision_edges(base, test_rows, species, pair_scores)
        updates.update(labels)
        all_edges.extend(accepted)
        rule_report.append(
            {
                "species": species,
                "pair_score_path": str(path),
                "candidate_edges_above_threshold": candidates,
                "accepted_edges_after_size_guard": len(accepted),
                **PRECISION_RULES[species],
            }
        )

    final["cluster"] = final["image_id"].astype(int).map(lambda image_id: updates.get(int(image_id), base_map[int(image_id)]))
    validate_submission(final, sample)

    final_path = submissions_dir / TARGET_FILENAME
    final.to_csv(final_path, index=False)
    final.to_csv(out_root / TARGET_FILENAME, index=False)
    if Path("/kaggle/working").exists():
        final.to_csv(Path("/kaggle/working") / TARGET_FILENAME, index=False)
    else:
        final.to_csv(Path.cwd() / TARGET_FILENAME, index=False)

    edge_report = pd.DataFrame(all_edges)
    edge_report.to_csv(reports_dir / "precision_edge_rescue_accepted_edges.csv", index=False)
    rule_df = pd.DataFrame(rule_report)
    rule_df.to_csv(reports_dir / "precision_edge_rescue_rule_report.csv", index=False)

    candidate_report = pd.DataFrame(summarize(base, test_rows, "base_029889_texas_astrodot_2025reuse_swing") + summarize(final, test_rows, TARGET_FILENAME))

    reference_check = maybe_validate_reference(final, reference_submission)
    texas_nooval_path = find_optional_file(TEXAS_NOOVAL_FILENAME, texas_nooval_submission)
    final_031811_check: dict = {"written": False, "reason": f"{TEXAS_NOOVAL_FILENAME} not found"}
    final_031811_path: Path | None = None
    if texas_nooval_path is not None:
        texas_nooval = load_submission(texas_nooval_path)
        final_031811 = splice_nooval_texas(final, texas_nooval, test_rows, sample)
        validate_submission(final_031811, sample)
        final_031811_path = submissions_dir / FINAL_031811_FILENAME
        final_031811.to_csv(final_031811_path, index=False)
        final_031811.to_csv(out_root / FINAL_031811_FILENAME, index=False)
        # If the no-oval Texas branch is available, this is the actual 0.31811 final.
        final_031811.to_csv(out_root / "submission.csv", index=False)
        if Path("/kaggle/working").exists():
            final_031811.to_csv(Path("/kaggle/working") / FINAL_031811_FILENAME, index=False)
            final_031811.to_csv(Path("/kaggle/working") / "submission.csv", index=False)
        else:
            final_031811.to_csv(Path.cwd() / FINAL_031811_FILENAME, index=False)
        final_031811_check = {
            "written": True,
            "texas_nooval_source": str(texas_nooval_path),
            "output_submission": str(final_031811_path),
            "top_level_submission": str(out_root / "submission.csv"),
            "reference_check": maybe_validate_reference(final_031811, final_reference_submission),
        }
        candidate_report = pd.concat(
            [
                candidate_report,
                pd.DataFrame(summarize(final_031811, test_rows, FINAL_031811_FILENAME)),
            ],
            ignore_index=True,
        )
    else:
        # Fallback behavior for runs that only attach the older Texas branch.
        final.to_csv(out_root / "submission.csv", index=False)
        if Path("/kaggle/working").exists():
            final.to_csv(Path("/kaggle/working") / "submission.csv", index=False)
    candidate_report.to_csv(reports_dir / "precision_edge_rescue_candidate_report.csv", index=False)

    summary = {
        "version": VERSION,
        "data_root": str(data_path),
        "base_submission": str(base_path),
        "texas_nooval_submission": str(texas_nooval_path) if texas_nooval_path is not None else None,
        "pair_score_paths": pair_paths,
        "output_submission": str(final_path),
        "final_031811": final_031811_check,
        "top_level_submission": str(out_root / "submission.csv"),
        "rules": PRECISION_RULES,
        "rule_report": rule_report,
        "reference_check": reference_check,
        "note": (
            "Rebuilds the 0.31763 precision-edge rescue candidate from the 0.29889 "
            "Texas astro-stack base plus WildFusion gamble pair-score reports. If the "
            "Texas no-oval output is attached, it also writes the direct 0.31811 final by "
            "replacing only TexasHornedLizards with the no-oval/no-flip splitmerge_swing branch. "
            "It does not load target submissions unless optional reference copies are provided "
            "for validation."
        ),
    }
    (reports_dir / "precision_edge_rescue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"data_root={data_path}")
    print(f"base_submission={base_path}")
    for species, path in pair_paths.items():
        print(f"pair_scores[{species}]={path}")
    print(f"wrote={final_path}")
    if final_031811_path is not None:
        print(f"wrote_final_031811={final_031811_path}")
        print(f"top_level_submission={out_root / 'submission.csv'}")
    print(candidate_report.to_string(index=False))
    print("reference_check=", json.dumps(reference_check, indent=2))
    print("final_031811=", json.dumps(final_031811_check, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the AnimalCLEF2026 precision-edge rescue v5 candidate.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-submission", type=str, default=None)
    parser.add_argument("--texas-nooval-submission", type=str, default=None)
    parser.add_argument("--pair-report-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--reference-submission", type=str, default=None)
    parser.add_argument("--final-reference-submission", type=str, default=None)
    ns = parser.parse_args()
    main(
        data_root=ns.data_root,
        base_submission=ns.base_submission,
        texas_nooval_submission=ns.texas_nooval_submission,
        pair_report_root=ns.pair_report_root,
        output_root=ns.output_root,
        reference_submission=ns.reference_submission,
        final_reference_submission=ns.final_reference_submission,
    )
