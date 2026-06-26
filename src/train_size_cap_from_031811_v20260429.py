
import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VERSION = "train_size_cap_from_031811_v20260429"
SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]
BASE_FILENAME = "submission_031811_precisionedge_texas_nooval_v20260428.csv"


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

    def union_if_cap(self, a: int, b: int, cap: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] + self.size[rb] > cap:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def find_data_root(user_value: str | None = None) -> Path:
    candidates = []
    if user_value:
        candidates.append(Path(user_value))
    candidates.extend(
        [
            Path.cwd() / "animal-clef-2026",
            Path.cwd().parent / "animal-clef-2026",
            Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\animal-clef-2026"),
            Path("/kaggle/input/animal-clef-2026"),
            Path("/kaggle/input/competitions/animal-clef-2026"),
        ]
    )
    for root in candidates:
        if (root / "metadata.csv").exists() and (root / "sample_submission.csv").exists():
            return root.resolve()
    for root in [Path.cwd(), Path.cwd().parent, Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026"), Path("/kaggle/input")]:
        if not root.exists():
            continue
        try:
            for meta in root.rglob("metadata.csv"):
                if (meta.parent / "sample_submission.csv").exists():
                    return meta.parent.resolve()
        except Exception:
            continue
    raise FileNotFoundError("Could not locate animal-clef-2026 metadata.csv and sample_submission.csv")


def find_file(filename: str, user_value: str | None = None) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\archive"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
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
        if "local_smoke" in text or "/smoke" in text:
            penalty += 50
        if "kaggle_outputs" in text:
            penalty -= 10
        if "precision_edge_rescue_reproducer_sv315006700" in text:
            penalty -= 5
        if "winner_reuse_wildfusion_gamble_v20260427" in text:
            penalty -= 3
        # For duplicate report names, the real Kaggle pair-score CSV is much
        # larger than smoke output. Prefer larger files after path quality.
        return penalty, -int(path.stat().st_size), text

    matches = sorted({p.resolve() for p in matches}, key=preference)
    if not matches:
        raise FileNotFoundError(f"Could not find {filename}")
    return matches[0]


def optional_pair_scores(species: str) -> Path | None:
    names = {
        "LynxID2025": "winner_reuse_wildfusion_gamble_v20260427_LynxID2025_test_pair_scores.csv",
        "SalamanderID2025": "winner_reuse_wildfusion_gamble_v20260427_SalamanderID2025_test_pair_scores.csv",
        "SeaTurtleID2022": "winner_reuse_wildfusion_gamble_v20260427_SeaTurtleID2022_test_pair_scores.csv",
        "TexasHornedLizards": "texas_astrodot_nooval_v20260427_Texas_pair_scores.csv",
    }
    try:
        return find_file(names[species])
    except FileNotFoundError:
        return None


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns")
    out = df[["image_id", "cluster"]].copy()
    out["image_id"] = out["image_id"].astype(int)
    out["cluster"] = out["cluster"].astype(str)
    return out


def prepare_metadata(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(data_root / "metadata.csv")
    if "species_id" not in metadata.columns:
        metadata["species_id"] = metadata["dataset"].astype(str)
    if "split" not in metadata.columns:
        metadata["split"] = np.where(
            metadata["path"].astype(str).str.contains("/test/|\\\\test\\\\", regex=True),
            "test",
            "train",
        )
    metadata["image_id"] = metadata["image_id"].astype(int)
    sample = pd.read_csv(data_root / "sample_submission.csv")[["image_id"]].copy()
    sample["image_id"] = sample["image_id"].astype(int)
    test = metadata[metadata["split"].eq("test")][["image_id", "species_id"]].copy()
    return metadata, test, sample


def train_caps(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = metadata[metadata["split"].eq("train") & metadata["identity"].notna()].copy()
    for species in SPECIES_ORDER:
        g = train[train["species_id"].eq(species)]
        if g.empty:
            rows.append(
                {
                    "species": species,
                    "train_images": 0,
                    "train_identities": 0,
                    "cap_max": None,
                    "cap_p99": None,
                    "cap_p95": None,
                    "cap_p90": None,
                    "top_train_sizes": "",
                    "note": "no train identities",
                }
            )
            continue
        counts = g["identity"].astype(str).value_counts()
        rows.append(
            {
                "species": species,
                "train_images": int(len(g)),
                "train_identities": int(len(counts)),
                "cap_max": int(counts.max()),
                "cap_p99": int(math.ceil(float(counts.quantile(0.99)))),
                "cap_p95": int(math.ceil(float(counts.quantile(0.95)))),
                "cap_p90": int(math.ceil(float(counts.quantile(0.90)))),
                "top_train_sizes": " ".join(map(str, counts.head(12).astype(int).tolist())),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def summarize(submission: pd.DataFrame, test: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = test.merge(submission, on="image_id", how="left")
    rows = []
    for species in SPECIES_ORDER:
        g = merged[merged["species_id"].eq(species)]
        counts = g["cluster"].value_counts()
        rows.append(
            {
                "submission": name,
                "species": species,
                "n_images": int(len(g)),
                "n_clusters": int(len(counts)),
                "singletons": int((counts == 1).sum()) if len(counts) else 0,
                "max_cluster": int(counts.max()) if len(counts) else 0,
                "top_cluster_sizes": " ".join(map(str, counts.head(12).astype(int).tolist())),
            }
        )
    return pd.DataFrame(rows)


def compact_labels(submission: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    species_map = dict(zip(test["image_id"].astype(int), test["species_id"].astype(str)))
    label_map = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    out_map: dict[int, str] = {}
    for species in SPECIES_ORDER:
        ids = sorted([i for i, sp in species_map.items() if sp == species])
        groups: dict[str, list[int]] = {}
        for image_id in ids:
            groups.setdefault(label_map[image_id], []).append(image_id)
        for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
            label = f"cluster_{species}_{idx}"
            for image_id in members:
                out_map[image_id] = label
    out = submission.copy()
    out["cluster"] = out["image_id"].astype(int).map(out_map)
    return out


def pair_edge_table(species: str) -> pd.DataFrame:
    path = optional_pair_scores(species)
    if path is None:
        return pd.DataFrame(columns=["image_id_a", "image_id_b", "weight"])
    df = pd.read_csv(path)
    weight_col = None
    for col in ["fused_score", "score", "local_score", "descriptor_cosine"]:
        if col in df.columns:
            weight_col = col
            break
    if weight_col is None or "image_id_a" not in df.columns or "image_id_b" not in df.columns:
        return pd.DataFrame(columns=["image_id_a", "image_id_b", "weight"])
    out = df[["image_id_a", "image_id_b", weight_col]].copy()
    out.columns = ["image_id_a", "image_id_b", "weight"]
    out["image_id_a"] = out["image_id_a"].astype(int)
    out["image_id_b"] = out["image_id_b"].astype(int)
    out["weight"] = out["weight"].astype(float)
    return out


def split_members_by_edges(members: list[int], cap: int, edge_table: pd.DataFrame) -> list[list[int]]:
    members = sorted(map(int, members))
    if len(members) <= cap:
        return [members]
    member_set = set(members)
    edges = edge_table[
        edge_table["image_id_a"].isin(member_set) & edge_table["image_id_b"].isin(member_set)
    ].sort_values("weight", ascending=False)
    uf = UnionFind(members)
    for row in edges.itertuples(index=False):
        uf.union_if_cap(int(row.image_id_a), int(row.image_id_b), cap)
    groups: dict[int, list[int]] = {}
    for image_id in members:
        groups.setdefault(uf.find(image_id), []).append(image_id)
    # If no useful edges were available, or if isolated groups can still be
    # packed without violating cap, perform a deterministic final packing pass.
    packed: list[list[int]] = []
    for group in sorted(groups.values(), key=lambda xs: (-len(xs), min(xs))):
        if len(group) > cap:
            for i in range(0, len(group), cap):
                packed.append(group[i : i + cap])
            continue
        placed = False
        for bucket in packed:
            if len(bucket) + len(group) <= cap:
                bucket.extend(group)
                bucket.sort()
                placed = True
                break
        if not placed:
            packed.append(list(group))
    return sorted([sorted(g) for g in packed], key=lambda xs: (min(xs), len(xs), max(xs)))


def apply_cap(submission: pd.DataFrame, test: pd.DataFrame, caps: dict[str, int | None], strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_map = dict(zip(test["image_id"].astype(int), test["species_id"].astype(str)))
    labels = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    new_labels = labels.copy()
    report_rows = []
    for species in SPECIES_ORDER:
        cap = caps.get(species)
        if cap is None or cap <= 0:
            continue
        edge_table = pair_edge_table(species)
        ids = [i for i, sp in species_map.items() if sp == species]
        groups: dict[str, list[int]] = {}
        for image_id in ids:
            groups.setdefault(labels[image_id], []).append(image_id)
        next_suffix = 0
        for old_label, members in sorted(groups.items(), key=lambda kv: (min(kv[1]), len(kv[1]), kv[0])):
            if len(members) <= cap:
                continue
            parts = split_members_by_edges(members, cap, edge_table)
            for part in parts:
                new_label = f"{old_label}__{strategy}_{next_suffix}"
                next_suffix += 1
                for image_id in part:
                    new_labels[image_id] = new_label
            report_rows.append(
                {
                    "species": species,
                    "strategy": strategy,
                    "cap": int(cap),
                    "old_cluster": old_label,
                    "old_size": int(len(members)),
                    "new_sizes": " ".join(map(str, [len(p) for p in parts])),
                    "n_parts": int(len(parts)),
                    "edge_rows_available": int(len(edge_table)),
                }
            )
    out = submission.copy()
    out["cluster"] = out["image_id"].astype(int).map(new_labels)
    out = compact_labels(out, test)
    return out, pd.DataFrame(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-submission", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    project = Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423")
    out_root = Path(args.output_root) if args.output_root else project
    sub_dir = out_root / "submissions"
    report_dir = out_root / "reports" / VERSION
    sub_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    data_root = find_data_root(args.data_root)
    metadata, test, sample = prepare_metadata(data_root)
    base_path = find_file(BASE_FILENAME, args.base_submission)
    base = load_submission(base_path)
    if base["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Base submission image_id order does not match sample_submission.csv")

    cap_df = train_caps(metadata)
    cap_df.to_csv(report_dir / "train_identity_caps.csv", index=False)

    outputs = {}
    reports = [summarize(base, test, "base_031811")]
    split_reports = []
    for strategy, cap_col in [("trainmax", "cap_max"), ("p95", "cap_p95"), ("p90", "cap_p90")]:
        caps = {
            str(row.species): (None if pd.isna(getattr(row, cap_col)) else int(getattr(row, cap_col)))
            for row in cap_df.itertuples(index=False)
        }
        capped, split_report = apply_cap(base, test, caps, strategy)
        name = f"submission_031811_{strategy}_cluster_cap_v20260429.csv"
        path = sub_dir / name
        capped.to_csv(path, index=False)
        outputs[strategy] = str(path)
        reports.append(summarize(capped, test, strategy))
        if not split_report.empty:
            split_reports.append(split_report)
    summary = pd.concat(reports, ignore_index=True)
    summary.to_csv(report_dir / "candidate_shape_report.csv", index=False)
    if split_reports:
        pd.concat(split_reports, ignore_index=True).to_csv(report_dir / "split_cluster_report.csv", index=False)
    else:
        pd.DataFrame().to_csv(report_dir / "split_cluster_report.csv", index=False)

    print("base:", base_path)
    print("caps:")
    print(cap_df.to_string(index=False))
    print("outputs:")
    for key, value in outputs.items():
        print(key, value)
    print("shape report:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
