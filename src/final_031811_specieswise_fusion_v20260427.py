
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VERSION = "final_031811_specieswise_fusion_v20260427"

SPECIES_ORDER = [
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
]

BASE_031763_FILENAME = "submission_precision_edge_rescue_v5_midprecision_st_turtle_v20260427.csv"
TEXAS_NOOVAL_FILENAME = "submission_texas_astrodot_nooval_v20260427_splitmerge_swing.csv"

IMAGE_PREPARATION_POLICY = {
    "LynxID2025": {
        "pattern_source": "original image",
        "mask_use": "SAM/fused mask is a subject guide only; low-light enhancement should affect subject pixels, not the whole frame",
        "background": "preserve original dark context or neutral/dim outside-mask pixels; do not use white SAM cutout as primary",
        "normalization": "generous body/flank ROI, optional subject-only CLAHE, fixed canvas for diagnostics",
        "submission_source": BASE_031763_FILENAME,
    },
    "SalamanderID2025": {
        "pattern_source": "0.31763 precision-edge branch",
        "mask_use": "old branch used original-image foreground/ROI normalization effectively; avoid the failed SAM+YOLO metadata rotation",
        "background": "if experimenting with SAM cutouts, keep white/neutral safer than black because the animal pattern itself is black/yellow",
        "normalization": "the submitted 0.31763 partition is preserved exactly",
        "submission_source": BASE_031763_FILENAME,
    },
    "SeaTurtleID2022": {
        "pattern_source": "0.31763 precision-edge branch",
        "mask_use": "keep current strong branch",
        "background": "no new preprocessing in final reproduction",
        "normalization": "preserve 0.31763 turtle partition exactly",
        "submission_source": BASE_031763_FILENAME,
    },
    "TexasHornedLizards": {
        "pattern_source": "SAM+YOLO/no-oval/no-flip Texas astro-dot branch",
        "mask_use": "body-completeness guidance; no oval crop; head is assumed already at top, so no flip",
        "background": "white/neutral is safer than black when features are gated to belly/body; black padding is diagnostic only",
        "normalization": "use no-oval/no-flip splitmerge_swing Texas partition",
        "submission_source": TEXAS_NOOVAL_FILENAME,
    },
}


def candidate_roots() -> list[Path]:
    roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\current_wildfusion_graph_v20260423"),
        Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\archive\generated_runs_v20260427"),
    ]
    return [p for p in roots if p.exists()]


def find_data_root(user_value: str | None = None) -> Path:
    roots = []
    if user_value:
        roots.append(Path(user_value))
    roots.extend(
        [
            Path("/kaggle/input/animal-clef-2026"),
            Path("/kaggle/input/competitions/animal-clef-2026"),
            Path.cwd() / "animal-clef-2026",
            Path.cwd().parent / "animal-clef-2026",
            Path(r"C:\Users\Hanif\Documents\kaggle\AnimalCLEF2026\animal-clef-2026"),
        ]
    )
    for root in roots:
        if (root / "metadata.csv").exists() and (root / "sample_submission.csv").exists():
            return root.resolve()
    for root in candidate_roots():
        try:
            for meta in root.rglob("metadata.csv"):
                data_root = meta.parent
                if (data_root / "sample_submission.csv").exists():
                    return data_root.resolve()
        except Exception:
            continue
    raise FileNotFoundError("Could not locate AnimalCLEF2026 metadata.csv and sample_submission.csv.")


def find_file(filename: str, user_value: str | None = None, roots: Iterable[Path] | None = None) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p.resolve()
    search_roots = list(roots or candidate_roots())
    for root in search_roots:
        direct = root / filename
        if direct.exists():
            return direct.resolve()
    matches: list[Path] = []
    for root in search_roots:
        try:
            matches.extend(root.rglob(filename))
        except Exception:
            continue
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Add the dataset/notebook output containing this file as a Kaggle input."
        )

    def preference(path: Path) -> tuple[int, int, int, str]:
        text = str(path).replace("\\", "/").lower()
        smoke_penalty = 10 if "local_smoke" in text or "/smoke" in text else 0
        cache_penalty = 3 if "__pycache__" in text else 0
        kaggle_output_bonus = -5 if "kaggle_output" in text or "/kaggle/input/" in text else 0
        return (smoke_penalty + cache_penalty + kaggle_output_bonus, len(text), 0, text)

    matches.sort(key=preference)
    return matches[0].resolve()


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{path} is missing image_id column.")
    cluster_col = "cluster" if "cluster" in df.columns else "identity" if "identity" in df.columns else None
    if cluster_col is None:
        raise ValueError(f"{path} must contain cluster or identity column.")
    df = df[["image_id", cluster_col]].rename(columns={cluster_col: "cluster"}).copy()
    df["image_id"] = df["image_id"].astype(int)
    df["cluster"] = df["cluster"].astype(str)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"{path} contains duplicate image_id values, e.g. {dupes}")
    return df


def prepare_metadata(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(data_root / "metadata.csv").copy()
    if "species_id" not in metadata.columns:
        if "dataset" in metadata.columns:
            metadata["species_id"] = metadata["dataset"].astype(str)
        else:
            metadata["species_id"] = metadata["path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[1]
    if "split" not in metadata.columns:
        metadata["split"] = np.where(metadata["path"].astype(str).str.contains("/test/|\\\\test\\\\", regex=True), "test", "train")
    if "image_id" not in metadata.columns:
        metadata["image_id"] = np.arange(len(metadata), dtype=np.int64)
    test_rows = metadata[metadata["split"].eq("test")].copy()
    sample = pd.read_csv(data_root / "sample_submission.csv")
    sample["image_id"] = sample["image_id"].astype(int)
    test_rows["image_id"] = test_rows["image_id"].astype(int)
    return metadata, test_rows


def compact_labels_by_species(submission: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    species_map = dict(zip(test_rows["image_id"].astype(int), test_rows["species_id"].astype(str)))
    label_map = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    compact: dict[int, str] = {}
    for species in SPECIES_ORDER:
        ids = sorted([int(i) for i, sp in species_map.items() if sp == species])
        groups: dict[str, list[int]] = {}
        for image_id in ids:
            groups.setdefault(label_map[image_id], []).append(image_id)
        ordered_groups = sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))
        for idx, members in enumerate(ordered_groups):
            label = f"cluster_{species}_{idx}"
            for image_id in members:
                compact[image_id] = label
    out = submission.copy()
    out["cluster"] = out["image_id"].astype(int).map(compact)
    return out


def splice_final(base: pd.DataFrame, texas: pd.DataFrame, test_rows: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    base_map = dict(zip(base["image_id"].astype(int), base["cluster"].astype(str)))
    texas_map = dict(zip(texas["image_id"].astype(int), texas["cluster"].astype(str)))
    species_map = dict(zip(test_rows["image_id"].astype(int), test_rows["species_id"].astype(str)))

    missing_base = [i for i in sample["image_id"].astype(int).tolist() if i not in base_map]
    if missing_base:
        raise ValueError(f"Base submission is missing image ids, e.g. {missing_base[:8]}")
    texas_ids = [i for i, sp in species_map.items() if sp == "TexasHornedLizards"]
    missing_texas = [i for i in texas_ids if i not in texas_map]
    if missing_texas:
        raise ValueError(f"Texas source submission is missing Texas image ids, e.g. {missing_texas[:8]}")

    final_map = dict(base_map)
    for image_id in texas_ids:
        final_map[int(image_id)] = texas_map[int(image_id)]

    out = sample[["image_id"]].copy()
    out["image_id"] = out["image_id"].astype(int)
    out["cluster"] = out["image_id"].map(final_map).astype(str)
    out = compact_labels_by_species(out, test_rows)
    return out[["image_id", "cluster"]]


def _percentile_from_sorted(values: list[int], q: float) -> int:
    if not values:
        return 0
    idx = int(np.floor((len(values) - 1) * q))
    return int(values[max(0, min(idx, len(values) - 1))])


def train_identity_priors(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = metadata[metadata["split"].eq("train")].copy()
    for species in SPECIES_ORDER:
        species_train = train[train["species_id"].eq(species)].copy()
        if "identity" not in species_train.columns:
            rows.append(
                {
                    "species": species,
                    "train_images": int(len(species_train)),
                    "train_identities": 0,
                    "train_max_cluster": np.nan,
                    "train_p99_cluster": np.nan,
                    "train_p975_cluster": np.nan,
                    "train_p95_cluster": np.nan,
                    "train_p90_cluster": np.nan,
                    "train_median_cluster": np.nan,
                    "train_mean_cluster": np.nan,
                    "train_top_cluster_sizes": "",
                    "cap_source": "no_identity_column",
                }
            )
            continue
        valid = species_train[species_train["identity"].notna() & species_train["identity"].astype(str).ne("")]
        counts = sorted(valid.groupby("identity").size().astype(int).tolist())
        rows.append(
            {
                "species": species,
                "train_images": int(len(species_train)),
                "train_identities": int(len(counts)),
                "train_max_cluster": int(max(counts)) if counts else np.nan,
                "train_p99_cluster": _percentile_from_sorted(counts, 0.99) if counts else np.nan,
                "train_p975_cluster": _percentile_from_sorted(counts, 0.975) if counts else np.nan,
                "train_p95_cluster": _percentile_from_sorted(counts, 0.95) if counts else np.nan,
                "train_p90_cluster": _percentile_from_sorted(counts, 0.90) if counts else np.nan,
                "train_median_cluster": _percentile_from_sorted(counts, 0.50) if counts else np.nan,
                "train_mean_cluster": float(np.mean(counts)) if counts else np.nan,
                "train_top_cluster_sizes": " ".join(map(str, sorted(counts, reverse=True)[:12])),
                "cap_source": "observed_train_identity_max" if counts else "no_train_identity_labels",
            }
        )
    return pd.DataFrame(rows)


def summarize(
    submission: pd.DataFrame,
    test_rows: pd.DataFrame,
    label: str,
    priors: pd.DataFrame | None = None,
) -> list[dict]:
    prior_map = {}
    if priors is not None and not priors.empty:
        prior_map = priors.set_index("species").to_dict(orient="index")
    sub_map = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    rows = []
    for species in SPECIES_ORDER:
        ids = test_rows.loc[test_rows["species_id"].eq(species), "image_id"].astype(int).tolist()
        labels = [sub_map[i] for i in ids]
        counts = pd.Series(labels).value_counts()
        max_cluster = int(counts.max()) if not counts.empty else 0
        prior = prior_map.get(species, {})
        train_max = prior.get("train_max_cluster", np.nan)
        has_train_cap = pd.notna(train_max)
        rows.append(
            {
                "submission": label,
                "species": species,
                "n_images": int(len(ids)),
                "n_clusters": int(counts.shape[0]),
                "singletons": int((counts == 1).sum()) if not counts.empty else 0,
                "max_cluster": max_cluster,
                "top_cluster_sizes": " ".join(map(str, counts.head(12).astype(int).tolist())),
                "train_max_cluster_cap": int(train_max) if has_train_cap else np.nan,
                "train_p99_cluster": int(prior["train_p99_cluster"]) if pd.notna(prior.get("train_p99_cluster", np.nan)) else np.nan,
                "train_p95_cluster": int(prior["train_p95_cluster"]) if pd.notna(prior.get("train_p95_cluster", np.nan)) else np.nan,
                "max_cluster_vs_train_max": round(float(max_cluster) / float(train_max), 4) if has_train_cap and train_max else np.nan,
                "exceeds_train_max_cap": bool(has_train_cap and max_cluster > int(train_max)),
                "train_cap_source": prior.get("cap_source", "missing_train_prior"),
            }
        )
    return rows


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != ["image_id", "cluster"]:
        raise ValueError("Submission must contain exactly image_id, cluster columns.")
    if len(submission) != len(sample):
        raise ValueError(f"Submission row count {len(submission)} != sample row count {len(sample)}.")
    if submission["cluster"].isna().any():
        raise ValueError("Submission contains missing cluster labels.")
    if submission["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Submission image_id order does not match sample_submission.csv.")
    max_len = int(submission["cluster"].astype(str).str.len().max())
    if max_len > 64:
        raise ValueError(f"Cluster labels are too long: max length {max_len}.")


def validate_against_train_max(report: pd.DataFrame, final_label: str = "final_031811_repro") -> None:
    final_rows = report[report["submission"].eq(final_label)].copy()
    bad = final_rows[final_rows["exceeds_train_max_cap"].fillna(False)]
    if bad.empty:
        return
    details = bad[["species", "max_cluster", "train_max_cluster_cap"]].to_dict(orient="records")
    raise ValueError(f"Final submission exceeds train-derived max cluster cap: {details}")


def main(
    data_root: str | None = None,
    base_031763: str | None = None,
    texas_nooval: str | None = None,
    output_root: str | None = None,
) -> dict:
    data_path = find_data_root(data_root)
    metadata, test_rows = prepare_metadata(data_path)
    sample = pd.read_csv(data_path / "sample_submission.csv")
    sample["image_id"] = sample["image_id"].astype(int)

    base_path = find_file(BASE_031763_FILENAME, base_031763)
    texas_path = find_file(TEXAS_NOOVAL_FILENAME, texas_nooval)
    out_root = Path(output_root or Path.cwd() / VERSION).resolve()
    submissions_dir = out_root / "submissions"
    reports_dir = out_root / "reports"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    base = load_submission(base_path)
    texas = load_submission(texas_path)
    final = splice_final(base, texas, test_rows, sample)
    validate_submission(final, sample)

    final_path = submissions_dir / "submission_final_031811_repro_v20260427.csv"
    final.to_csv(final_path, index=False)
    # Kaggle submit button usually expects /kaggle/working/submission.csv.
    final.to_csv(out_root / "submission.csv", index=False)
    final.to_csv(Path.cwd() / "submission.csv", index=False)

    priors = train_identity_priors(metadata)
    priors.to_csv(reports_dir / "train_identity_cluster_priors.csv", index=False)

    summary_rows = []
    summary_rows.extend(summarize(base, test_rows, "base_031763_precision_edge_v5", priors))
    summary_rows.extend(summarize(texas, test_rows, "texas_nooval_noflip_splitmerge_swing", priors))
    summary_rows.extend(summarize(final, test_rows, "final_031811_repro", priors))
    report = pd.DataFrame(summary_rows)
    validate_against_train_max(report)
    report.to_csv(reports_dir / "final_031811_candidate_report.csv", index=False)

    source_summary = {
        "version": VERSION,
        "data_root": str(data_path),
        "base_031763_source": str(base_path),
        "texas_nooval_source": str(texas_path),
        "output_submission": str(final_path),
        "top_level_submission": str(out_root / "submission.csv"),
        "cwd_submission": str(Path.cwd() / "submission.csv"),
        "label_format": "cluster_<SpeciesID>_<number>",
        "max_label_length": int(final["cluster"].astype(str).str.len().max()),
        "n_rows": int(len(final)),
        "train_identity_priors": priors.to_dict(orient="records"),
        "image_preparation_policy": IMAGE_PREPARATION_POLICY,
        "note": (
            "This reproduces the public-LB 0.31811 strategy: preserve Lynx, Salamander, "
            "and SeaTurtle from the 0.31763 precision-edge branch, then replace only "
            "TexasHornedLizards with the no-oval/no-flip Texas astro-dot branch."
        ),
    }
    (reports_dir / "final_031811_source_summary.json").write_text(json.dumps(source_summary, indent=2), encoding="utf-8")
    (reports_dir / "image_preparation_policy.json").write_text(
        json.dumps(IMAGE_PREPARATION_POLICY, indent=2), encoding="utf-8"
    )

    print(f"data_root={data_path}")
    print(f"base_031763={base_path}")
    print(f"texas_nooval={texas_path}")
    print(f"wrote={final_path}")
    print(f"top_level={out_root / 'submission.csv'}")
    print(f"cwd_submission={Path.cwd() / 'submission.csv'}")
    print(report.to_string(index=False))
    return source_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the 0.31811 AnimalCLEF2026 species-wise final fusion.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--base-031763", type=str, default=None)
    parser.add_argument("--texas-nooval", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    ns = parser.parse_args()
    main(
        data_root=ns.data_root,
        base_031763=ns.base_031763,
        texas_nooval=ns.texas_nooval,
        output_root=ns.output_root,
    )
