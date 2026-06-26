
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import train_size_cap_from_031811_v20260429 as base


VERSION = "train_size_cap_lb032583_from_031811_v20260429"
SUBMISSION_NAME = "submission_032583_salamander_p80_cluster_cap_v20260429.csv"


def train_quantile_caps(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train = metadata[metadata["split"].eq("train") & metadata["identity"].notna()].copy()
    for species in base.SPECIES_ORDER:
        g = train[train["species_id"].eq(species)]
        if g.empty:
            rows.append(
                {
                    "species": species,
                    "train_images": 0,
                    "train_identities": 0,
                    "cap_p95": None,
                    "cap_p90": None,
                    "cap_p85": None,
                    "cap_p80": None,
                    "cap_p75": None,
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
                "cap_p95": int(math.ceil(float(counts.quantile(0.95)))),
                "cap_p90": int(math.ceil(float(counts.quantile(0.90)))),
                "cap_p85": int(math.ceil(float(counts.quantile(0.85)))),
                "cap_p80": int(math.ceil(float(counts.quantile(0.80)))),
                "cap_p75": int(math.ceil(float(counts.quantile(0.75)))),
                "top_train_sizes": " ".join(map(str, counts.head(12).astype(int).tolist())),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != ["image_id", "cluster"]:
        raise ValueError("Submission columns must be exactly image_id, cluster")
    if submission["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Submission image_id order does not match sample_submission.csv")
    if submission["cluster"].isna().any():
        raise ValueError("Submission contains missing cluster labels")
    max_len = int(submission["cluster"].astype(str).str.len().max())
    if max_len > 64:
        raise ValueError(f"Submission cluster labels are too long: max length {max_len}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the 0.32583 public-LB submission: confirmed 0.31811 base "
            "plus Salamander-only train p80 cluster-size cap."
        )
    )
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

    data_root = base.find_data_root(args.data_root)
    metadata, test, sample = base.prepare_metadata(data_root)
    base_path = base.find_file(base.BASE_FILENAME, args.base_submission)
    base_submission = base.load_submission(base_path)
    validate_submission(base_submission, sample)

    cap_df = train_quantile_caps(metadata)
    cap_df.to_csv(report_dir / "train_quantile_caps.csv", index=False)

    salamander_row = cap_df[cap_df["species"].eq("SalamanderID2025")]
    if salamander_row.empty or pd.isna(salamander_row.iloc[0]["cap_p80"]):
        raise ValueError("Could not compute Salamander p80 cap from train identities")
    salamander_p80 = int(salamander_row.iloc[0]["cap_p80"])

    # This is intentionally more conservative than applying p80 to all species.
    # The submitted 0.32583 artifact changed only Salamander. Lynx, SeaTurtle,
    # and Texas remain exactly on the proven 0.31811 partition.
    caps = {
        "LynxID2025": None,
        "SalamanderID2025": salamander_p80,
        "SeaTurtleID2022": None,
        "TexasHornedLizards": None,
    }
    capped, split_report = base.apply_cap(base_submission, test, caps, "salamander_p80")
    validate_submission(capped, sample)

    submission_path = sub_dir / SUBMISSION_NAME
    capped.to_csv(submission_path, index=False)
    split_report.to_csv(report_dir / "split_cluster_report.csv", index=False)

    summary = pd.concat(
        [
            base.summarize(base_submission, test, "base_031811"),
            base.summarize(capped, test, "lb032583_salamander_p80"),
        ],
        ignore_index=True,
    )
    summary.to_csv(report_dir / "candidate_shape_report.csv", index=False)

    top_level = out_root / "submission.csv"
    capped.to_csv(top_level, index=False)

    print("base:", base_path)
    print("salamander_p80_cap:", salamander_p80)
    print("submission:", submission_path)
    print("top_level_submission:", top_level)
    print("caps:")
    print(cap_df.to_string(index=False))
    print("shape report:")
    print(summary.to_string(index=False))
    if not split_report.empty:
        print("split report:")
        print(split_report.to_string(index=False))


if __name__ == "__main__":
    main()
