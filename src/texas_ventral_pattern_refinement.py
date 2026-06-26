import hashlib
from pathlib import Path

import pandas as pd


def repo_root():
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "paper_submissions_manifest.csv").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def reproduce(output_path=None):
    root = repo_root()
    species = pd.read_csv(root / "input/source_components/test_species.csv")
    base = pd.read_csv(
        root / "input/source_components/submission_precision_edge_rescue_v5_midprecision_st_turtle_v20260427.csv",
        dtype={"cluster": str},
    )
    texas = pd.read_csv(
        root / "input/source_components/submission_texas_astrodot_nooval_v20260427_splitmerge_swing.csv",
        dtype={"cluster": str},
    )
    texas_ids = set(species.loc[species.dataset.eq("TexasHornedLizards"), "image_id"].astype(int))
    out = base.copy()
    labels = texas[texas.image_id.astype(int).isin(texas_ids)].set_index("image_id")["cluster"].astype(str)
    mask = out.image_id.astype(int).isin(texas_ids)
    out.loc[mask, "cluster"] = out.loc[mask, "image_id"].map(labels)
    output = Path(output_path) if output_path is not None else root / "paper_submissions" / "texas_ventral_pattern_refinement.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, lineterminator="\r\n")
    return output, hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    path, digest = reproduce()
    print(path)
    print(digest)
