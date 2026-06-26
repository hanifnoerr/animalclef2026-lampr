import csv
import hashlib
import importlib
import sys
from pathlib import Path


def repo_root():
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "paper_submissions_manifest.csv").exists():
            return candidate
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists():
            for manifest in base.glob("*/paper_submissions_manifest.csv"):
                return manifest.parent
    raise FileNotFoundError("paper_submissions_manifest.csv")


def animalclef_data_root(repo=None):
    repo = Path(repo) if repo is not None else repo_root()
    candidates = [
        repo / "input" / "animal-clef-2026",
        repo.parent / "AnimalCLEF2026" / "01_DATASET_AND_REFERENCES" / "animal-clef-2026",
        Path("/kaggle/input/animal-clef-2026"),
        Path("/kaggle/input/competitions/animal-clef-2026"),
    ]
    for path in candidates:
        if (path / "metadata.csv").exists() and (path / "sample_submission.csv").exists():
            return path
    raise FileNotFoundError("animal-clef-2026 dataset")


def expected_row(slug, repo=None):
    repo = Path(repo) if repo is not None else repo_root()
    with open(repo / "paper_submissions_manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["slug"] == slug:
                return row
    raise KeyError(slug)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_submission(path, slug, repo=None):
    row = expected_row(slug, repo)
    digest = sha256_file(path)
    if digest != row["sha256"]:
        raise RuntimeError(f"{slug}: {digest} != {row['sha256']}")
    print(path)
    print(digest)
    print(row["public_ari"], row["private_ari"])
    return digest


def verify_partition_equivalent(path, reference_path):
    import pandas as pd

    produced = pd.read_csv(path, dtype={"cluster": str})
    reference = pd.read_csv(reference_path, dtype={"cluster": str})
    if produced["image_id"].astype(int).tolist() != reference["image_id"].astype(int).tolist():
        raise RuntimeError("image_id order differs")
    left_to_right = {}
    right_to_left = {}
    for left, right in zip(produced["cluster"].astype(str), reference["cluster"].astype(str)):
        if left in left_to_right and left_to_right[left] != right:
            raise RuntimeError("produced cluster maps to multiple reference clusters")
        if right in right_to_left and right_to_left[right] != left:
            raise RuntimeError("reference cluster maps to multiple produced clusters")
        left_to_right[left] = right
        right_to_left[right] = left
    print("partition-equivalent")
    return True


def run_module_main(module_name, argv):
    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name + ".py", *map(str, argv)]
        module.main()
    finally:
        sys.argv = old_argv
