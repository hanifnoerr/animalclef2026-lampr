import csv
import hashlib
import shutil
from pathlib import Path


def locate_repo():
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / "paper_submissions_manifest.csv").exists():
            return candidate
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists():
            for manifest in base.glob("*/paper_submissions_manifest.csv"):
                return manifest.parent
    raise FileNotFoundError("paper_submissions_manifest.csv")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_rows(repo):
    with open(repo / "paper_submissions_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reproduce_submission(slug, output_name="submission.csv"):
    repo = locate_repo()
    rows = manifest_rows(repo)
    matches = [row for row in rows if row["slug"] == slug]
    if not matches:
        raise KeyError(slug)
    row = matches[0]
    source = repo / row["input_csv"]
    output = repo / output_name
    shutil.copyfile(source, output)
    digest = sha256_file(output)
    if digest != row["sha256"]:
        raise RuntimeError(f"{slug}: {digest} != {row['sha256']}")
    return output, digest, row
