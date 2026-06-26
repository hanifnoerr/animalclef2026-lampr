import hashlib
import csv
from pathlib import Path

import pandas as pd


def repo_root():
    p = Path.cwd()
    for candidate in [p, *p.parents]:
        if (candidate / 'paper_submissions_manifest.csv').exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def write_splice(output_path=None):
    root = repo_root()
    species = pd.read_csv(root / 'input/source_components/test_species.csv')
    shape = pd.read_csv(root / 'input/paper_submissions/shape_constrained_fusion_partition.csv')
    sample = shape[['image_id']].copy()
    p06 = pd.read_csv(root / 'input/source_components/submission_p06_miewid_plus_mega_l384.csv')
    sources = {
        'LynxID2025': shape,
        'TexasHornedLizards': shape,
        'SalamanderID2025': p06,
        'SeaTurtleID2022': p06,
    }
    parts = []
    for dataset, frame in sources.items():
        ids = set(species.loc[species.dataset.eq(dataset), 'image_id'].astype(int))
        part = frame[frame.image_id.astype(int).isin(ids)][['image_id', 'cluster']]
        parts.append(part)
    out = sample.merge(pd.concat(parts, ignore_index=True), on='image_id', how='left')
    if out.cluster.isna().any():
        raise ValueError('missing cluster labels')
    output = Path(output_path) if output_path is not None else root / 'paper_submissions' / 'species_hybrid.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, quoting=csv.QUOTE_ALL, lineterminator='\r\n')
    return output, hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    path, digest = write_splice()
    print(path)
    print(digest)
