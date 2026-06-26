import hashlib
import re
from pathlib import Path

import pandas as pd


def repo_root():
    p = Path.cwd()
    for candidate in [p, *p.parents]:
        if (candidate / 'paper_submissions_manifest.csv').exists():
            return candidate
    return Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, values):
        self.parent = {str(v): str(v) for v in values}

    def find(self, value):
        value = str(value)
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def label_key(label):
    match = re.search(r'_(\d+)$', str(label))
    return (str(label).rsplit('_', 1)[0], int(match.group(1)) if match else 10**9, str(label))


def reproduce(output_path=None):
    root = repo_root()
    base = pd.read_csv(root / 'input/paper_submissions/salamander_eva02_aliked_local_link_refinement.csv', dtype={'cluster': str})
    uf = UnionFind(base.cluster.unique())
    for name in ['final_gamble_034411_lynx_actions_v20260430.csv', 'final_gamble_034411_salamander_actions_v20260430.csv']:
        actions = pd.read_csv(root / 'input/action_reports' / name)
        for row in actions.itertuples(index=False):
            uf.union(row.cluster_a_current, row.cluster_b_current)
    components = {}
    for label in base.cluster.unique():
        components.setdefault(uf.find(label), []).append(label)
    replacement = {}
    for labels in components.values():
        chosen = min(labels, key=label_key)
        for label in labels:
            replacement[label] = chosen
    out = base.copy()
    out['cluster'] = out['cluster'].map(replacement)
    output = Path(output_path) if output_path is not None else root / 'paper_submissions' / 'extended_lynx_salamander_refinement.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, lineterminator='\r\n')
    return output, hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    path, digest = reproduce()
    print(path)
    print(digest)
