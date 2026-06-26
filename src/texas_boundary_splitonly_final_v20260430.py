
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import texas_astrodot_2025reuse_v20260426 as base
import texas_astrodot_boundary_suppressed_v20260429 as boundary


VERSION = "texas_boundary_splitonly_final_v20260430"
TEXAS = base.TEXAS
SPECIES_ORDER = base.core.SPECIES_ORDER


class UnionFind:
    def __init__(self, values):
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

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final Texas split-only verifier. Starts from the current 0.32583 "
            "submission, computes boundary-suppressed Texas belly-dot scores, "
            "and only splits existing Texas clusters. No cross-cluster merges."
        )
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sam-manifest", type=str, default=None)
    parser.add_argument("--current-best-submission", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-side", type=int, default=760)
    parser.add_argument("--texas-canvas-w", type=int, default=224)
    parser.add_argument("--texas-canvas-h", type=int, default=320)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=18)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} must contain image_id and cluster columns")
    out = df[["image_id", "cluster"]].copy()
    out["image_id"] = out["image_id"].astype(int)
    out["cluster"] = out["cluster"].astype(str)
    return out


def score_within_current_clusters(items: list[base.TexasDotItem]) -> pd.DataFrame:
    by_cluster: dict[str, list[base.TexasDotItem]] = {}
    for item in items:
        by_cluster.setdefault(item.current_cluster, []).append(item)
    rows = []
    total = sum(len(v) * (len(v) - 1) // 2 for v in by_cluster.values())
    done = 0
    for cluster, members in sorted(by_cluster.items(), key=lambda kv: (min(it.image_id for it in kv[1]), kv[0])):
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(sorted(members, key=lambda it: it.image_id), 2):
            score = boundary.texas_pair_score_boundary(a, b)
            rows.append(
                {
                    "species": TEXAS,
                    "image_id_a": int(a.image_id),
                    "image_id_b": int(b.image_id),
                    "current_cluster": cluster,
                    "current_cluster_a": a.current_cluster,
                    "current_cluster_b": b.current_cluster,
                    **score,
                }
            )
            done += 1
            if done % 1000 == 0:
                print(f"[Texas split-only] scored {done}/{total} intra-cluster pairs")
    return pd.DataFrame(rows)


def preserve_cluster(members: list[int], label: str) -> dict[int, str]:
    return {int(i): label for i in members}


def split_cluster_by_edges(
    members: list[int],
    old_label: str,
    pair_scores: pd.DataFrame,
    variant: str,
    keep_thr: float,
    min_point: float,
    min_stack: float,
    preserve_up_to: int,
    min_edges_per_component: int,
) -> tuple[dict[int, str], dict]:
    members = sorted(map(int, members))
    n = len(members)
    if n <= preserve_up_to:
        return preserve_cluster(members, old_label), {
            "variant": variant,
            "old_cluster": old_label,
            "old_size": n,
            "action": "preserve_small",
            "new_sizes": str(n),
            "accepted_edges": 0,
        }

    inside = pair_scores[pair_scores["current_cluster"].eq(old_label)].copy()
    if inside.empty:
        return preserve_cluster(members, old_label), {
            "variant": variant,
            "old_cluster": old_label,
            "old_size": n,
            "action": "preserve_no_scores",
            "new_sizes": str(n),
            "accepted_edges": 0,
        }

    # Keep only interior belly-dot evidence. Correlation alone is not enough:
    # the whole point is avoiding silhouette-driven false links.
    accepted = inside[
        (inside["score"].astype(float) >= keep_thr)
        & (
            (inside["point_score"].astype(float) >= min_point)
            | (inside["stack_gain"].astype(float) >= min_stack)
        )
    ].copy()

    if accepted.empty:
        return preserve_cluster(members, old_label), {
            "variant": variant,
            "old_cluster": old_label,
            "old_size": n,
            "action": "preserve_no_trusted_edges",
            "new_sizes": str(n),
            "accepted_edges": 0,
        }

    uf = UnionFind(members)
    for row in accepted.sort_values(["score", "point_score", "stack_gain"], ascending=False).itertuples(index=False):
        uf.union(int(row.image_id_a), int(row.image_id_b))

    comps: dict[int, list[int]] = {}
    for image_id in members:
        comps.setdefault(uf.find(image_id), []).append(image_id)
    parts = sorted([sorted(v) for v in comps.values()], key=lambda xs: (min(xs), len(xs), max(xs)))

    if len(parts) == 1:
        return preserve_cluster(members, old_label), {
            "variant": variant,
            "old_cluster": old_label,
            "old_size": n,
            "action": "preserve_connected",
            "new_sizes": str(n),
            "accepted_edges": int(len(accepted)),
        }

    # Avoid destructive atomization. If most parts are isolated and the cluster
    # is not huge, this may just be poor image quality rather than a false merge.
    singleton_parts = sum(1 for p in parts if len(p) == 1)
    if singleton_parts >= len(parts) - 1 and n <= 7:
        return preserve_cluster(members, old_label), {
            "variant": variant,
            "old_cluster": old_label,
            "old_size": n,
            "action": "preserve_atomization_guard",
            "new_sizes": " ".join(map(str, [len(p) for p in parts])),
            "accepted_edges": int(len(accepted)),
        }

    # Require at least a tiny amount of support inside every multi-image part.
    # Singletons are allowed because they are the suspected weak outliers.
    if min_edges_per_component > 0:
        for part in parts:
            if len(part) <= 1:
                continue
            part_set = set(part)
            edge_count = int(
                accepted["image_id_a"].isin(part_set).mul(accepted["image_id_b"].isin(part_set)).sum()
            )
            if edge_count < min_edges_per_component:
                return preserve_cluster(members, old_label), {
                    "variant": variant,
                    "old_cluster": old_label,
                    "old_size": n,
                    "action": "preserve_component_support_guard",
                    "new_sizes": " ".join(map(str, [len(p) for p in parts])),
                    "accepted_edges": int(len(accepted)),
                }

    out: dict[int, str] = {}
    for idx, part in enumerate(parts):
        new_label = f"{old_label}__{variant}_{idx}"
        for image_id in part:
            out[image_id] = new_label
    return out, {
        "variant": variant,
        "old_cluster": old_label,
        "old_size": n,
        "action": "split",
        "new_sizes": " ".join(map(str, [len(p) for p in parts])),
        "accepted_edges": int(len(accepted)),
    }


def splitonly_labels(
    items: list[base.TexasDotItem],
    pair_scores: pd.DataFrame,
    variant: str,
    keep_thr: float,
    min_point: float,
    min_stack: float,
    preserve_up_to: int,
    min_edges_per_component: int,
) -> tuple[dict[int, str], pd.DataFrame]:
    by_cluster: dict[str, list[int]] = {}
    for item in items:
        by_cluster.setdefault(item.current_cluster, []).append(int(item.image_id))
    labels: dict[int, str] = {}
    reports = []
    for old_label, members in sorted(by_cluster.items(), key=lambda kv: (min(kv[1]), len(kv[1]), kv[0])):
        cluster_labels, report = split_cluster_by_edges(
            members,
            old_label,
            pair_scores,
            variant,
            keep_thr,
            min_point,
            min_stack,
            preserve_up_to,
            min_edges_per_component,
        )
        labels.update(cluster_labels)
        reports.append(report)
    return labels, pd.DataFrame(reports)


def compact_labels(submission: pd.DataFrame, test_rows: pd.DataFrame) -> pd.DataFrame:
    species_map = dict(zip(test_rows["image_id"].astype(int), test_rows["species_id"].astype(str)))
    labels = dict(zip(submission["image_id"].astype(int), submission["cluster"].astype(str)))
    out_map: dict[int, str] = {}
    for species in SPECIES_ORDER:
        ids = sorted([i for i, sp in species_map.items() if sp == species])
        groups: dict[str, list[int]] = {}
        for image_id in ids:
            groups.setdefault(labels[image_id], []).append(image_id)
        for idx, members in enumerate(sorted(groups.values(), key=lambda xs: (min(xs), len(xs), max(xs)))):
            label = f"cluster_{species}_{idx}"
            for image_id in members:
                out_map[image_id] = label
    out = submission.copy()
    out["cluster"] = out["image_id"].astype(int).map(out_map)
    if out["cluster"].isna().any():
        raise ValueError("Compacted labels contain missing values")
    return out


def build_submission(
    current: pd.DataFrame,
    test_rows: pd.DataFrame,
    texas_labels: dict[int, str],
    out_path: Path,
) -> pd.DataFrame:
    current_map = dict(zip(current["image_id"].astype(int), current["cluster"].astype(str)))
    texas_ids = set(test_rows.loc[test_rows["species_id"].eq(TEXAS), "image_id"].astype(int).tolist())
    sub = current.copy()
    sub["cluster"] = sub["image_id"].astype(int).map(
        lambda i: texas_labels.get(i, current_map[i]) if i in texas_ids else current_map[i]
    )
    sub = compact_labels(sub, test_rows)
    sub.to_csv(out_path, index=False)
    return sub


def summarize_submission(sub: pd.DataFrame, current: pd.DataFrame, test_rows: pd.DataFrame, variant: str) -> list[dict]:
    cur_map = dict(zip(current["image_id"].astype(int), current["cluster"].astype(str)))
    sub_map = dict(zip(sub["image_id"].astype(int), sub["cluster"].astype(str)))
    rows = []
    for species in SPECIES_ORDER:
        ids = test_rows.loc[test_rows["species_id"].eq(species), "image_id"].astype(int).tolist()
        labels = [sub_map[i] for i in ids]
        counts = pd.Series(labels).value_counts()
        cur_groups: dict[str, set[int]] = {}
        sub_groups: dict[str, set[int]] = {}
        for image_id in ids:
            cur_groups.setdefault(cur_map[image_id], set()).add(image_id)
            sub_groups.setdefault(sub_map[image_id], set()).add(image_id)
        cur_membership = {image_id: cur_groups[cur_map[image_id]] for image_id in ids}
        sub_membership = {image_id: sub_groups[sub_map[image_id]] for image_id in ids}
        partition_changed = int(sum(1 for image_id in ids if cur_membership[image_id] != sub_membership[image_id]))
        rows.append(
            {
                "variant": variant,
                "species": species,
                "n_images": int(len(ids)),
                "n_clusters": int(counts.shape[0]),
                "singletons": int((counts == 1).sum()),
                "max_cluster_size": int(counts.max()) if not counts.empty else 0,
                "rows_changed_vs_current": partition_changed,
            }
        )
    return rows


def validate_upload(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != ["image_id", "cluster"]:
        raise ValueError("Submission columns must be exactly image_id, cluster")
    if submission["image_id"].astype(int).tolist() != sample["image_id"].astype(int).tolist():
        raise ValueError("Submission image order does not match sample_submission.csv")
    if submission["cluster"].isna().any():
        raise ValueError("Submission contains missing cluster labels")
    max_len = int(submission["cluster"].astype(str).str.len().max())
    if max_len > 64:
        raise ValueError(f"Cluster labels too long: max length {max_len}")


def choose_safe_candidate(candidate_report: pd.DataFrame) -> str:
    texas = candidate_report[candidate_report["species"].eq(TEXAS)].copy()
    safe = texas[
        (texas["rows_changed_vs_current"].astype(int) > 0)
        & (texas["rows_changed_vs_current"].astype(int) <= 70)
        & (texas["n_clusters"].astype(int).between(80, 92))
        & (texas["max_cluster_size"].astype(int) <= 26)
    ].copy()
    priority = {"balanced": 0, "strict": 1, "mild": 2}
    if safe.empty:
        return "current_passthrough"
    safe["priority"] = safe["variant"].map(priority).fillna(99).astype(int)
    safe = safe.sort_values(["priority", "rows_changed_vs_current", "n_clusters"])
    return str(safe.iloc[0]["variant"])


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.save_visualizations = True

    data_root = base.find_data_root(args.data_root)
    sam_manifest = base.find_sam_manifest(args.sam_manifest)
    metadata, manifest_info = base.prepare_metadata(data_root, sam_manifest)
    metadata = metadata[metadata["species_id"].isin(SPECIES_ORDER)].copy()
    test_rows = metadata[metadata["split"].eq("test")].copy()
    sample = pd.read_csv(data_root / "sample_submission.csv")[["image_id"]].copy()
    sample["image_id"] = sample["image_id"].astype(int)

    output_root = Path(args.output_root) if args.output_root else Path.cwd() / f"animalclef_{VERSION}"
    reports_dir = output_root / "reports"
    sub_dir = output_root / "submissions"
    viz_dir = output_root / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        viz_dir.mkdir(parents=True, exist_ok=True)

    current_best_path = boundary.find_current_best_boundary(args.current_best_submission, data_root)
    current = load_submission(current_best_path)
    validate_upload(current, sample)
    current_labels = dict(zip(current["image_id"].astype(int), current["cluster"].astype(str)))

    print(f"VERSION={VERSION}")
    print(f"data_root={data_root}")
    print(f"sam_manifest={sam_manifest}")
    print(f"current_best={current_best_path}")
    print(f"output_root={output_root}")
    print(json.dumps(manifest_info, indent=2))

    texas_rows = test_rows[test_rows["species_id"].eq(TEXAS)].sort_values("image_id").copy()
    if args.smoke:
        texas_rows = texas_rows.head(42)

    items: list[base.TexasDotItem] = []
    for idx, row in enumerate(texas_rows.to_dict("records"), start=1):
        image_id = int(row["image_id"])
        try:
            items.append(boundary.texas_belly_template_boundary(row, current_labels[image_id], args))
        except Exception as exc:
            print(f"[warn] Texas template failed image_id={image_id}: {exc}")
        if idx % 75 == 0:
            print(f"[Texas split-only] templates {idx}/{len(texas_rows)}")

    if args.save_visualizations:
        base.save_texas_preview(items, viz_dir / f"{VERSION}_Texas_template_preview.jpg", args.visual_limit)

    pair_scores = score_within_current_clusters(items)
    pair_scores.to_csv(reports_dir / f"{VERSION}_Texas_intra_cluster_pair_scores.csv", index=False)

    variants = {
        "mild": dict(keep_thr=0.285, min_point=0.18, min_stack=0.38, preserve_up_to=6, min_edges_per_component=0),
        "balanced": dict(keep_thr=0.310, min_point=0.24, min_stack=0.42, preserve_up_to=4, min_edges_per_component=0),
        "strict": dict(keep_thr=0.340, min_point=0.30, min_stack=0.46, preserve_up_to=3, min_edges_per_component=0),
    }

    candidate_rows: list[dict] = []
    split_reports = []
    outputs: dict[str, str] = {}

    # Keep a no-op copy so the notebook can fall back safely if shape guards fail.
    passthrough_path = sub_dir / f"submission_{VERSION}_current_passthrough.csv"
    current_compact = compact_labels(current, test_rows)
    current_compact.to_csv(passthrough_path, index=False)
    outputs["current_passthrough"] = str(passthrough_path)
    candidate_rows.extend(summarize_submission(current_compact, current, test_rows, "current_passthrough"))

    for variant, params in variants.items():
        labels, split_report = splitonly_labels(items, pair_scores, variant=variant, **params)
        split_report["variant"] = variant
        split_reports.append(split_report)
        out_path = sub_dir / f"submission_{VERSION}_{variant}.csv"
        sub = build_submission(current, test_rows, labels, out_path)
        validate_upload(sub, sample)
        outputs[variant] = str(out_path)
        candidate_rows.extend(summarize_submission(sub, current, test_rows, variant))
        print(f"wrote {out_path}")

    candidate_report = pd.DataFrame(candidate_rows)
    candidate_report.to_csv(reports_dir / f"{VERSION}_candidate_report.csv", index=False)
    if split_reports:
        pd.concat(split_reports, ignore_index=True).to_csv(reports_dir / f"{VERSION}_split_report.csv", index=False)
    else:
        pd.DataFrame().to_csv(reports_dir / f"{VERSION}_split_report.csv", index=False)

    selected = choose_safe_candidate(candidate_report)
    selected_path = Path(outputs[selected])
    final = pd.read_csv(selected_path)
    validate_upload(final, sample)
    top_level = output_root / "submission.csv"
    final.to_csv(top_level, index=False)

    selection = {
        "selected_variant": selected,
        "selected_path": str(selected_path),
        "top_level_submission": str(top_level),
        "outputs": outputs,
        "manifest_info": manifest_info,
    }
    (reports_dir / f"{VERSION}_selected_submission.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    summary = {
        "version": VERSION,
        "data_root": str(data_root),
        "sam_manifest": str(sam_manifest) if sam_manifest else None,
        "current_best": str(current_best_path),
        "manifest_info": manifest_info,
        "texas_items": int(len(items)),
        "texas_intra_pairs": int(len(pair_scores)),
        "selected_variant": selected,
        "outputs": outputs,
    }
    (reports_dir / f"{VERSION}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCandidate report:")
    print(candidate_report.to_string(index=False))
    print("\nSelected variant:", selected)
    print("Top-level submission:", top_level)


if __name__ == "__main__":
    main()
