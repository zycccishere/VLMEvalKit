#!/usr/bin/env python3
"""Compute paired, source-question-clustered statistics for readout-v2 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


CLUSTER_SPECS = {
    "DynaMath": ("DynaMath.tsv", "qid"),
    "WeMath": ("WeMath.tsv", "ID"),
    "MMBench_DEV_EN_V11": (None, "row_position"),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_cluster_mean(
    cluster_sums_and_sizes: list[tuple[int, int]],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    cluster_count = len(cluster_sums_and_sizes)
    draws: list[float] = []
    for _ in range(resamples):
        delta_sum = 0
        row_count = 0
        for _ in range(cluster_count):
            cluster_sum, cluster_size = cluster_sums_and_sizes[
                rng.randrange(cluster_count)
            ]
            delta_sum += cluster_sum
            row_count += cluster_size
        draws.append(delta_sum / row_count)
    draws.sort()
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def exact_mcnemar_p(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = min(gained, lost)
    probability = sum(
        math.comb(discordant, count) for count in range(tail + 1)
    ) * (0.5**discordant)
    return min(1.0, 2.0 * probability)


def dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {
        (record["dataset"], record["row_position"]): record
        for record in manifest["records"]
    }
    if len(records) != len(manifest["records"]):
        raise ValueError("Manifest contains duplicate dataset/row_position keys")

    predictions: dict[tuple[str, int], dict] = {}
    for path in sorted(args.predictions_dir.glob("shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                prediction = json.loads(line)
                key = (prediction["dataset"], prediction["row_position"])
                if key in predictions:
                    raise ValueError(f"Duplicate prediction key: {key}")
                predictions[key] = prediction
    if predictions.keys() != records.keys():
        missing = sorted(records.keys() - predictions.keys())[:10]
        unexpected = sorted(predictions.keys() - records.keys())[:10]
        raise ValueError(f"Prediction key mismatch: missing={missing}, unexpected={unexpected}")

    source_rows: dict[str, list[dict[str, str]]] = {}
    for dataset, (filename, _) in CLUSTER_SPECS.items():
        if filename is not None:
            source_rows[dataset] = read_tsv(args.data_root / filename)

    output = {
        "schema": "topic-image-replay/readout-v2-cluster-statistics/v1",
        "method": {
            "estimand": "per-row paired accuracy difference",
            "primary_confidence_interval": (
                "percentile bootstrap resampling source-question clusters with replacement"
            ),
            "bootstrap_resamples": args.resamples,
            "bootstrap_master_seed": args.seed,
            "cluster_keys": {
                dataset: cluster_key
                for dataset, (_, cluster_key) in CLUSTER_SPECS.items()
            },
            "row_level_mcnemar": (
                "descriptive only when rows share a source-question cluster"
            ),
        },
        "provenance": {
            "records_sha256": manifest["records_sha256"],
            "implementation_sha256": manifest["implementation_sha256"],
            "repo_head": manifest["repo_snapshot"]["head"],
            "analysis_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "datasets": {},
    }

    for dataset in CLUSTER_SPECS:
        dataset_predictions = [
            prediction
            for (record_dataset, _), prediction in predictions.items()
            if record_dataset == dataset
        ]
        dataset_predictions.sort(key=lambda item: item["row_position"])
        _, cluster_key = CLUSTER_SPECS[dataset]

        by_cluster: dict[str, list[int]] = defaultdict(list)
        baseline_hits = 0
        readout_hits = 0
        full_hits = 0
        gained = 0
        lost = 0
        for prediction in dataset_predictions:
            position = prediction["row_position"]
            if cluster_key == "row_position":
                cluster = str(position)
            else:
                cluster = source_rows[dataset][position][cluster_key]
            baseline = int(prediction["conditions"]["baseline"]["hit"])
            readout = int(prediction["conditions"]["readout_v2"]["hit"])
            full = int(prediction["conditions"]["full"]["hit"])
            by_cluster[cluster].append(readout - baseline)
            baseline_hits += baseline
            readout_hits += readout
            full_hits += full
            gained += int(readout == 1 and baseline == 0)
            lost += int(readout == 0 and baseline == 1)

        row_count = len(dataset_predictions)
        cluster_sums_and_sizes = [
            (sum(differences), len(differences))
            for differences in by_cluster.values()
        ]
        bootstrap_seed = dataset_seed(args.seed, dataset)
        confidence_interval = bootstrap_cluster_mean(
            cluster_sums_and_sizes,
            resamples=args.resamples,
            seed=bootstrap_seed,
        )
        baseline_accuracy = baseline_hits / row_count
        readout_accuracy = readout_hits / row_count
        full_accuracy = full_hits / row_count
        full_gap = full_accuracy - baseline_accuracy
        group_size_histogram = Counter(
            len(differences) for differences in by_cluster.values()
        )

        output["datasets"][dataset] = {
            "n_rows": row_count,
            "cluster_key": cluster_key,
            "n_clusters": len(by_cluster),
            "cluster_size_histogram": {
                str(size): count for size, count in sorted(group_size_histogram.items())
            },
            "hits": {
                "baseline": baseline_hits,
                "readout_v2": readout_hits,
                "full": full_hits,
            },
            "accuracy": {
                "baseline": baseline_accuracy,
                "readout_v2": readout_accuracy,
                "full": full_accuracy,
            },
            "readout_minus_baseline": {
                "delta": readout_accuracy - baseline_accuracy,
                "cluster_bootstrap_95_ci": confidence_interval,
                "cluster_bootstrap_seed": bootstrap_seed,
                "gained_rows": gained,
                "lost_rows": lost,
                "row_level_mcnemar_exact_p_descriptive": exact_mcnemar_p(
                    gained, lost
                ),
            },
            "full_accuracy_gap_recovered": (
                (readout_accuracy - baseline_accuracy) / full_gap
                if full_gap != 0
                else None
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
