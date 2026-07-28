import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.readout_v2_cluster_statistics import (  # noqa: E402
    pooled_cluster_mean,
    resolve_cluster,
)


class ReadoutV2ClusterStatisticsTest(unittest.TestCase):
    def test_pooled_cluster_mean_weights_rows_not_clusters(self):
        self.assertEqual(pooled_cluster_mean([(1, 1), (-1, 3)]), 0.0)

    def test_resolve_cluster_rejects_reordered_source_rows(self):
        record = {"row_position": 0, "sample_index": "25"}
        source_rows = {"DynaMath": [{"index": "68", "qid": "68"}]}

        with self.assertRaisesRegex(ValueError, "row join mismatch"):
            resolve_cluster("DynaMath", record, source_rows)

    def test_resolve_cluster_rejects_empty_cluster_key(self):
        record = {"row_position": 0, "sample_index": "25"}
        source_rows = {"DynaMath": [{"index": "25", "qid": ""}]}

        with self.assertRaisesRegex(ValueError, "empty 'qid'"):
            resolve_cluster("DynaMath", record, source_rows)


if __name__ == "__main__":
    unittest.main()
