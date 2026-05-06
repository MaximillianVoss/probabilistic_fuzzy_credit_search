from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pandas.core.frame import DataFrame

from src.datasets.base import (
    AnalyticalDataset,
    assign_quantile_group,
    baseline_search,
    build_experiment_summary,
    build_target_distribution,
    build_query,
    min_max_normalize,
    prepare_numeric_frame,
    proposed_search,
    quantize_frame,
    quantize_query,
    build_quantile_bounds,
    build_quantile_summary,
)


class DatasetHelperTests(unittest.TestCase):
    def test_prepare_numeric_frame_drops_invalid_rows(self) -> None:
        frame = DataFrame(
            {
                "age": ["10", "bad", "30"],
                "amount": ["100", "200", None],
                "ignored": ["x", "y", "z"],
            }
        )

        prepared = prepare_numeric_frame(frame, ["age", "amount"])

        self.assertEqual(prepared.to_dict(orient="records"), [{"age": 10.0, "amount": 100.0}])

    def test_min_max_normalize_handles_constant_columns(self) -> None:
        frame = DataFrame({"x": [5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0]})

        normalized = min_max_normalize(frame)

        self.assertEqual(normalized["x"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(normalized["y"].tolist(), [0.0, 0.5, 1.0])

    def test_assign_quantile_group_handles_boundaries(self) -> None:
        bounds = (10.0, 20.0, 30.0)

        self.assertEqual(assign_quantile_group(5.0, bounds), 0)
        self.assertEqual(assign_quantile_group(10.0, bounds), 0)
        self.assertEqual(assign_quantile_group(15.0, bounds), 1)
        self.assertEqual(assign_quantile_group(20.0, bounds), 1)
        self.assertEqual(assign_quantile_group(25.0, bounds), 2)
        self.assertEqual(assign_quantile_group(30.0, bounds), 2)
        self.assertEqual(assign_quantile_group(35.0, bounds), 3)

    def test_build_query_is_deterministic_with_seed(self) -> None:
        frame = DataFrame({"age": [10.0, 20.0, 30.0], "amount": [100.0, 200.0, 300.0]})

        row_index, query = build_query(
            frame,
            ["age", "amount"],
            {"age": 1.1, "amount": 0.9},
            seed=7,
        )

        self.assertEqual(row_index, 1)
        self.assertEqual(query, {"age": 22.0, "amount": 180.0})

    def test_build_query_raises_for_empty_frame(self) -> None:
        with self.assertRaises(ValueError):
            build_query(DataFrame(columns=["age", "amount"]), ["age", "amount"], {"age": 1.0, "amount": 1.0})

    def test_baseline_search_prefers_exact_match(self) -> None:
        frame = DataFrame({"age": [20.0, 30.0, 40.0], "amount": [200.0, 300.0, 400.0]})

        ranked = baseline_search(frame, {"age": 30.0, "amount": 300.0}, top_k=2)

        self.assertEqual(int(ranked.iloc[0]["record_id"]), 1)
        self.assertAlmostEqual(float(ranked.iloc[0]["score"]), 1.0, places=6)

    def test_proposed_search_returns_empty_frame_when_no_candidates(self) -> None:
        frame = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
        quantile_summary = build_quantile_summary(frame)
        quantile_bounds = build_quantile_bounds(quantile_summary)
        quantized_frame = quantize_frame(frame, ["x", "y"], quantile_bounds)

        query = {"x": 100.0, "y": 1000.0}
        quantized_query = quantize_query(query, quantile_bounds)
        ranked, candidate_count, steps = proposed_search(
            frame,
            quantized_frame,
            query,
            quantized_query,
            top_k=3,
            relative_window=0.0,
            neighbor_radius=0,
            minimum_span=0.1,
        )

        self.assertTrue(ranked.empty)
        self.assertIn("record_id", ranked.columns)
        self.assertEqual(candidate_count, 0)
        self.assertEqual(len(steps), 4)

    def test_build_target_distribution_returns_none_for_missing_column(self) -> None:
        frame = DataFrame({"x": [1, 2, 3]})
        self.assertIsNone(build_target_distribution(frame, "missing"))

    def test_build_experiment_summary_calculates_overlap_and_speedup(self) -> None:
        numeric_frame = DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        baseline_result = DataFrame({"record_id": [0, 1, 2]})
        proposed_result = DataFrame({"record_id": [1, 2, 3]})
        time_comparison = DataFrame(
            [
                {"method": "baseline", "mean_seconds": 2.0, "std_seconds": 0.1, "processed_rows": 4},
                {"method": "proposed", "mean_seconds": 0.5, "std_seconds": 0.05, "processed_rows": 2},
            ]
        )

        summary = build_experiment_summary(
            numeric_frame,
            candidate_count=2,
            baseline_result=baseline_result,
            proposed_result=proposed_result,
            time_comparison=time_comparison,
        )

        metrics = dict(zip(summary["metric"], summary["value"]))
        self.assertEqual(metrics["После фильтра"], 2)
        self.assertAlmostEqual(metrics["Сокращение кандидатов, %"], 50.0)
        self.assertAlmostEqual(metrics["Экономия времени, с"], 1.5)
        self.assertAlmostEqual(metrics["Ускорение, %"], 75.0)
        self.assertEqual(metrics["Пересечение top-k"], 2)
        self.assertAlmostEqual(metrics["Совпадение top-k, %"], 66.66666666666666)

    def test_download_if_missing_keeps_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "file.txt"
            target.write_bytes(b"existing")

            result = AnalyticalDataset.download_if_missing("https://example.com/file.txt", target)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_download_if_missing_writes_chunks_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "file.txt"
            response = MagicMock()
            response.read.side_effect = [b"abc", b"def", b""]
            response.__enter__.return_value = response
            response.__exit__.return_value = None

            with patch("src.datasets.base.urlopen", return_value=response):
                result = AnalyticalDataset.download_if_missing("https://example.com/file.txt", target)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"abcdef")


if __name__ == "__main__":
    unittest.main()
