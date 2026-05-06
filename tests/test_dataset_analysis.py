from __future__ import annotations

import unittest

from pandas.core.frame import DataFrame

from src.datasets.base import (
    build_quantile_bounds,
    build_quantile_summary,
    filter_candidates,
    quantize_frame,
    quantize_query,
    run_analysis,
)


class DatasetAnalysisTests(unittest.TestCase):
    def test_quantize_frame_assigns_groups_zero_to_three(self) -> None:
        frame = DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [10.0, 20.0, 30.0, 40.0],
            }
        )

        quantile_summary = build_quantile_summary(frame)
        quantile_bounds = build_quantile_bounds(quantile_summary)
        quantized = quantize_frame(frame, ["x", "y"], quantile_bounds)

        self.assertEqual(quantized["x"].tolist(), [0, 1, 2, 3])
        self.assertEqual(quantized["y"].tolist(), [0, 1, 2, 3])

    def test_filter_candidates_uses_quartiles_and_window(self) -> None:
        frame = DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [10.0, 20.0, 30.0, 40.0],
            }
        )
        quantile_summary = build_quantile_summary(frame)
        quantile_bounds = build_quantile_bounds(quantile_summary)
        quantized_frame = quantize_frame(frame, ["x", "y"], quantile_bounds)

        query = {"x": 2.0, "y": 20.0}
        quantized_query = quantize_query(query, quantile_bounds)
        filtered, steps = filter_candidates(
            frame,
            quantized_frame,
            query,
            quantized_query,
            relative_window=0.0,
            neighbor_radius=0,
            minimum_span=0.6,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0].to_dict(), {"x": 2.0, "y": 20.0})
        self.assertEqual(steps["stage"].tolist(), ["Квартильный фильтр", "Квартильный фильтр", "Числовое окно", "Числовое окно"])
        self.assertEqual(int(steps.iloc[-1]["matched_rows"]), 1)

    def test_run_analysis_builds_full_quantization_workflow(self) -> None:
        frame = DataFrame(
            {
                "income": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "balance": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0],
                "duration": [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0],
                "target": ["A", "A", "B", "B", "A", "B", "A", "B"],
            }
        )

        analysis = run_analysis(
            dataset_name="Synthetic Dataset",
            frame=frame,
            source_name="unit-test",
            feature_columns=["income", "balance", "duration"],
            query_perturbation={"income": 1.0, "balance": 1.0, "duration": 1.0},
            weights={"income": 1.0, "balance": 1.0, "duration": 1.0},
            relative_window=0.2,
            neighbor_radius=1,
            alpha=3.0,
            top_k=3,
            repeats=2,
            target_column="target",
            minimum_span=1.0,
        )

        self.assertEqual(analysis.dataset_name, "Synthetic Dataset")
        self.assertEqual(analysis.raw_shape, (8, 4))
        self.assertEqual(analysis.numeric_shape, (8, 3))
        self.assertEqual(list(analysis.quantile_summary.columns), ["feature", "min", "q1", "median", "q3", "max"])
        self.assertEqual(list(analysis.query_summary.columns), ["feature", "query_value", "query_group", "allowed_groups"])
        self.assertEqual(len(analysis.filter_steps), 6)
        self.assertGreaterEqual(analysis.candidate_count, 1)
        self.assertLessEqual(analysis.candidate_count, len(analysis.numeric_frame))
        self.assertIn("Ускорение, %", analysis.experiment_summary["metric"].tolist())
        self.assertEqual(set(analysis.quantized_frame.columns), {"income", "balance", "duration"})
        self.assertTrue(set(analysis.quantized_query.values()).issubset({0, 1, 2, 3}))
        self.assertEqual(int(analysis.group_distribution.groupby("feature")["count"].sum().min()), len(analysis.numeric_frame))
        self.assertEqual(set(analysis.time_comparison["method"]), {"baseline", "proposed"})
        self.assertEqual(int(analysis.target_distribution["count"].sum()), len(frame))


if __name__ == "__main__":
    unittest.main()
