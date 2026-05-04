from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from numpy import exp, mean, std
from numpy.random import default_rng
from pandas import DataFrame, Series, concat, to_numeric


@dataclass(slots=True)
class DatasetAnalysis:
    dataset_name: str
    source_name: str
    row_index: int
    query: dict[str, float]
    quantized_query: dict[str, int]
    raw_shape: tuple[int, int]
    numeric_shape: tuple[int, int]
    candidate_count: int
    feature_columns: list[str]
    numeric_frame: DataFrame
    normalized_frame: DataFrame
    quantized_frame: DataFrame
    feature_summary: DataFrame
    quantile_summary: DataFrame
    query_summary: DataFrame
    group_distribution: DataFrame
    filter_steps: DataFrame
    baseline_result: DataFrame
    proposed_result: DataFrame
    time_comparison: DataFrame
    experiment_summary: DataFrame
    target_distribution: DataFrame | None = None


class AnalyticalDataset(ABC):
    dataset_name: str
    feature_columns: list[str]
    query_perturbation: dict[str, float]
    weights: dict[str, float]
    relative_window: float = 0.20
    neighbor_radius: int = 1
    alpha: float = 3.0
    top_k: int = 5
    repeats: int = 20
    target_column: str | None = None
    minimum_span: float = 1.0

    @abstractmethod
    def load_dataset(self) -> tuple[DataFrame, str]:
        """Load the raw dataset and return the source name for the UI."""

    def analyze(self) -> DatasetAnalysis:
        frame, source_name = self.load_dataset()
        return run_analysis(
            dataset_name=self.dataset_name,
            frame=frame,
            source_name=source_name,
            feature_columns=self.feature_columns,
            query_perturbation=self.query_perturbation,
            weights=self.weights,
            relative_window=self.relative_window,
            neighbor_radius=self.neighbor_radius,
            alpha=self.alpha,
            top_k=self.top_k,
            repeats=self.repeats,
            target_column=self.target_column,
            minimum_span=self.minimum_span,
        )

    @staticmethod
    def download_if_missing(url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            return target

        print(f"Скачивание: {url}")
        with urlopen(url, timeout=300) as response, target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return target


def prepare_numeric_frame(frame: DataFrame, feature_columns: list[str]) -> DataFrame:
    prepared = frame[feature_columns].apply(to_numeric, errors="coerce")
    prepared = prepared.dropna().reset_index(drop=True)
    return prepared


def min_max_normalize(frame: DataFrame) -> DataFrame:
    mins = frame.min()
    spans = (frame.max() - mins).replace(0, 1)
    return (frame - mins) / spans


def build_quantile_summary(frame: DataFrame) -> DataFrame:
    quantiles = frame.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).transpose().reset_index(names="feature")
    quantiles.columns = ["feature", "min", "q1", "median", "q3", "max"]
    return quantiles


def build_quantile_bounds(quantile_summary: DataFrame) -> dict[str, tuple[float, float, float]]:
    return {
        row.feature: (float(row.q1), float(row.median), float(row.q3))
        for row in quantile_summary.itertuples(index=False)
    }


def assign_quantile_group(value: float, bounds: tuple[float, float, float]) -> int:
    q1, median, q3 = bounds
    if value <= q1:
        return 0
    if value <= median:
        return 1
    if value <= q3:
        return 2
    return 3


def quantize_frame(
    frame: DataFrame,
    feature_columns: list[str],
    quantile_bounds: dict[str, tuple[float, float, float]],
) -> DataFrame:
    quantized = DataFrame(index=frame.index)
    for feature in feature_columns:
        bounds = quantile_bounds[feature]
        quantized[feature] = frame[feature].map(lambda value: assign_quantile_group(float(value), bounds))
    return quantized.astype(int)


def quantize_query(
    query: dict[str, float],
    quantile_bounds: dict[str, tuple[float, float, float]],
) -> dict[str, int]:
    return {
        feature: assign_quantile_group(float(value), quantile_bounds[feature])
        for feature, value in query.items()
    }


def build_query(
    frame: DataFrame,
    feature_columns: list[str],
    query_perturbation: dict[str, float],
    seed: int = 42,
) -> tuple[int, dict[str, float]]:
    if frame.empty:
        raise ValueError("После подготовки не осталось строк для построения запроса.")

    rng = default_rng(seed)
    row_index = int(rng.integers(0, len(frame)))
    source_row = frame.iloc[row_index]
    query = {
        feature: float(source_row[feature]) * query_perturbation[feature]
        for feature in feature_columns
    }
    return row_index, query


def relative_difference(value: float, query_value: float, eps: float = 1e-9) -> float:
    return abs(value - query_value) / max(abs(query_value), eps)


def baseline_score(row: Series, query: dict[str, float]) -> float:
    deltas = [relative_difference(row[column], query[column]) for column in query]
    return float(exp(-sum(deltas)))


def baseline_search(frame: DataFrame, query: dict[str, float], top_k: int = 10) -> DataFrame:
    result = frame.copy()
    result.insert(0, "record_id", frame.index)
    result["score"] = result.apply(lambda row: baseline_score(row, query), axis=1)
    return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def build_query_summary(
    query: dict[str, float],
    quantized_query: dict[str, int],
    neighbor_radius: int = 1,
) -> DataFrame:
    rows = []
    for feature, value in query.items():
        query_group = quantized_query[feature]
        left_group = max(0, query_group - neighbor_radius)
        right_group = min(3, query_group + neighbor_radius)
        rows.append(
            {
                "feature": feature,
                "query_value": value,
                "query_group": query_group,
                "allowed_groups": f"{left_group}-{right_group}",
            }
        )
    return DataFrame(rows)


def filter_by_quantile_groups(
    frame: DataFrame,
    quantized_frame: DataFrame,
    query: dict[str, float],
    quantized_query: dict[str, int],
    neighbor_radius: int = 1,
) -> tuple[DataFrame, DataFrame]:
    current_mask = Series(True, index=frame.index)
    steps: list[dict[str, str | float | int]] = []

    for feature, query_group in quantized_query.items():
        left_group = max(0, query_group - neighbor_radius)
        right_group = min(3, query_group + neighbor_radius)
        allowed_groups = list(range(left_group, right_group + 1))
        current_mask &= quantized_frame[feature].isin(allowed_groups)
        steps.append(
            {
                "stage": "Квартильный фильтр",
                "feature": feature,
                "query_value": query[feature],
                "query_group": query_group,
                "allowed_groups": ", ".join(str(group) for group in allowed_groups),
                "matched_rows": int(current_mask.sum()),
            }
        )

    filtered = frame.loc[current_mask].copy()
    return filtered, DataFrame(steps)


def filter_by_relative_window(
    frame: DataFrame,
    query: dict[str, float],
    relative_window: float = 0.20,
    minimum_span: float = 1.0,
) -> tuple[DataFrame, DataFrame]:
    filtered = frame.copy()
    steps: list[dict[str, str | float | int]] = []

    for feature, query_value in query.items():
        span = max(abs(query_value) * relative_window, minimum_span)
        left = query_value - span
        right = query_value + span
        filtered = filtered[(filtered[feature] >= left) & (filtered[feature] <= right)].copy()
        steps.append(
            {
                "stage": "Числовое окно",
                "feature": feature,
                "query_value": query_value,
                "window_left": left,
                "window_right": right,
                "matched_rows": len(filtered),
            }
        )

    return filtered, DataFrame(steps)


def filter_candidates(
    frame: DataFrame,
    quantized_frame: DataFrame,
    query: dict[str, float],
    quantized_query: dict[str, int],
    relative_window: float = 0.20,
    neighbor_radius: int = 1,
    minimum_span: float = 1.0,
) -> tuple[DataFrame, DataFrame]:
    quartile_filtered, quartile_steps = filter_by_quantile_groups(
        frame,
        quantized_frame,
        query,
        quantized_query,
        neighbor_radius=neighbor_radius,
    )
    window_filtered, window_steps = filter_by_relative_window(
        quartile_filtered,
        query,
        relative_window=relative_window,
        minimum_span=minimum_span,
    )
    return window_filtered, concat([quartile_steps, window_steps], ignore_index=True)


def proposed_score(
    row: Series,
    query: dict[str, float],
    weights: dict[str, float] | None = None,
    alpha: float = 3.0,
) -> float:
    if weights is None:
        weights = {column: 1.0 for column in query}

    score = 0.0
    for column in query:
        distance = relative_difference(row[column], query[column])
        score += weights[column] * exp(-alpha * distance)
    return float(score)


def proposed_search(
    frame: DataFrame,
    quantized_frame: DataFrame,
    query: dict[str, float],
    quantized_query: dict[str, int],
    top_k: int = 10,
    relative_window: float = 0.20,
    neighbor_radius: int = 1,
    weights: dict[str, float] | None = None,
    alpha: float = 3.0,
    minimum_span: float = 1.0,
) -> tuple[DataFrame, int, DataFrame]:
    candidates, filter_steps = filter_candidates(
        frame,
        quantized_frame,
        query,
        quantized_query,
        relative_window=relative_window,
        neighbor_radius=neighbor_radius,
        minimum_span=minimum_span,
    )
    if candidates.empty:
        empty = candidates.copy()
        if "record_id" not in empty.columns:
            empty.insert(0, "record_id", [])
        return empty, 0, filter_steps

    ranked = candidates.copy()
    ranked.insert(0, "record_id", candidates.index)
    ranked["score"] = ranked.apply(
        lambda row: proposed_score(row, query, weights=weights, alpha=alpha),
        axis=1,
    )
    ranked = ranked.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return ranked, len(candidates), filter_steps


def measure_time(function, *args, repeats: int = 30, **kwargs) -> tuple[float, float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args, **kwargs)
        timings.append(time.perf_counter() - start)
    return float(mean(timings)), float(std(timings))


def build_feature_summary(frame: DataFrame) -> DataFrame:
    return DataFrame(
        {
            "min": frame.min(),
            "max": frame.max(),
            "mean": frame.mean(),
            "std": frame.std(),
        }
    ).reset_index(names="feature")


def build_group_distribution(quantized_frame: DataFrame) -> DataFrame:
    rows = []
    total_count = len(quantized_frame)
    for feature in quantized_frame.columns:
        counts = quantized_frame[feature].value_counts().sort_index()
        for group, count in counts.items():
            rows.append(
                {
                    "feature": feature,
                    "group": int(group),
                    "count": int(count),
                    "share": float(count / total_count) if total_count else 0.0,
                }
            )
    return DataFrame(rows)


def build_target_distribution(frame: DataFrame, target_column: str) -> DataFrame | None:
    if target_column not in frame.columns:
        return None

    counts = frame[target_column].value_counts(dropna=False).reset_index()
    counts.columns = [target_column, "count"]
    return counts


def build_experiment_summary(
    numeric_frame: DataFrame,
    candidate_count: int,
    baseline_result: DataFrame,
    proposed_result: DataFrame,
    time_comparison: DataFrame,
) -> DataFrame:
    baseline_mean = float(
        time_comparison.loc[time_comparison["method"] == "baseline", "mean_seconds"].iloc[0]
    )
    proposed_mean = float(
        time_comparison.loc[time_comparison["method"] == "proposed", "mean_seconds"].iloc[0]
    )
    time_saved = baseline_mean - proposed_mean
    speedup_percent = (time_saved / baseline_mean * 100.0) if baseline_mean else 0.0
    candidate_reduction = (
        (1.0 - candidate_count / len(numeric_frame)) * 100.0 if len(numeric_frame) else 0.0
    )
    baseline_ids = set(baseline_result.get("record_id", Series(dtype="int64")).tolist())
    proposed_ids = set(proposed_result.get("record_id", Series(dtype="int64")).tolist())
    overlap_count = len(baseline_ids & proposed_ids)
    overlap_percent = (
        overlap_count / max(min(len(baseline_result), len(proposed_result)), 1) * 100.0
    )

    return DataFrame(
        [
            {"metric": "Всего записей", "value": len(numeric_frame)},
            {"metric": "После фильтра", "value": candidate_count},
            {"metric": "Сокращение кандидатов, %", "value": candidate_reduction},
            {"metric": "Базовый поиск, с", "value": baseline_mean},
            {"metric": "Предлагаемый поиск, с", "value": proposed_mean},
            {"metric": "Экономия времени, с", "value": time_saved},
            {"metric": "Ускорение, %", "value": speedup_percent},
            {"metric": "Пересечение top-k", "value": overlap_count},
            {"metric": "Совпадение top-k, %", "value": overlap_percent},
        ]
    )


def run_analysis(
    dataset_name: str,
    frame: DataFrame,
    source_name: str,
    feature_columns: list[str],
    query_perturbation: dict[str, float],
    weights: dict[str, float],
    relative_window: float = 0.20,
    neighbor_radius: int = 1,
    alpha: float = 3.0,
    top_k: int = 5,
    repeats: int = 20,
    target_column: str | None = None,
    minimum_span: float = 1.0,
) -> DatasetAnalysis:
    numeric_frame = prepare_numeric_frame(frame, feature_columns)
    normalized_frame = min_max_normalize(numeric_frame)
    quantile_summary = build_quantile_summary(numeric_frame)
    quantile_bounds = build_quantile_bounds(quantile_summary)
    quantized_frame = quantize_frame(numeric_frame, feature_columns, quantile_bounds)

    row_index, query = build_query(numeric_frame, feature_columns, query_perturbation)
    quantized_query = quantize_query(query, quantile_bounds)

    baseline_result = baseline_search(numeric_frame, query, top_k=top_k)
    proposed_result, candidate_count, filter_steps = proposed_search(
        numeric_frame,
        quantized_frame,
        query,
        quantized_query,
        top_k=top_k,
        relative_window=relative_window,
        neighbor_radius=neighbor_radius,
        weights=weights,
        alpha=alpha,
        minimum_span=minimum_span,
    )

    baseline_mean, baseline_std = measure_time(
        baseline_search,
        numeric_frame,
        query,
        top_k=top_k,
        repeats=repeats,
    )
    proposed_mean, proposed_std = measure_time(
        proposed_search,
        numeric_frame,
        quantized_frame,
        query,
        quantized_query,
        top_k=top_k,
        relative_window=relative_window,
        neighbor_radius=neighbor_radius,
        weights=weights,
        alpha=alpha,
        minimum_span=minimum_span,
        repeats=repeats,
    )

    time_comparison = DataFrame(
        [
            {
                "method": "baseline",
                "mean_seconds": baseline_mean,
                "std_seconds": baseline_std,
                "processed_rows": len(numeric_frame),
            },
            {
                "method": "proposed",
                "mean_seconds": proposed_mean,
                "std_seconds": proposed_std,
                "processed_rows": candidate_count,
            },
        ]
    )

    target_distribution = (
        build_target_distribution(frame, target_column)
        if target_column is not None
        else None
    )

    return DatasetAnalysis(
        dataset_name=dataset_name,
        source_name=source_name,
        row_index=row_index,
        query=query,
        quantized_query=quantized_query,
        raw_shape=frame.shape,
        numeric_shape=numeric_frame.shape,
        candidate_count=candidate_count,
        feature_columns=feature_columns,
        numeric_frame=numeric_frame,
        normalized_frame=normalized_frame,
        quantized_frame=quantized_frame,
        feature_summary=build_feature_summary(numeric_frame),
        quantile_summary=quantile_summary,
        query_summary=build_query_summary(query, quantized_query, neighbor_radius=neighbor_radius),
        group_distribution=build_group_distribution(quantized_frame),
        filter_steps=filter_steps,
        baseline_result=baseline_result,
        proposed_result=proposed_result,
        time_comparison=time_comparison,
        experiment_summary=build_experiment_summary(
            numeric_frame,
            candidate_count,
            baseline_result,
            proposed_result,
            time_comparison,
        ),
        target_distribution=target_distribution,
    )
