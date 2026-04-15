from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DatasetAnalysis:
    dataset_name: str
    source_name: str
    row_index: int
    query: dict[str, float]
    raw_shape: tuple[int, int]
    numeric_shape: tuple[int, int]
    candidate_count: int
    feature_columns: list[str]
    numeric_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    feature_summary: pd.DataFrame
    baseline_result: pd.DataFrame
    proposed_result: pd.DataFrame
    time_comparison: pd.DataFrame
    target_distribution: pd.DataFrame | None = None


def prepare_numeric_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    prepared = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.dropna().reset_index(drop=True)
    return prepared


def min_max_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    mins = frame.min()
    spans = (frame.max() - mins).replace(0, 1)
    return (frame - mins) / spans


def build_query(
    frame: pd.DataFrame,
    feature_columns: list[str],
    query_perturbation: dict[str, float],
    seed: int = 42,
) -> tuple[int, dict[str, float]]:
    if frame.empty:
        raise ValueError("После подготовки не осталось строк для построения запроса.")

    rng = np.random.default_rng(seed)
    row_index = int(rng.integers(0, len(frame)))
    source_row = frame.iloc[row_index]
    query = {
        feature: float(source_row[feature]) * query_perturbation[feature]
        for feature in feature_columns
    }
    return row_index, query


def relative_difference(value: float, query_value: float, eps: float = 1e-9) -> float:
    return abs(value - query_value) / max(abs(query_value), eps)


def baseline_score(row: pd.Series, query: dict[str, float]) -> float:
    deltas = [relative_difference(row[column], query[column]) for column in query]
    return float(np.exp(-sum(deltas)))


def baseline_search(frame: pd.DataFrame, query: dict[str, float], top_k: int = 10) -> pd.DataFrame:
    result = frame.copy()
    result["score"] = result.apply(lambda row: baseline_score(row, query), axis=1)
    return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def filter_candidates(
    frame: pd.DataFrame,
    query: dict[str, float],
    relative_window: float = 0.20,
    minimum_span: float = 1.0,
) -> pd.DataFrame:
    filtered = frame.copy()
    for column, query_value in query.items():
        span = max(abs(query_value) * relative_window, minimum_span)
        left = query_value - span
        right = query_value + span
        filtered = filtered[(filtered[column] >= left) & (filtered[column] <= right)]
    return filtered


def proposed_score(
    row: pd.Series,
    query: dict[str, float],
    weights: dict[str, float] | None = None,
    alpha: float = 3.0,
) -> float:
    if weights is None:
        weights = {column: 1.0 for column in query}

    score = 0.0
    for column in query:
        distance = relative_difference(row[column], query[column])
        score += weights[column] * np.exp(-alpha * distance)
    return float(score)


def proposed_search(
    frame: pd.DataFrame,
    query: dict[str, float],
    top_k: int = 10,
    relative_window: float = 0.20,
    weights: dict[str, float] | None = None,
    alpha: float = 3.0,
    minimum_span: float = 1.0,
) -> tuple[pd.DataFrame, int]:
    candidates = filter_candidates(
        frame,
        query,
        relative_window=relative_window,
        minimum_span=minimum_span,
    )
    if candidates.empty:
        return candidates.copy(), 0

    ranked = candidates.copy()
    ranked["score"] = ranked.apply(
        lambda row: proposed_score(row, query, weights=weights, alpha=alpha),
        axis=1,
    )
    ranked = ranked.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return ranked, len(candidates)


def measure_time(function, *args, repeats: int = 30, **kwargs) -> tuple[float, float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args, **kwargs)
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings)), float(np.std(timings))


def build_feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "min": frame.min(),
            "max": frame.max(),
            "mean": frame.mean(),
            "std": frame.std(),
        }
    ).reset_index(names="feature")


def build_target_distribution(frame: pd.DataFrame, target_column: str) -> pd.DataFrame | None:
    if target_column not in frame.columns:
        return None

    counts = frame[target_column].value_counts(dropna=False).reset_index()
    counts.columns = [target_column, "count"]
    return counts


def run_analysis(
    dataset_name: str,
    frame: pd.DataFrame,
    source_name: str,
    feature_columns: list[str],
    query_perturbation: dict[str, float],
    weights: dict[str, float],
    relative_window: float = 0.20,
    alpha: float = 3.0,
    top_k: int = 5,
    repeats: int = 20,
    target_column: str | None = None,
    minimum_span: float = 1.0,
) -> DatasetAnalysis:
    numeric_frame = prepare_numeric_frame(frame, feature_columns)
    normalized_frame = min_max_normalize(numeric_frame)
    row_index, query = build_query(numeric_frame, feature_columns, query_perturbation)

    baseline_result = baseline_search(numeric_frame, query, top_k=top_k)
    proposed_result, candidate_count = proposed_search(
        numeric_frame,
        query,
        top_k=top_k,
        relative_window=relative_window,
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
        query,
        top_k=top_k,
        relative_window=relative_window,
        weights=weights,
        alpha=alpha,
        minimum_span=minimum_span,
        repeats=repeats,
    )

    time_comparison = pd.DataFrame(
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
        raw_shape=frame.shape,
        numeric_shape=numeric_frame.shape,
        candidate_count=candidate_count,
        feature_columns=feature_columns,
        numeric_frame=numeric_frame,
        normalized_frame=normalized_frame,
        feature_summary=build_feature_summary(numeric_frame),
        baseline_result=baseline_result,
        proposed_result=proposed_result,
        time_comparison=time_comparison,
        target_distribution=target_distribution,
    )
