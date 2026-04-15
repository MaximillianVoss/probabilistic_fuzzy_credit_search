from __future__ import annotations

import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "german_credit"
LOCAL_KAGGLE_FILE = DATA_DIR / "german_credit_data.csv"
UCI_FILE = DATA_DIR / "german.data"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

FEATURE_COLUMNS = ["age", "credit_amount", "duration_months"]
QUERY_PERTURBATION = {
    "age": 1.05,
    "credit_amount": 0.95,
    "duration_months": 1.10,
}

UCI_COLUMNS = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "target",
]

KAGGLE_RENAME_MAP = {
    "Age": "age",
    "Credit amount": "credit_amount",
    "Duration": "duration_months",
}


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


def load_dataset() -> tuple[pd.DataFrame, str]:
    # Приоритет у локальной Kaggle-версии, иначе берём официальные данные UCI.
    if LOCAL_KAGGLE_FILE.exists():
        frame = pd.read_csv(LOCAL_KAGGLE_FILE).rename(columns=KAGGLE_RENAME_MAP)
        return frame, LOCAL_KAGGLE_FILE.name

    raw_path = download_if_missing(UCI_URL, UCI_FILE)
    frame = pd.read_csv(raw_path, sep=r"\s+", header=None, names=UCI_COLUMNS)
    return frame, raw_path.name


def prepare_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.dropna().reset_index(drop=True)
    return prepared


def min_max_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    mins = frame.min()
    spans = (frame.max() - mins).replace(0, 1)
    return (frame - mins) / spans


def build_query(frame: pd.DataFrame, seed: int = 42) -> tuple[int, dict[str, float]]:
    if frame.empty:
        raise ValueError("После подготовки не осталось строк для построения запроса.")

    rng = np.random.default_rng(seed)
    row_index = int(rng.integers(0, len(frame)))
    source_row = frame.iloc[row_index]
    query = {
        feature: float(source_row[feature]) * QUERY_PERTURBATION[feature]
        for feature in FEATURE_COLUMNS
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
) -> tuple[pd.DataFrame, int]:
    candidates = filter_candidates(frame, query, relative_window=relative_window)
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


def run_demo() -> None:
    frame, source_name = load_dataset()
    numeric_frame = prepare_numeric_frame(frame)
    normalized_frame = min_max_normalize(numeric_frame)
    row_index, query = build_query(numeric_frame)

    print(f"Источник данных: {source_name}")
    print(f"Размер исходного датасета: {frame.shape}")
    print(f"Размер рабочей выборки: {numeric_frame.shape}")
    print("\nМинимум и максимум по признакам:")
    summary = pd.DataFrame({"min": numeric_frame.min(), "max": numeric_frame.max()})
    print(summary.to_string())

    print(f"\nСлучайная запись для запроса: {row_index}")
    print("Неточный запрос:")
    print(query)

    baseline_result = baseline_search(numeric_frame, query, top_k=5)
    proposed_result, candidate_count = proposed_search(
        numeric_frame,
        query,
        top_k=5,
        relative_window=0.20,
        weights={"age": 1.0, "credit_amount": 1.5, "duration_months": 1.0},
        alpha=3.0,
    )

    baseline_mean, baseline_std = measure_time(
        baseline_search,
        numeric_frame,
        query,
        top_k=5,
        repeats=20,
    )
    proposed_mean, proposed_std = measure_time(
        proposed_search,
        numeric_frame,
        query,
        top_k=5,
        relative_window=0.20,
        weights={"age": 1.0, "credit_amount": 1.5, "duration_months": 1.0},
        alpha=3.0,
        repeats=20,
    )

    print("\nTop-5 базового метода:")
    print(baseline_result.to_string(index=False))

    print("\nTop-5 ускоренного метода:")
    if proposed_result.empty:
        print("Нет кандидатов при текущем окне фильтрации.")
    else:
        print(proposed_result.to_string(index=False))

    print("\nСравнение времени:")
    print(
        pd.DataFrame(
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
        ).to_string(index=False)
    )

    print("\nПример нормализованных признаков:")
    print(normalized_frame.head().to_string(index=False))


if __name__ == "__main__":
    run_demo()
