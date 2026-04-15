from __future__ import annotations

import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "credit_card_default"
LOCAL_CSV_FILE = DATA_DIR / "UCI_Credit_Card.csv"
LOCAL_XLS_FILE = DATA_DIR / "default of credit card clients.xls"
UCI_FILE = DATA_DIR / "default_of_credit_card_clients.xls"
UCI_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)

FEATURE_COLUMNS = ["limit_bal", "age", "pay_amt1"]
QUERY_PERTURBATION = {
    "limit_bal": 0.95,
    "age": 1.05,
    "pay_amt1": 1.10,
}

COLUMN_RENAME_MAP = {
    "LIMIT_BAL": "limit_bal",
    "AGE": "age",
    "PAY_AMT1": "pay_amt1",
    "default payment next month": "default_next_month",
    "default.payment.next.month": "default_next_month",
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


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned_names = {column: str(column).strip() for column in frame.columns}
    frame = frame.rename(columns=cleaned_names)
    return frame.rename(columns=COLUMN_RENAME_MAP)


def load_dataset() -> tuple[pd.DataFrame, str]:
    # Если пользователь положил CSV из Kaggle, берём его. Иначе работаем с официальным XLS.
    if LOCAL_CSV_FILE.exists():
        frame = pd.read_csv(LOCAL_CSV_FILE)
        return standardize_columns(frame), LOCAL_CSV_FILE.name

    xls_path = LOCAL_XLS_FILE if LOCAL_XLS_FILE.exists() else download_if_missing(UCI_URL, UCI_FILE)
    try:
        frame = pd.read_excel(xls_path, header=1)
    except ImportError as exc:
        raise RuntimeError(
            "Для чтения XLS нужен xlrd. Установите зависимости командой: pip install -r requirements.txt"
        ) from exc
    return standardize_columns(frame), xls_path.name


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
    minimum_span: float = 100.0,
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
    if "default_next_month" in frame.columns:
        print("\nРаспределение целевого признака:")
        print(frame["default_next_month"].value_counts(dropna=False).to_string())

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
        weights={"limit_bal": 1.5, "age": 1.0, "pay_amt1": 1.0},
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
        weights={"limit_bal": 1.5, "age": 1.0, "pay_amt1": 1.0},
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
