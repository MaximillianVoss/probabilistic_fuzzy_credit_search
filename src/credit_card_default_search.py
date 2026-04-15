from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.search_analysis import DatasetAnalysis, run_analysis

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
WEIGHTS = {"limit_bal": 1.5, "age": 1.0, "pay_amt1": 1.0}

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


def analyze_dataset() -> DatasetAnalysis:
    frame, source_name = load_dataset()
    return run_analysis(
        dataset_name="Credit Card Default Dataset",
        frame=frame,
        source_name=source_name,
        feature_columns=FEATURE_COLUMNS,
        query_perturbation=QUERY_PERTURBATION,
        weights=WEIGHTS,
        relative_window=0.20,
        alpha=3.0,
        top_k=5,
        repeats=20,
        target_column="default_next_month",
        minimum_span=100.0,
    )


def run_demo() -> None:
    analysis = analyze_dataset()

    print(f"Источник данных: {analysis.source_name}")
    print(f"Размер исходного датасета: {analysis.raw_shape}")
    print(f"Размер рабочей выборки: {analysis.numeric_shape}")
    if analysis.target_distribution is not None:
        print("\nРаспределение целевого признака:")
        print(analysis.target_distribution.to_string(index=False))

    print("\nМинимум и максимум по признакам:")
    print(analysis.feature_summary.to_string(index=False))

    print(f"\nСлучайная запись для запроса: {analysis.row_index}")
    print("Неточный запрос:")
    print(analysis.query)

    print("\nTop-5 базового метода:")
    print(analysis.baseline_result.to_string(index=False))

    print("\nTop-5 ускоренного метода:")
    if analysis.proposed_result.empty:
        print("Нет кандидатов при текущем окне фильтрации.")
    else:
        print(analysis.proposed_result.to_string(index=False))

    print("\nСравнение времени:")
    print(analysis.time_comparison.to_string(index=False))

    print("\nПример нормализованных признаков:")
    print(analysis.normalized_frame.head().to_string(index=False))


if __name__ == "__main__":
    run_demo()
