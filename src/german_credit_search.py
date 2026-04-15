from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.search_analysis import DatasetAnalysis, run_analysis

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
WEIGHTS = {"age": 1.0, "credit_amount": 1.5, "duration_months": 1.0}

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


def analyze_dataset() -> DatasetAnalysis:
    frame, source_name = load_dataset()
    return run_analysis(
        dataset_name="German Credit Dataset",
        frame=frame,
        source_name=source_name,
        feature_columns=FEATURE_COLUMNS,
        query_perturbation=QUERY_PERTURBATION,
        weights=WEIGHTS,
        relative_window=0.20,
        alpha=3.0,
        top_k=5,
        repeats=20,
        target_column="target",
        minimum_span=1.0,
    )


def run_demo() -> None:
    analysis = analyze_dataset()

    print(f"Источник данных: {analysis.source_name}")
    print(f"Размер исходного датасета: {analysis.raw_shape}")
    print(f"Размер рабочей выборки: {analysis.numeric_shape}")
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
