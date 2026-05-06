from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame

from .base import AnalyticalDataset, DatasetAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "german_credit"
LOCAL_KAGGLE_FILE = DATA_DIR / "german_credit_data.csv"
UCI_FILE = DATA_DIR / "german.data"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

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


class GermanCreditDataset(AnalyticalDataset):
    dataset_name = "German Credit Dataset"
    feature_columns = ["age", "credit_amount", "duration_months"]
    query_perturbation = {
        "age": 1.05,
        "credit_amount": 0.95,
        "duration_months": 1.10,
    }
    weights = {"age": 1.0, "credit_amount": 1.5, "duration_months": 1.0}
    target_column = "target"

    def load_dataset(self) -> tuple[DataFrame, str]:
        if LOCAL_KAGGLE_FILE.exists():
            frame = pd.read_csv(LOCAL_KAGGLE_FILE).rename(columns=KAGGLE_RENAME_MAP)
            return frame, LOCAL_KAGGLE_FILE.name

        raw_path = self.download_if_missing(UCI_URL, UCI_FILE)
        frame = pd.read_csv(raw_path, sep=r"\s+", header=None, names=UCI_COLUMNS)
        return frame, raw_path.name


def analyze_dataset() -> DatasetAnalysis:
    return GermanCreditDataset().analyze()
