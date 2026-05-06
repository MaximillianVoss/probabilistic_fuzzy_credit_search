from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame

from .base import AnalyticalDataset, DatasetAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "credit_card_default"
LOCAL_CSV_FILE = DATA_DIR / "UCI_Credit_Card.csv"
LOCAL_XLS_FILE = DATA_DIR / "default of credit card clients.xls"
UCI_FILE = DATA_DIR / "default_of_credit_card_clients.xls"
UCI_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)

COLUMN_RENAME_MAP = {
    "LIMIT_BAL": "limit_bal",
    "AGE": "age",
    "PAY_AMT1": "pay_amt1",
    "default payment next month": "default_next_month",
    "default.payment.next.month": "default_next_month",
}


def standardize_columns(frame: DataFrame) -> DataFrame:
    cleaned_names = {column: str(column).strip() for column in frame.columns}
    frame = frame.rename(columns=cleaned_names)
    return frame.rename(columns=COLUMN_RENAME_MAP)


class CreditCardDefaultDataset(AnalyticalDataset):
    dataset_name = "Credit Card Default Dataset"
    feature_columns = ["limit_bal", "age", "pay_amt1"]
    query_perturbation = {
        "limit_bal": 0.95,
        "age": 1.05,
        "pay_amt1": 1.10,
    }
    weights = {"limit_bal": 1.5, "age": 1.0, "pay_amt1": 1.0}
    target_column = "default_next_month"
    minimum_span = 100.0

    def load_dataset(self) -> tuple[DataFrame, str]:
        if LOCAL_CSV_FILE.exists():
            frame = pd.read_csv(LOCAL_CSV_FILE)
            return standardize_columns(frame), LOCAL_CSV_FILE.name

        xls_path = (
            LOCAL_XLS_FILE
            if LOCAL_XLS_FILE.exists()
            else self.download_if_missing(UCI_URL, UCI_FILE)
        )
        try:
            frame = pd.read_excel(xls_path, header=1)
        except ImportError as exc:
            raise RuntimeError(
                "Для чтения XLS нужен xlrd. Установите зависимости командой: pip install -r requirements.txt"
            ) from exc
        return standardize_columns(frame), xls_path.name


def analyze_dataset() -> DatasetAnalysis:
    return CreditCardDefaultDataset().analyze()
