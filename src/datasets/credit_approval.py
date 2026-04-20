from __future__ import annotations

import csv
from pathlib import Path

from pandas import DataFrame

from .base import AnalyticalDataset, DatasetAnalysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "credit_approval"
LOCAL_DATA_FILE = DATA_DIR / "crx.data"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

COLUMNS = [f"A{index}" for index in range(1, 17)]


class CreditApprovalDataset(AnalyticalDataset):
    dataset_name = "Credit Approval Dataset"
    feature_columns = ["A2", "A3", "A8"]
    query_perturbation = {
        "A2": 1.05,
        "A3": 0.95,
        "A8": 1.10,
    }
    weights = {"A2": 1.0, "A3": 1.2, "A8": 1.0}
    target_column = "A16"

    def load_dataset(self) -> tuple[DataFrame, str]:
        data_path = self.download_if_missing(UCI_URL, LOCAL_DATA_FILE)
        rows: list[list[str | None]] = []
        with data_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append([None if value == "?" else value for value in row])

        frame = DataFrame(rows, columns=COLUMNS)
        return frame, data_path.name


def analyze_dataset() -> DatasetAnalysis:
    return CreditApprovalDataset().analyze()
