from __future__ import annotations

import unittest
from unittest.mock import patch

from pandas.core.frame import DataFrame

from src.datasets.credit_approval import CreditApprovalDataset
from src.datasets.credit_card_default import CreditCardDefaultDataset, standardize_columns
from src.datasets.german_credit import GermanCreditDataset


class DatasetLoaderTests(unittest.TestCase):
    def test_german_credit_dataset_loads_local_data(self) -> None:
        frame, source_name = GermanCreditDataset().load_dataset()

        self.assertEqual(source_name, "german.data")
        self.assertEqual(frame.shape, (1000, 21))
        self.assertTrue({"age", "credit_amount", "duration_months", "target"}.issubset(frame.columns))

    def test_credit_card_standardize_columns_handles_variants(self) -> None:
        frame = DataFrame(
            {
                " LIMIT_BAL ": [1000],
                "AGE": [30],
                "PAY_AMT1": [500],
                "default payment next month": [1],
            }
        )

        standardized = standardize_columns(frame)

        self.assertEqual(
            standardized.columns.tolist(),
            ["limit_bal", "age", "pay_amt1", "default_next_month"],
        )

    def test_credit_card_default_dataset_loads_local_xls(self) -> None:
        frame, source_name = CreditCardDefaultDataset().load_dataset()

        self.assertEqual(source_name, "default_of_credit_card_clients.xls")
        self.assertGreater(len(frame), 1000)
        self.assertTrue({"limit_bal", "age", "pay_amt1", "default_next_month"}.issubset(frame.columns))

    def test_credit_card_default_dataset_wraps_import_error(self) -> None:
        dataset = CreditCardDefaultDataset()

        with patch("src.datasets.credit_card_default.pd.read_excel", side_effect=ImportError("xlrd missing")):
            with self.assertRaises(RuntimeError) as context:
                dataset.load_dataset()

        self.assertIn("xlrd", str(context.exception))

    def test_credit_approval_dataset_loads_columns_and_missing_values(self) -> None:
        frame, source_name = CreditApprovalDataset().load_dataset()

        self.assertEqual(source_name, "crx.data")
        self.assertEqual(frame.shape[1], 16)
        self.assertEqual(frame.columns.tolist()[0], "A1")
        self.assertTrue(frame.isna().any().any())


if __name__ == "__main__":
    unittest.main()
