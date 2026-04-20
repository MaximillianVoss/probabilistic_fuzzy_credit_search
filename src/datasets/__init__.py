from .base import AnalyticalDataset, DatasetAnalysis
from .credit_approval import CreditApprovalDataset, analyze_dataset as analyze_credit_approval
from .credit_card_default import CreditCardDefaultDataset, analyze_dataset as analyze_credit_card_default
from .german_credit import GermanCreditDataset, analyze_dataset as analyze_german_credit

__all__ = [
    "AnalyticalDataset",
    "DatasetAnalysis",
    "GermanCreditDataset",
    "CreditCardDefaultDataset",
    "CreditApprovalDataset",
    "analyze_german_credit",
    "analyze_credit_card_default",
    "analyze_credit_approval",
]
