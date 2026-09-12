"""
data_loader.py

INPUT stage of the IPO framework: load the raw dataset from disk and do
the minimum cleaning needed before it can be fed to a model.
"""

import pandas as pd

from .config import DATA_PATH, MISSING_CATEGORY_FILL


def load_dataset(path=None) -> pd.DataFrame:
    """Load the orders dataset from its original Excel file.

    Missing values in CouponCode mean "no coupon was used" — that's a
    real, meaningful category, not something to drop or guess at, so
    it's filled with an explicit label instead of a random value.
    """
    path = path or DATA_PATH
    df = pd.read_excel(path)
    df["CouponCode"] = df["CouponCode"].fillna(MISSING_CATEGORY_FILL)
    return df


def dataset_summary(df: pd.DataFrame) -> str:
    """A short, human-readable summary of the loaded dataset."""
    lines = [
        f"Rows: {len(df)}",
        f"Columns: {list(df.columns)}",
        f"Missing values remaining: {int(df.isna().sum().sum())}",
    ]
    return "\n".join(lines)
