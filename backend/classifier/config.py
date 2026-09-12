"""
config.py

Central settings for the classification pipeline.
"""

from pathlib import Path

# Dataset kept under its ORIGINAL filename, as given — not renamed.
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Dataset_for_Data_Analytics.xlsx"

TARGET_COLUMN = "OrderStatus"

NUMERIC_FEATURES = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
CATEGORICAL_FEATURES = ["Product", "PaymentMethod", "CouponCode", "ReferralSource"]

MISSING_CATEGORY_FILL = "NoCoupon"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_NEIGHBORS = 5

MODEL_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "model.joblib"
REPORT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = Path(__file__).resolve().parents[1] / "outputs" / "confusion_matrix.png"
