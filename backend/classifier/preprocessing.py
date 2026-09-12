"""
preprocessing.py

Builds the preprocessing half of the pipeline: scales numeric columns
and one-hot encodes categorical columns, wrapped in a ColumnTransformer
so it can be chained with the model in a single sklearn Pipeline (the
scaler/encoder are fit ONLY on training data — no leakage into test).
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def split_features_and_target(df, target_column):
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[target_column]
    return X, y
