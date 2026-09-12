"""
Unit tests for the Project 2 classification pipeline.

Run from backend/:
    python -m unittest discover -s tests -v
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classifier.data_loader import load_dataset, dataset_summary
from classifier.preprocessing import split_features_and_target, build_preprocessor
from classifier.model import build_model_pipeline, split_train_test, evaluate_model
from classifier.config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.df = load_dataset()

    def test_loads_expected_row_count(self):
        self.assertEqual(len(self.df), 1200)

    def test_no_missing_values_after_cleaning(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_summary_mentions_row_count(self):
        self.assertIn("1200", dataset_summary(self.df))


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = load_dataset()

    def test_split_features_and_target_shapes(self):
        X, y = split_features_and_target(self.df, TARGET_COLUMN)
        self.assertEqual(len(X), len(y))
        self.assertEqual(list(X.columns), NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def test_preprocessor_builds_without_error(self):
        preprocessor = build_preprocessor()
        X, _ = split_features_and_target(self.df, TARGET_COLUMN)
        transformed = preprocessor.fit_transform(X)
        self.assertEqual(transformed.shape[0], len(X))


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset()
        cls.X, cls.y = split_features_and_target(cls.df, TARGET_COLUMN)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_train_test(cls.X, cls.y)
        cls.model = build_model_pipeline()
        cls.model.fit(cls.X_train, cls.y_train)

    def test_train_test_split_sizes(self):
        total = len(self.X_train) + len(self.X_test)
        self.assertEqual(total, len(self.X))
        self.assertAlmostEqual(len(self.X_test) / total, 0.2, delta=0.02)

    def test_model_predicts_known_classes(self):
        predictions = self.model.predict(self.X_test)
        known_classes = set(self.y_train.unique())
        self.assertTrue(set(predictions).issubset(known_classes))

    def test_evaluate_model_returns_expected_keys(self):
        results = evaluate_model(self.model, self.X_test, self.y_test)
        for key in ["accuracy", "f1_weighted", "confusion_matrix", "labels"]:
            self.assertIn(key, results)

    def test_accuracy_is_a_valid_probability(self):
        results = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)

    def test_confusion_matrix_shape_matches_class_count(self):
        results = evaluate_model(self.model, self.X_test, self.y_test)
        n_classes = len(results["labels"])
        self.assertEqual(results["confusion_matrix"].shape, (n_classes, n_classes))


if __name__ == "__main__":
    unittest.main()
