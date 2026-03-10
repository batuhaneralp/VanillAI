import unittest
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.regularized.ridge import RidgeRegression, RidgeClassifier


class TestRidgeRegression(unittest.TestCase):
    """Test cases for Ridge Regression."""

    def setUp(self):
        """Set up test data."""
        # Simple linear relationship: y = 2*x1 + 3*x2 + 1
        self.X = [[1, 1], [2, 2], [3, 3], [4, 4]]
        self.y = [6, 11, 16, 21]  # 2*1 + 3*1 + 1 = 6, etc.

    def test_initialization(self):
        """Test RidgeRegression initialization."""
        model = RidgeRegression(alpha=0.5)
        self.assertEqual(model.alpha, 0.5)
        self.assertIsNone(model.coefficients)
        self.assertEqual(model.intercept, 0.0)

        # Test invalid alpha
        with self.assertRaises(ValueError):
            RidgeRegression(alpha=-1.0)

    def test_fit_and_predict(self):
        """Test basic fitting and prediction."""
        model = RidgeRegression(alpha=0.1)
        model.fit(self.X, self.y)

        # Check that coefficients were learned
        self.assertIsNotNone(model.coefficients)
        self.assertIsInstance(model.intercept, float)

        # Test predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

        # Predictions should be reasonable (not exactly equal due to regularization)
        for pred, true in zip(predictions, self.y):
            self.assertAlmostEqual(pred, true, delta=2.0)  # Allow some tolerance

    def test_regularization_effect(self):
        """Test that regularization shrinks coefficients."""
        # Fit with small alpha (less regularization)
        model_small_alpha = RidgeRegression(alpha=0.01)
        model_small_alpha.fit(self.X, self.y)

        # Fit with large alpha (more regularization)
        model_large_alpha = RidgeRegression(alpha=10.0)
        model_large_alpha.fit(self.X, self.y)

        # Coefficients should be smaller with larger alpha
        for small_coeff, large_coeff in zip(model_small_alpha.coefficients,
                                          model_large_alpha.coefficients):
            self.assertLess(abs(large_coeff), abs(small_coeff))

    def test_predict_unfitted_model(self):
        """Test that predicting without fitting raises error."""
        model = RidgeRegression()
        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_single_feature(self):
        """Test Ridge regression with single feature."""
        X_single = [[1], [2], [3], [4]]
        y_single = [2, 4, 6, 8]  # y = 2*x + 0

        model = RidgeRegression(alpha=0.1)
        model.fit(X_single, y_single)
        predictions = model.predict(X_single)

        # Should approximate the linear relationship
        for i, pred in enumerate(predictions):
            expected = 2 * X_single[i][0]
            self.assertAlmostEqual(pred, expected, delta=1.0)


class TestRidgeClassifier(unittest.TestCase):
    """Test cases for Ridge Classifier."""

    def setUp(self):
        """Set up test data."""
        # Binary classification: separable data
        self.X_binary = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        self.y_binary = [0, 0, 0, 1, 1, 1]

        # Multiclass classification
        self.X_multi = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        self.y_multi = [0, 0, 1, 1, 2, 2]

    def test_initialization(self):
        """Test RidgeClassifier initialization."""
        model = RidgeClassifier(alpha=0.5, learning_rate=0.1, iterations=500)
        self.assertEqual(model.alpha, 0.5)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.iterations, 500)
        self.assertIsNone(model.coefficients)
        self.assertIsNone(model.intercept)

    def test_binary_classification(self):
        """Test binary classification."""
        model = RidgeClassifier(alpha=0.1, iterations=1000)
        model.fit(self.X_binary, self.y_binary)

        # Check that model was fitted
        self.assertIsNotNone(model.coefficients)
        self.assertIsInstance(model.intercept, float)

        # Test predictions
        predictions = model.predict(self.X_binary)
        self.assertEqual(len(predictions), len(self.y_binary))

        # Should achieve reasonable accuracy
        accuracy = model.eval(self.X_binary, self.y_binary)
        self.assertGreater(accuracy, 0.5)  # Better than random

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        model = RidgeClassifier(alpha=0.1, iterations=1000)
        model.fit(self.X_multi, self.y_multi)

        # Check that model was fitted
        self.assertIsNotNone(model.coefficients)
        self.assertIsNotNone(model.intercept)
        self.assertIsNotNone(model.classes_)

        # Coefficients should be list of lists for multiclass
        self.assertIsInstance(model.coefficients, list)
        self.assertIsInstance(model.coefficients[0], list)
        self.assertIsInstance(model.intercept, list)

        # Test predictions
        predictions = model.predict(self.X_multi)
        self.assertEqual(len(predictions), len(self.y_multi))

        # Should achieve reasonable accuracy
        accuracy = model.eval(self.X_multi, self.y_multi)
        self.assertGreater(accuracy, 0.4)  # Better than random (1/3)

    def test_predict_unfitted_model(self):
        """Test that predicting without fitting raises error."""
        model = RidgeClassifier()
        with self.assertRaises(ValueError):
            model.predict(self.X_binary)

    def test_eval_method(self):
        """Test evaluation method."""
        model = RidgeClassifier(alpha=0.1, iterations=1000)
        model.fit(self.X_binary, self.y_binary)

        accuracy = model.eval(self.X_binary, self.y_binary)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_perfect_separation(self):
        """Test with clearly separable data."""
        # Create clearly separable binary data
        X_sep = [[0, 0], [0, 1], [1, 0], [1, 1],  # Class 0: bottom-left
                 [5, 5], [6, 5], [5, 6], [6, 6]]  # Class 0: top-right
        y_sep = [0, 0, 0, 0, 1, 1, 1, 1]

        model = RidgeClassifier(alpha=0.01, iterations=2000, learning_rate=0.1)
        model.fit(X_sep, y_sep)

        # Should achieve high accuracy on training data
        accuracy = model.eval(X_sep, y_sep)
        self.assertGreaterEqual(accuracy, 0.9)  # Allow some tolerance


if __name__ == '__main__':
    unittest.main()