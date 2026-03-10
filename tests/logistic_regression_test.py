import unittest
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.linear.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        """Set up simple linearly separable data for testing."""
        # Simple binary classification: y = 0 if x1 + x2 < 2, else y = 1
        self.X = [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [2.0, 2.0],
            [2.5, 2.5],
            [3.0, 3.0]
        ]
        self.y = [0, 0, 0, 1, 1, 1]

    def test_initialization(self):
        """Test model initialization."""
        model = LogisticRegression(learning_rate=0.01, iterations=1000)
        self.assertEqual(model.learning_rate, 0.01)
        self.assertEqual(model.iterations, 1000)
        self.assertIsNone(model.coefficients)
        self.assertEqual(model.intercept, 0.0)

    def test_fit_and_predict(self):
        """Test that fit and predict work without errors."""
        model = LogisticRegression(learning_rate=0.1, iterations=1000)
        model.fit(self.X, self.y)
        
        # Check that coefficients were learned
        self.assertIsNotNone(model.coefficients)
        self.assertEqual(len(model.coefficients), 2)
        
        # Predict on training data
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        
        # All predictions should be 0 or 1
        for pred in predictions:
            self.assertIn(pred, [0, 1])

    def test_accuracy_on_training_data(self):
        """Test that model achieves reasonable accuracy on training data."""
        model = LogisticRegression(learning_rate=0.1, iterations=5000)
        model.fit(self.X, self.y)
        
        accuracy = model.eval(self.X, self.y)
        # Should achieve at least 50% accuracy (baseline) on simple separable data
        self.assertGreaterEqual(accuracy, 0.5)
        # Ideally should be > 80% on this simple data
        self.assertGreater(accuracy, 0.6)

    def test_sigmoid_function(self):
        """Test sigmoid function values."""
        model = LogisticRegression()
        
        # Test sigmoid(0) = 0.5
        self.assertAlmostEqual(model._sigmoid(0), 0.5, places=6)
        
        # Test sigmoid is in (0, 1)
        self.assertGreater(model._sigmoid(-10), 0)
        self.assertLess(model._sigmoid(-10), 1)
        self.assertGreater(model._sigmoid(10), 0)
        self.assertLess(model._sigmoid(10), 1)
        
        # Test sigmoid monotonicity
        self.assertGreater(model._sigmoid(1), model._sigmoid(-1))

    def test_predict_proba(self):
        """Test probability predictions."""
        model = LogisticRegression(learning_rate=0.1, iterations=1000)
        model.fit(self.X, self.y)
        
        probas = model._predict_proba(self.X)
        
        # Check that probabilities are in [0, 1]
        for p in probas:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)
        
        # Check that we have the right number of predictions
        self.assertEqual(len(probas), len(self.X))

    def test_different_learning_rates(self):
        """Test models with different learning rates."""
        X_test = [[2.0, 2.0], [0.0, 0.0]]
        
        model_slow = LogisticRegression(learning_rate=0.001, iterations=1000)
        model_slow.fit(self.X, self.y)
        preds_slow = model_slow.predict(X_test)
        
        model_fast = LogisticRegression(learning_rate=0.1, iterations=1000)
        model_fast.fit(self.X, self.y)
        preds_fast = model_fast.predict(X_test)
        
        # Both should make predictions
        self.assertEqual(len(preds_slow), 2)
        self.assertEqual(len(preds_fast), 2)

    def test_eval_accuracy(self):
        """Test the eval (accuracy) metric."""
        model = LogisticRegression(learning_rate=0.1, iterations=5000)
        model.fit(self.X, self.y)
        
        accuracy = model.eval(self.X, self.y)
        
        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        
        # Should be > 0 (better than random on this data)
        self.assertGreater(accuracy, 0)

    def test_single_feature_classification(self):
        """Test with single feature data."""
        X_single = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        y_single = [0, 0, 0, 1, 1, 1]
        
        model = LogisticRegression(learning_rate=0.1, iterations=1000)
        model.fit(X_single, y_single)
        
        predictions = model.predict(X_single)
        self.assertEqual(len(predictions), 6)
        
        # Check that low values predict 0 and high values predict 1
        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[-1], 1)


if __name__ == '__main__':
    unittest.main()
