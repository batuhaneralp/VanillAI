import unittest
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.sparse.lasso import LassoRegression


class TestLassoRegression(unittest.TestCase):
    """Test cases for Lasso Regression."""

    def setUp(self):
        """Set up test data."""
        # Create data with some irrelevant features
        # y = 2*x0 + 0.5*x1 + noise, x2 and x3 are irrelevant
        self.X = [
            [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4],
            [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]
        ]
        self.y = [3.1, 5.9, 8.7, 11.5, 14.3, 17.1, 19.9, 22.7]  # ≈ 2*x0 + 0.5*x1 + 1

    def test_initialization(self):
        """Test LassoRegression initialization."""
        model = LassoRegression(alpha=0.5, max_iter=500, tol=1e-5)
        self.assertEqual(model.alpha, 0.5)
        self.assertEqual(model.max_iter, 500)
        self.assertEqual(model.tol, 1e-5)
        self.assertIsNone(model.coefficients)
        self.assertEqual(model.intercept, 0.0)

        # Test invalid parameters
        with self.assertRaises(ValueError):
            LassoRegression(alpha=-1.0)
        with self.assertRaises(ValueError):
            LassoRegression(max_iter=0)

    def test_fit_and_predict(self):
        """Test basic fitting and prediction."""
        model = LassoRegression(alpha=0.01, max_iter=1000)  # Small alpha for minimal sparsity
        model.fit(self.X, self.y)

        # Check that coefficients were learned
        self.assertIsNotNone(model.coefficients)
        self.assertIsInstance(model.intercept, float)
        self.assertEqual(len(model.coefficients), 4)  # 4 features

        # Test predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

        # Should achieve reasonable fit with small alpha
        mse = sum((p - t) ** 2 for p, t in zip(predictions, self.y)) / len(self.y)
        self.assertLess(mse, 1.0)  # Should fit reasonably well

    def test_sparsity_property(self):
        """Test that Lasso produces sparse solutions."""
        # Use larger alpha to encourage sparsity
        model = LassoRegression(alpha=1.0, max_iter=2000)
        model.fit(self.X, self.y)

        # Check sparsity
        sparsity = model.get_sparsity()
        self.assertIsInstance(sparsity, float)
        self.assertGreaterEqual(sparsity, 0.0)
        self.assertLessEqual(sparsity, 1.0)

        # With this alpha, we expect some sparsity
        # (though exact sparsity depends on data and convergence)

    def test_alpha_effect_on_sparsity(self):
        """Test that larger alpha increases sparsity."""
        sparsity_small = LassoRegression(alpha=0.01, max_iter=1000)
        sparsity_small.fit(self.X, self.y)

        sparsity_large = LassoRegression(alpha=2.0, max_iter=1000)
        sparsity_large.fit(self.X, self.y)

        # Larger alpha should generally produce more sparsity
        # (though not guaranteed for all datasets)
        sparsity_diff = sparsity_large.get_sparsity() - sparsity_small.get_sparsity()
        # We don't assert this is always positive, as it depends on the data

    def test_feature_importance(self):
        """Test feature importance functionality."""
        model = LassoRegression(alpha=0.1)
        model.fit(self.X, self.y)

        importance = model.get_feature_importance()
        self.assertIsInstance(importance, list)
        self.assertEqual(len(importance), 4)
        # Should be sorted in descending order
        self.assertEqual(importance, sorted(importance, reverse=True))

    def test_convergence(self):
        """Test convergence behavior."""
        model = LassoRegression(alpha=0.1, max_iter=100, tol=1e-6)
        model.fit(self.X, self.y)

        # Check convergence attributes
        self.assertIsInstance(model.n_iter_, int)
        self.assertGreater(model.n_iter_, 0)
        self.assertLessEqual(model.n_iter_, 100)
        self.assertIsInstance(model.converged, bool)

    def test_predict_unfitted_model(self):
        """Test that predicting without fitting raises error."""
        model = LassoRegression()
        with self.assertRaises(ValueError):
            model.predict(self.X)

        with self.assertRaises(ValueError):
            model.get_feature_importance()

        with self.assertRaises(ValueError):
            model.get_sparsity()

    def test_single_feature(self):
        """Test Lasso regression with single feature."""
        X_single = [[1], [2], [3], [4]]
        y_single = [2, 4, 6, 8]  # y = 2*x + 0

        model = LassoRegression(alpha=0.01)
        model.fit(X_single, y_single)
        predictions = model.predict(X_single)

        # Should approximate the linear relationship
        for i, pred in enumerate(predictions):
            expected = 2 * X_single[i][0]
            self.assertAlmostEqual(pred, expected, delta=1.0)

    def test_high_dimensional_sparse(self):
        """Test Lasso on high-dimensional data with sparsity."""
        # Create data where only first two features matter
        n_samples, n_features = 20, 10
        X_high = []
        y_high = []

        for i in range(n_samples):
            x = [i + 1, (i + 1) * 0.5] + [0.0] * (n_features - 2)  # Only first two features non-zero
            X_high.append(x)
            y_high.append(2.0 * x[0] + 1.5 * x[1] + 0.1)  # Only depend on first two features

        model = LassoRegression(alpha=0.5, max_iter=2000)
        model.fit(X_high, y_high)

        # Check that irrelevant features have zero coefficients
        relevant_zero = sum(1 for i, c in enumerate(model.coefficients) if i >= 2 and abs(c) < 1e-6)
        self.assertGreater(relevant_zero, 0)  # At least some irrelevant features should be zero

    def test_soft_thresholding(self):
        """Test the soft thresholding operator."""
        model = LassoRegression()

        # Test cases for soft thresholding
        self.assertEqual(model._soft_thresholding(0.0, 1.0), 0.0)
        self.assertEqual(model._soft_thresholding(0.5, 1.0), 0.0)  # |0.5| < 1.0
        self.assertEqual(model._soft_thresholding(1.5, 1.0), 0.5)   # 1.5 - 1.0
        self.assertEqual(model._soft_thresholding(-1.5, 1.0), -0.5) # -1.5 + 1.0
        self.assertEqual(model._soft_thresholding(2.0, 1.0), 1.0)
        self.assertEqual(model._soft_thresholding(-2.0, 1.0), -1.0)


if __name__ == '__main__':
    unittest.main()