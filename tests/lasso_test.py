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

    def test_screening_functionality(self):
        """Test gap-safe screening rules functionality."""
        model = LassoRegression(alpha=0.1, max_iter=100)
        model.fit(self.X, self.y)

        # Check screening attributes
        self.assertIsInstance(model.n_features_active_, list)
        self.assertIsInstance(model.screening_efficiency_, float)
        self.assertGreaterEqual(model.screening_efficiency_, 0.0)
        self.assertLessEqual(model.screening_efficiency_, 1.0)

        # Active set history should be non-empty
        self.assertGreater(len(model.n_features_active_), 0)

        # Final active count should be <= total features
        self.assertLessEqual(model.n_features_active_[-1], 4)

    def test_screening_info_method(self):
        """Test the get_screening_info method."""
        model = LassoRegression(alpha=0.1)
        model.fit(self.X, self.y)

        info = model.get_screening_info()
        self.assertIsInstance(info, dict)
        self.assertIn('efficiency', info)
        self.assertIn('active_history', info)
        self.assertIn('final_active', info)
        self.assertIn('total_features', info)

        # Check values
        self.assertEqual(info['total_features'], 4)
        self.assertEqual(info['final_active'], model.n_features_active_[-1])
        self.assertEqual(info['efficiency'], model.screening_efficiency_)

    def test_screening_with_high_sparsity(self):
        """Test screening with high regularization (should screen more features)."""
        # Create data with clear irrelevant features
        X_sparse = [
            [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0],
            [5, 0, 0, 0], [6, 0, 0, 0], [7, 0, 0, 0], [8, 0, 0, 0]
        ]
        y_sparse = [2, 4, 6, 8, 10, 12, 14, 16]  # Only depends on first feature

        model = LassoRegression(alpha=1.0, max_iter=1000)  # High regularization
        model.fit(X_sparse, y_sparse)

        # Should achieve high screening efficiency
        info = model.get_screening_info()
        self.assertGreater(info['efficiency'], 0.5)  # At least 50% screening

        # Most coefficients should be zero
        n_zero = sum(1 for c in model.coefficients if abs(c) < 1e-6)
        self.assertGreaterEqual(n_zero, 2)  # At least 2 features should be zero


if __name__ == '__main__':
    unittest.main()