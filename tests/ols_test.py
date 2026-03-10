import unittest
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.linear.ols import OrdinaryLeastSquares

class TestOrdinaryLeastSquares(unittest.TestCase):

    def setUp(self):
        # Linear relationship: y = 2x + 1
        self.X = [[1], [2], [3], [4]]
        self.y = [3, 5, 7, 9]
        self.model = OrdinaryLeastSquares()
        self.model.fit(self.X, self.y)

    def test_coefficients(self):
        self.assertAlmostEqual(self.model.coefficients[0], 2.0, places=6)

    def test_intercept(self):
        self.assertAlmostEqual(self.model.intercept, 1.0, places=6)

    def test_predictions(self):
        preds = self.model.predict([[5], [6]])
        self.assertAlmostEqual(preds[0], 11.0, places=6)
        self.assertAlmostEqual(preds[1], 13.0, places=6)

    def test_fit_on_multivariate(self):
        # y = 3*x1 - 2*x2 + 5
        X = [[1, 1], [2, 1], [3, 2], [4, 2]]
        y = [6, 9, 10, 13]
        model = OrdinaryLeastSquares()
        model.fit(X, y)
        self.assertAlmostEqual(model.coefficients[0], 3.0, places=6)
        self.assertAlmostEqual(model.coefficients[1], -2.0, places=6)
        self.assertAlmostEqual(model.intercept, 5.0, places=6)

if __name__ == "__main__":
    unittest.main()
