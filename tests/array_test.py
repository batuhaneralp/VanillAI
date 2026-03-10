import unittest
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.array import Array

class TestArray(unittest.TestCase):

    def setUp(self):
        self.a = Array([[1, 2], [3, 4]])
        self.b = Array([[5], [6]])
        self.c = Array([[1, 2, 3], [4, 5, 6]])

    def test_shape(self):
        self.assertEqual(self.a.shape(), (2, 2))
        self.assertEqual(self.c.shape(), (2, 3))

    def test_transpose(self):
        t = self.c.T()
        self.assertEqual(t.data, [[1, 4], [2, 5], [3, 6]])

    def test_matmul(self):
        result = self.a.matmul(self.b)
        self.assertEqual(result.data, [[17], [39]])

    def test_matvec(self):
        result = self.a.matvec([1, 1])
        self.assertEqual(result, [3, 7])

    def test_inverse(self):
        inv = self.a.inverse()
        expected = [[-2.0, 1.0], [1.5, -0.5]]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(inv[i][j], expected[i][j], places=6)

if __name__ == '__main__':
    unittest.main()
