import unittest
import os
from data.csv import Csv
import csv


class TestCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a sample CSV file for testing."""
        cls.test_file = "test_input.csv"
        with open(cls.test_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Header1", "Header2", "Header3"])
            writer.writerows(
                [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9], [10.1, 11.1, 12.1]]
            )

    @classmethod
    def tearDownClass(cls):
        """Remove test files after tests are complete."""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
        if os.path.exists("test_output.csv"):
            os.remove("test_output.csv")

    def test_init(self):
        """Test initialization with a file."""
        csv_obj = Csv(self.test_file)
        self.assertEqual(len(csv_obj), 4)  # Should have 4 rows

    def test_add_row(self):
        """Test adding a valid row and raising an error for invalid row."""
        csv_obj = Csv(self.test_file)
        new_row = [10.1, 11.1, 12.1]
        csv_obj.add_row(new_row)
        self.assertEqual(len(csv_obj), 5)  # Should have 5 rows after adding
        with self.assertRaises(ValueError):
            csv_obj.add_row([1, 2])  # Invalid row

    def test_get_row(self):
        """Test retrieving a specific row."""
        csv_obj = Csv(self.test_file)
        row = csv_obj.get_row(1)
        self.assertEqual(row, [4.4, 5.5, 6.6])  # Row at index 1 should match

    def test_visualize(self):
        """Test visualize function."""
        csv_obj = Csv(self.test_file)
        csv_obj.visualize(row_start=1, row_end=2)  # Should print rows 1 to 2

    def test_set_header(self):
        """Test setting a new header."""
        csv_obj = Csv(self.test_file)
        new_header = ["Col1", "Col2", "Col3"]
        csv_obj.set_header(new_header)
        self.assertEqual(csv_obj.get_header(), new_header)

    def test_get_header(self):
        """Test getting the header."""
        csv_obj = Csv(self.test_file)
        self.assertEqual(csv_obj.get_header(), ["Header1", "Header2", "Header3"])

    def test_head(self):
        """Test getting the first n rows."""
        csv_obj = Csv(self.test_file)
        csv_obj.head(2)  # Should print first 2 rows

    def test_tail(self):
        """Test getting the last n rows."""
        csv_obj = Csv(self.test_file)
        csv_obj.tail(2)  # Should print last 2 rows

    def test_save(self):
        """Test saving to a new CSV file."""
        csv_obj = Csv(self.test_file)
        csv_obj.save("test_output.csv")

        # Verify that the file is created and the content matches
        self.assertTrue(os.path.exists("test_output.csv"))
        with open("test_output.csv", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, ["Header1", "Header2", "Header3"])
        self.assertEqual(
            rows,
            [
                ["1.1", "2.2", "3.3"],
                ["4.4", "5.5", "6.6"],
                ["7.7", "8.8", "9.9"],
                ["10.1", "11.1", "12.1"],
            ],
        )

    def test_train_test_split(self):
        """Test train-test split functionality."""
        csv_obj = Csv(self.test_file)

        # Test with 50% split and a fixed seed for reproducibility
        train_csv, test_csv = csv_obj.train_test_split(test_size=0.5, seed=42)

        # Verify the split proportions (50% for each)
        self.assertEqual(len(train_csv), 2)  # Should have 2 rows in training set
        self.assertEqual(len(test_csv), 2)  # Should have 2 rows in testing set

        # Test with another seed and ensure reproducibility
        train_csv_2, test_csv_2 = csv_obj.train_test_split(test_size=0.5, seed=42)
        self.assertEqual(
            train_csv.rows, train_csv_2.rows
        )  # Should be identical due to the same seed
        self.assertEqual(test_csv.rows, test_csv_2.rows)  # Should be identical due to the same seed

        # Test for 80% train / 20% test split
        train_csv_3, test_csv_3 = csv_obj.train_test_split(test_size=0.2, seed=42)
        self.assertEqual(len(train_csv_3), 3)  # Should have 3 rows in training set
        self.assertEqual(len(test_csv_3), 1)  # Should have 1 row in testing set


if __name__ == "__main__":
    unittest.main()
