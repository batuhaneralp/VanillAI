import csv
import random


class Csv:
    """
    A lightweight CSV handler using only vanilla Python.

    This class allows you to load, manipulate, visualize, and save CSV data
    without relying on any external libraries beyond the Python standard library.
    """

    def __init__(self, file_path: str = None):
        """
        Initialize the Csv object. Optionally loads a CSV file from the given path.

        Args:
            file_path (str, optional): Path to the CSV file. Defaults to None.
        """
        self.header = None
        self.rows = []

        if file_path:
            self.file_path = file_path
            self.header, self.rows = self._load_csv()
            self._length = len(self.rows)

    def __len__(self) -> int:
        """
        Returns the number of rows in the dataset.

        Returns:
            int: Number of rows.
        """
        return len(self.rows)

    def _load_csv(self, has_header=True):
        """
        Load a CSV file from the specified file path.

        Args:
            has_header (bool): Whether the file has a header row. Defaults to True.

        Returns:
            tuple: A tuple (header, rows) where `header` is a list of column names (or None)
                   and `rows` is a list of lists of floats.

        Raises:
            ValueError: If rows cannot be converted to float.
        """
        with open(self.file_path, newline="") as f:
            reader = csv.reader(f)
            if has_header:
                self.header = next(reader)
            rows = [list(map(float, row)) for row in reader]
            return self.header, rows

    def add_row(self, row) -> None:
        """
        Add a new row to the dataset.

        Args:
            row (list): A list of numeric values.

        Raises:
            ValueError: If the row length does not match existing rows.
        """
        if len(row) != len(self.rows[0]):
            raise ValueError(
                f"The new row's column count ({len(row)}) is different from the current data's column count ({len(self.rows[0])})"
            )
        self.rows.append(row)

    def get_row(self, index: int) -> list:
        """
        Retrieve a row by index.

        Args:
            index (int): The index of the row to retrieve.

        Returns:
            list: The row at the specified index.
        """
        return self.rows[index]

    def visualize(self, row_start=None, row_end=None) -> None:
        """
        Print a portion of the dataset to the console.

        Args:
            row_start (int, optional): Starting row index. Defaults to None.
            row_end (int, optional): Ending row index (exclusive). Defaults to None.
        """
        if self.header:
            print(f"{' | '.join(self.header)}")
            print("-" * (len(self.header) * 4))  # Simple separator based on header length
        for row in self.rows[row_start:row_end]:
            print(f"{' | '.join(map(str, row))}")

    def set_header(self, header: list) -> None:
        """
        Set the header for the dataset.

        Args:
            header (list): A list of column names.
        """
        self.header = header

    def get_header(self) -> list:
        """
        Get the current header.

        Returns:
            list: The header list.
        """
        return self.header

    def head(self, n: int = 5) -> None:
        """
        Display the first `n` rows of the dataset.

        Args:
            n (int): Number of rows to display. Defaults to 5.
        """
        self.visualize(row_end=n)

    def tail(self, n: int = 5) -> None:
        """
        Display the last `n` rows of the dataset.

        Args:
            n (int): Number of rows to display. Defaults to 5.
        """
        self.visualize(row_end=n)

    def save(self, output_path: str) -> None:
        """
        Save the dataset to a CSV file.

        Args:
            output_path (str): Path to save the CSV file.
        """
        with open(output_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            if self.header:
                writer.writerow(self.header)
            writer.writerows(self.rows)

    def sample(self, n: int) -> list:
        """
        Return `n` random rows from the dataset.

        Args:
            n (int): Number of rows to sample.

        Returns:
            list: A list of sampled rows.

        Raises:
            ValueError: If n is greater than the number of available rows.
        """
        if n > len(self.rows):
            raise ValueError(
                f"Cannot sample more than the number of available rows ({len(self.rows)})"
            )
        return random.sample(self.rows, n)

    def train_test_split(self, test_size: float = 0.2, seed: int = None):
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float): Fraction of the dataset to use as test data.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            tuple: A tuple (train_csv, test_csv) containing two Csv objects.

        Raises:
            ValueError: If test_size is not between 0 and 1.
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")

        if seed is not None:
            random.seed(seed)

        random.shuffle(self.rows)

        test_size = int(len(self.rows) * test_size)
        train_rows = self.rows[test_size:]
        test_rows = self.rows[:test_size]

        train_csv = Csv()
        if self.header:
            train_csv.set_header(self.header)
        train_csv.rows = train_rows

        test_csv = Csv()
        if self.header:
            test_csv.set_header(self.header)
        test_csv.rows = test_rows

        return train_csv, test_csv
