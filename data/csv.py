import csv
import random

class Csv:
    def __init__(self, file_path: str = None):
        if file_path:
            self.file_path = file_path
            self.header, self.rows = self._load_csv()
            self._length = len(self.rows)
    
    def __len__(self) -> int:
        return len(self.rows)

    def _load_csv(self, has_header=True):
        with open(self.file_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader) if has_header else None
            rows = [list(map(float, row)) for row in reader]
            return header, rows

    def add_row(self, row) -> None:
        if len(row) != len(self.rows[0]):
            raise ValueError(f"The new row's column count ({len(row)}) than current data's column count ({len(self.rows[0])})")
        self.rows.append(row)

    def get_row(self, index: int) -> list:
        return self.rows[index]

    def visualize(self,row_start = None,row_end = None) -> None:
        if self.header:
            print(f"{' | '.join(self.header)}")
            print("-" * (len(self.header) * 4))  # Simple separator based on header length
            
        for row in self.rows[row_start:row_end]:
            print(f"{' | '.join(map(str, row))}")

    def set_header(self, header: list) -> None:
        self.header = header

    def get_header(self) -> list:
        return self.header
    
    def head(self,n:int = 5) -> None:
        self.visualize(row_end=n)

    def tail(self,n:int = 5) -> None:
        self.visualize(row_end=n)

    def save(self, output_path: str) -> None:
        with open(output_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            if self.header:
                writer.writerow(self.header)  # Write header if it exists
            writer.writerows(self.rows)  # Write all rows
    
    def sample(self, n: int) -> list:
        """Return n random samples from the rows."""
        if n > len(self.rows):
            raise ValueError(f"Cannot sample more than the number of available rows ({len(self.rows)})")
        return random.sample(self.rows, n)
