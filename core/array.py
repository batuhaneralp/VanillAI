class Array:
    def __init__(self, data):
        # Ensure all rows have the same length if 2D
        if isinstance(data[0], list):
            row_len = len(data[0])
            assert all(len(row) == row_len for row in data), "Inconsistent row lengths"
        self.data = data

    def shape(self):
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data), )

    def T(self):
        if isinstance(self.data[0], list):
            return Array([list(row) for row in zip(*self.data)])
        raise ValueError("Cannot transpose 1D array")

    def matmul(self, other):
        assert isinstance(other, Array)
        a = self.data
        b = other.data
        # b should be transposed
        b_t = list(zip(*b))
        result = [
            [sum(ai * bi for ai, bi in zip(row_a, col_b)) for col_b in b_t]
            for row_a in a
        ]
        return Array(result)

    def matvec(self, vector):
        assert isinstance(vector, list)
        result = [
            sum(a * v for a, v in zip(row, vector)) for row in self.data
        ]
        return result

    def inverse(self):
        m = [row[:] for row in self.data]
        n = len(m)
        I = [[float(i == j) for j in range(n)] for i in range(n)]

        for i in range(n):
            diag = m[i][i]
            if diag == 0:
                raise ValueError("Singular matrix")
            for j in range(n):
                m[i][j] /= diag
                I[i][j] /= diag
            for k in range(n):
                if k == i:
                    continue
                factor = m[k][i]
                for j in range(n):
                    m[k][j] -= factor * m[i][j]
                    I[k][j] -= factor * I[i][j]
        return Array(I)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"Array({self.data})"
