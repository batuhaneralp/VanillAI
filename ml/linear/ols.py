from core.array import Array  # Adjust import based on your layout

class OrdinaryLeastSquares:
    def __init__(self):
        self.coefficients = None  # type: Array
        self.intercept = 0.0

    def fit(self, X, y):
        # Convert to Array and append bias (column of 1s)
        X_b = [x + [1] for x in X]
        X_array = Array(X_b)

        # y must be a column vector
        y_array = Array([[val] for val in y])

        # β = (XᵀX)^(-1) Xᵀy
        Xt = X_array.T()
        XtX = Xt.matmul(X_array)
        XtX_inv = XtX.inverse()
        Xt_y = Xt.matmul(y_array)
        beta = XtX_inv.matmul(Xt_y)

        # Split coefficients and intercept
        beta_flat = [row[0] for row in beta.data]
        self.coefficients = beta_flat[:-1]
        self.intercept = beta_flat[-1]

    def predict(self, X):
        return [
            sum(w * xi for w, xi in zip(self.coefficients, x)) + self.intercept
            for x in X
        ]
