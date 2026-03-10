import math
from core.array import Array


class LassoRegression:
    """
    Lasso Regression (L1-Regularized Linear Regression).

    Lasso (Least Absolute Shrinkage and Selection Operator) adds L1 regularization to
    ordinary least squares, encouraging sparse solutions where many coefficients become
    exactly zero. This makes Lasso useful for feature selection and interpretable models.

    Mathematical Foundation:
    ========================

    1. Lasso Objective Function:
       J(β) = ||y - Xβ||² + α||β||₁

       Where:
       - ||y - Xβ||² = sum of squared residuals (SSE)
       - ||β||₁ = L1 norm (sum of absolute values of coefficients)
       - α = regularization strength (λ in some notations)

    2. Sparsity Property:
       Unlike Ridge regression (L2), Lasso can produce sparse solutions:
       - Many coefficients become exactly zero
       - Effective feature selection
       - More interpretable models

    3. Soft Thresholding:
       The coordinate descent update uses soft thresholding:
       βⱼ ← S(ρⱼ, α) / ||Xⱼ||²

       Where S(z, γ) = sign(z) · max(|z| - γ, 0) is the soft thresholding operator

       And ρⱼ = Xⱼᵀ(y - Xβ + Xⱼβⱼ) is the partial residual

    4. Geometric Interpretation:
       L1 regularization creates a diamond-shaped constraint region in the
       coefficient space, leading to sparse solutions at the vertices.

    Algorithm: Coordinate Descent
    =============================
    1. Initialize β = 0
    2. For each iteration:
       a. For each coordinate j:
          i. Compute partial residual: r = y - Xβ + Xⱼβⱼ
          ii. Compute ρⱼ = Xⱼᵀr
          iii. Apply soft thresholding: βⱼ ← S(ρⱼ, α) / ||Xⱼ||²
       b. Check convergence
    3. Return β

    Computational Considerations:
    ============================
    - Coordinate descent: O(iterations × n_features × n_samples)
    - Converges faster than gradient descent for sparse problems
    - Memory: O(n_samples × n_features)
    - Can handle high-dimensional data efficiently

    Hyperparameter Selection:
    ========================
    - α typically chosen via cross-validation
    - Common range: 0.01 to 10 (logarithmic scale)
    - Larger α → more sparsity (more coefficients = 0)
    - Smaller α → less regularization, similar to OLS

    Path Algorithms:
    ===============
    For efficient α selection, consider:
    - Compute entire regularization path
    - Warm starts: use solution from αᵢ as starting point for αᵢ₊₁
    - Early stopping when coefficients stabilize

    Args:
        alpha (float): L1 regularization strength. Must be positive. Default 1.0.
        max_iter (int): Maximum coordinate descent iterations. Default 1000.
        tol (float): Convergence tolerance. Default 1e-4.

    Attributes:
        coefficients (list): Learned regression coefficients (excluding intercept)
        intercept (float): Learned bias term
        alpha (float): Regularization strength used
        n_iter_ (int): Number of iterations run
        converged (bool): Whether the algorithm converged

    Examples:
        # Basic usage with feature selection
        model = LassoRegression(alpha=0.1)
        model.fit(X_train, y_train)
        print(f"Sparse coefficients: {sum(1 for c in model.coefficients if abs(c) < 1e-6)} features selected")

        # Effect of alpha on sparsity
        for alpha in [0.01, 0.1, 1.0]:
            model = LassoRegression(alpha=alpha)
            model.fit(X, y)
            n_zero = sum(1 for c in model.coefficients if abs(c) < 1e-6)
            print(f"Alpha={alpha}: {n_zero} zero coefficients")

        # Feature selection example
        model = LassoRegression(alpha=0.5)
        model.fit(X, y)
        selected_features = [i for i, coef in enumerate(model.coefficients) if abs(coef) > 1e-6]
        print(f"Selected features: {selected_features}")
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Initialize Lasso Regression model.

        Args:
            alpha (float): L1 regularization strength. Must be >= 0.
                         alpha=0 is equivalent to OLS. Default 1.0.
            max_iter (int): Maximum coordinate descent iterations. Default 1000.
            tol (float): Convergence tolerance for coordinate descent. Default 1e-4.

        Raises:
            ValueError: If alpha < 0 or max_iter <= 0.
        """
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.intercept = 0.0
        self.n_iter_ = 0
        self.converged = False

    def _soft_thresholding(self, z, gamma):
        """
        Soft thresholding operator for L1 regularization.

        S(z, γ) = sign(z) · max(|z| - γ, 0)

        Args:
            z (float): Input value
            gamma (float): Threshold value

        Returns:
            float: Soft-thresholded value
        """
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0

    def fit(self, X, y):
        """
        Fit Lasso regression model using coordinate descent.

        Coordinate Descent Algorithm:
        ============================
        The algorithm iteratively updates each coefficient while holding others fixed:

        For iteration t = 1, 2, ..., max_iter:
            For each coordinate j = 1, 2, ..., n_features:
                1. Compute partial residual: r = y - Xβ + Xⱼβⱼ
                2. Compute correlation: ρⱼ = Xⱼᵀr
                3. Apply soft thresholding: βⱼ ← S(ρⱼ, α) / ||Xⱼ||²

            Check convergence: ||β_new - β_old|| < tolerance

        Mathematical Details:
        ====================
        The coordinate descent update minimizes the Lasso objective for one coordinate
        while keeping others fixed. The soft thresholding operator naturally handles
        the L1 penalty, setting coefficients to zero when appropriate.

        Args:
            X (list): Feature matrix of shape (n_samples, n_features)
            y (list): Target values of shape (n_samples,)

        Updates:
            self.coefficients: Sparse regression coefficients
            self.intercept: Bias term (unregularized)
            self.n_iter_: Number of iterations performed
            self.converged: Whether algorithm converged

        Examples:
            # Sparse regression on high-dimensional data
            X_high_dim = [[...], [...]]  # 1000 features
            y = [...]
            model = LassoRegression(alpha=0.1)
            model.fit(X_high_dim, y)
            # Many coefficients will be exactly zero
        """
        # Convert to Array objects
        X_arr = Array(X)
        y_arr = Array([[yi] for yi in y])

        n_samples, n_features = X_arr.shape()

        # Initialize coefficients (only features, intercept handled separately)
        beta = [0.0] * n_features
        intercept = 0.0

        # Pre-compute norms for each feature
        norms = []
        for j in range(n_features):
            col_sum_sq = sum(X_arr.data[i][j] ** 2 for i in range(n_samples))
            norms.append(col_sum_sq if col_sum_sq > 1e-12 else 1e-12)  # Avoid division by zero

        # Coordinate descent
        for iteration in range(self.max_iter):
            beta_old = beta[:]
            intercept_old = intercept
            max_change = 0.0

            # Update intercept (unregularized)
            residual_sum = 0.0
            for i in range(n_samples):
                pred = intercept + sum(beta[j] * X_arr.data[i][j] for j in range(n_features))
                residual_sum += y_arr.data[i][0] - pred
            new_intercept = residual_sum / n_samples
            intercept = new_intercept
            max_change = max(max_change, abs(intercept - intercept_old))

            # Update each coefficient with L1 regularization
            for j in range(n_features):
                # Compute partial residual: y - intercept - sum_{k≠j} X_k * beta_k
                residual = 0.0
                for i in range(n_samples):
                    pred_other = intercept + sum(beta[k] * X_arr.data[i][k] for k in range(n_features) if k != j)
                    residual += X_arr.data[i][j] * (y_arr.data[i][0] - pred_other)

                # Apply soft thresholding
                new_beta = self._soft_thresholding(residual, self.alpha) / norms[j]
                change = abs(new_beta - beta[j])
                max_change = max(max_change, change)
                beta[j] = new_beta

            # Check convergence
            if max_change < self.tol:
                self.converged = True
                self.n_iter_ = iteration + 1
                break

        if not self.converged:
            self.n_iter_ = self.max_iter

        # Store results
        self.intercept = intercept
        self.coefficients = beta

    def predict(self, X):
        """
        Predict target values for given features.

        Mathematical Definition:
        =======================
        ŷ = X·β + b

        Where predictions use the sparse coefficient vector from Lasso.

        Args:
            X (list): Feature vectors of shape (n_samples, n_features)

        Returns:
            list: Predicted values of shape (n_samples,)

        Raises:
            ValueError: If model hasn't been fitted

        Examples:
            X_test = [[1.0, 2.0], [3.0, 4.0]]
            predictions = model.predict(X_test)
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions")

        predictions = []
        for x in X:
            pred = sum(w * xi for w, xi in zip(self.coefficients, x)) + self.intercept
            predictions.append(pred)

        return predictions

    def get_feature_importance(self):
        """
        Get feature importance based on absolute coefficient values.

        For Lasso, feature importance is determined by the magnitude of coefficients,
        since larger absolute values indicate more important features.

        Returns:
            list: Absolute coefficient values sorted in descending order

        Raises:
            ValueError: If model hasn't been fitted

        Examples:
            model.fit(X, y)
            importance = model.get_feature_importance()
            top_features = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before getting feature importance")

        return sorted([abs(c) for c in self.coefficients], reverse=True)

    def get_sparsity(self):
        """
        Get the sparsity level of the model.

        Sparsity is the fraction of coefficients that are exactly zero.

        Returns:
            float: Sparsity ratio (0.0 to 1.0)

        Raises:
            ValueError: If model hasn't been fitted

        Examples:
            model.fit(X, y)
            sparsity = model.get_sparsity()
            print(f"Model sparsity: {sparsity:.2%}")
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before getting sparsity")

        n_zero = sum(1 for c in self.coefficients if abs(c) < 1e-10)
        return n_zero / len(self.coefficients) if self.coefficients else 0.0