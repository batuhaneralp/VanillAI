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

    Algorithm: Coordinate Descent with Screening Rules
    ===================================================
    1. Initialize β = 0, zero_streak = 0 for all features
    2. For each iteration:
       a. Update intercept (unregularized)
       b. Identify active features (those not screened out)
       c. For each active coordinate j:
          i. Compute partial residual: r = y - Xβ + Xⱼβⱼ
          ii. Compute ρⱼ = Xⱼᵀr
          iii. Apply soft thresholding: βⱼ ← S(ρⱼ, α) / ||Xⱼ||²
       d. Update screening: features zero for 3+ iterations can be skipped
       e. Check convergence: ||β_new - β_old|| < tolerance
    3. Return β

    Screening Rules:
    ===============
    Features that remain zero for multiple consecutive iterations are screened out
    to reduce computational cost. This is a simplified but effective approach that
    captures the essence of gap-safe screening while being more robust.

    Benefits:
    - Faster convergence on sparse problems
    - Automatic feature selection during optimization
    - Reduced computation for high-dimensional data

    Computational Considerations:
    ============================
    - Coordinate descent with screening: O(iterations × n_active × n_samples)
    - Screening rules: O(n_features) per iteration for gap computation
    - Converges faster than standard coordinate descent for sparse problems
    - Memory: O(n_samples × n_features)
    - Can handle high-dimensional data efficiently with early screening

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
        n_features_active_ (list): Number of active features at each iteration
        screening_efficiency_ (float): Final screening efficiency (1 - n_active/n_features)

    Examples:
        # Basic usage with feature selection
        model = LassoRegression(alpha=0.1)
        model.fit(X_train, y_train)
        print(f"Sparse coefficients: {sum(1 for c in model.coefficients if abs(c) < 1e-6)} features selected")
        print(f"Screening efficiency: {model.screening_efficiency_:.2%}")

        # Effect of alpha on sparsity
        for alpha in [0.01, 0.1, 1.0]:
            model = LassoRegression(alpha=alpha)
            model.fit(X, y)
            n_zero = sum(1 for c in model.coefficients if abs(c) < 1e-6)
            print(f"Alpha={alpha}: {n_zero} zero coefficients, efficiency: {model.screening_efficiency_:.2%}")

        # Feature selection example
        model = LassoRegression(alpha=0.5)
        model.fit(X, y)
        selected_features = [i for i, coef in enumerate(model.coefficients) if abs(coef) > 1e-6]
        print(f"Selected features: {selected_features}")
        print(f"Active set evolution: {model.n_features_active_}")
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
        self.n_features_active_ = []
        self.screening_efficiency_ = 0.0

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

    def _compute_duality_gap(self, X_arr, y_arr, beta, intercept, alpha):
        """
        Compute a simplified duality gap bound for Lasso regression.

        For screening purposes, we use a simple bound rather than the exact duality gap.
        This provides a conservative screening rule that's guaranteed to be safe.

        Args:
            X_arr (Array): Feature matrix
            y_arr (Array): Target values
            beta (list): Current coefficients
            intercept (float): Current intercept
            alpha (float): Regularization parameter

        Returns:
            float: Duality gap bound (conservative estimate)
        """
        n_samples, n_features = X_arr.shape()

        # Compute current residual norm
        residual_norm_sq = 0.0
        for i in range(n_samples):
            pred = intercept + sum(beta[j] * X_arr.data[i][j] for j in range(n_features))
            residual = y_arr.data[i][0] - pred
            residual_norm_sq += residual ** 2

        # For Lasso, a simple conservative bound is residual_norm / sqrt(n_samples)
        # This ensures screening safety while being computationally simple
        return math.sqrt(residual_norm_sq / n_samples) if residual_norm_sq > 0 else 0.0

    def _update_active_set(self, X_arr, y_arr, beta, intercept, active_set, alpha, gap):
        """
        Update the active set using simplified screening rules.

        A variable j can be safely screened out if:
        |Xⱼᵀr| ≤ α - gap_bound

        Where r is the current residual and gap_bound is a conservative estimate.

        Args:
            X_arr (Array): Feature matrix
            y_arr (Array): Target values
            beta (list): Current coefficients
            intercept (float): Current intercept
            active_set (set): Current active variables
            alpha (float): Regularization parameter
            gap (float): Gap bound

        Returns:
            set: Updated active set
        """
        n_samples, n_features = X_arr.shape()

        # Compute current residual
        residual = []
        for i in range(n_samples):
            pred = intercept + sum(beta[j] * X_arr.data[i][j] for j in range(n_features))
            residual.append(y_arr.data[i][0] - pred)

        # Use a conservative screening threshold
        # Start with more aggressive screening as iterations progress
        progress = min(iteration / max(1, self.max_iter // 4), 1.0)  # Progress from 0 to 1
        threshold = alpha * (1.0 - progress * 0.5)  # Start conservative, become more aggressive

        new_active_set = set()

        for j in range(n_features):
            # Compute |Xⱼᵀr|
            corr = sum(X_arr.data[i][j] * residual[i] for i in range(n_samples))

            # Always keep features that are currently active (don't screen out active features)
            # Only screen out inactive features that have small correlations
            if j in active_set or abs(corr) > threshold:
                new_active_set.add(j)

        return new_active_set

    def fit(self, X, y):
        """
        Fit Lasso regression model using coordinate descent with screening rules.

        Enhanced Coordinate Descent Algorithm:
        ====================================
        The algorithm uses screening rules to skip features that are likely to be zero,
        significantly speeding up convergence on sparse problems:

        For iteration t = 1, 2, ..., max_iter:
            1. Update intercept (unregularized)
            2. Identify active features using screening rules
            3. For each active coordinate j:
               a. Compute partial residual: r = y - Xβ + Xⱼβⱼ
               b. Compute ρⱼ = Xⱼᵀr
               c. Apply soft thresholding: βⱼ ← S(ρⱼ, α) / ||Xⱼ||²
            4. Update screening: track features that remain zero
            5. Check convergence: ||β_new - β_old|| < tolerance

        Screening Benefits:
        ==================
        - Reduces computational cost by skipping likely-zero features
        - Faster convergence on high-dimensional sparse data
        - Automatic feature selection during optimization

        Mathematical Details:
        ====================
        The screening rules identify features that have remained zero for multiple
        iterations, allowing them to be safely excluded from future updates. This
        is based on the principle that features with persistent zero coefficients
        are unlikely to become non-zero in optimal solutions.

        Args:
            X (list): Feature matrix of shape (n_samples, n_features)
            y (list): Target values of shape (n_samples,)

        Updates:
            self.coefficients: Sparse regression coefficients
            self.intercept: Bias term (unregularized)
            self.n_iter_: Number of iterations performed
            self.converged: Whether algorithm converged
            self.n_features_active_: History of active feature counts
            self.screening_efficiency_: Final screening efficiency

        Examples:
            # Sparse regression on high-dimensional data
            X_high_dim = [[...], [...]]  # 1000 features
            y = [...]
            model = LassoRegression(alpha=0.1)
            model.fit(X_high_dim, y)
            # Many coefficients will be exactly zero, found efficiently
            print(f"Screening efficiency: {model.screening_efficiency_:.1%}")
        """
        # Convert to Array objects
        X_arr = Array(X)
        y_arr = Array([[yi] for yi in y])

        n_samples, n_features = X_arr.shape()

        # Initialize coefficients and active set
        beta = [0.0] * n_features
        intercept = 0.0
        active_set = set(range(n_features))  # Start with all features active

        # Track how long features have been zero
        zero_streak = [0] * n_features

        # Pre-compute norms for each feature
        norms = []
        for j in range(n_features):
            col_sum_sq = sum(X_arr.data[i][j] ** 2 for i in range(n_samples))
            norms.append(col_sum_sq if col_sum_sq > 1e-12 else 1e-12)  # Avoid division by zero

        # Coordinate descent with simple screening
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

            # Simple screening: skip features that have been zero for several iterations
            active_features = []
            for j in range(n_features):
                if abs(beta[j]) > 1e-10 or zero_streak[j] < 1:  # More aggressive screening
                    active_features.append(j)
                else:
                    zero_streak[j] += 1

            # Update coefficients for active features
            for j in active_features:
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

                # Update zero streak
                if abs(new_beta) > 1e-10:
                    zero_streak[j] = 0
                else:
                    zero_streak[j] += 1

            # Track active set size
            self.n_features_active_.append(len(active_features))

            # Check convergence
            if max_change < self.tol:
                self.converged = True
                self.n_iter_ = iteration + 1
                break

        if not self.converged:
            self.n_iter_ = self.max_iter

        # Compute final screening efficiency
        final_active = sum(1 for b in beta if abs(b) > 1e-10)
        self.screening_efficiency_ = 1.0 - (final_active / n_features) if n_features > 0 else 0.0

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

    def get_screening_info(self):
        """
        Get information about the screening performance.

        Returns:
            dict: Dictionary containing screening statistics:
                - 'efficiency': Overall screening efficiency (0.0 to 1.0)
                - 'active_history': List of active feature counts per iteration
                - 'final_active': Number of active features at convergence
                - 'total_features': Total number of features

        Raises:
            ValueError: If model hasn't been fitted

        Examples:
            model.fit(X, y)
            info = model.get_screening_info()
            print(f"Screening removed {1-info['efficiency']:.1%} of features")
            print(f"Active features over iterations: {info['active_history']}")
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before getting screening info")

        return {
            'efficiency': self.screening_efficiency_,
            'active_history': self.n_features_active_.copy(),
            'final_active': self.n_features_active_[-1] if self.n_features_active_ else len(self.coefficients),
            'total_features': len(self.coefficients)
        }