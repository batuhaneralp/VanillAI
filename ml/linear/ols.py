from core.array import Array  # Adjust import based on your layout

class OrdinaryLeastSquares:
    """
    Ordinary Least Squares (OLS) Linear Regression.
    
    OLS finds the best-fitting linear relationship between features and target variable
    by minimizing the sum of squared residuals using matrix algebra.
    
    Mathematical Foundation:
    ========================
    
    1. Linear Model:
       ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w·x + b
       
       Where:
       - ŷ = predicted value
       - x = feature vector
       - w = coefficient vector (slope parameters)
       - b = intercept (bias term)
    
    2. Loss Function (Sum of Squared Errors):
       SSE = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - (w·xᵢ + b))²
       
       Where:
       - yᵢ = true value for sample i
       - ŷᵢ = predicted value for sample i
       - Σ = sum over all training samples
    
    3. Matrix Formulation:
       Let X be the design matrix (n_samples × n_features+1):
       X = [x₁¹, x₁², ..., x₁ⁿ, 1;
            x₂¹, x₂², ..., x₂ⁿ, 1;
            ...
            xₘ¹, xₘ², ..., xₘⁿ, 1]
       
       Let y be the target vector (n_samples × 1):
       y = [y₁; y₂; ...; yₘ]
       
       Let β be the parameter vector (n_features+1 × 1):
       β = [w₁; w₂; ...; wₙ; b]
       
       Then: ŷ = Xβ
    
    4. Normal Equation (Closed-Form Solution):
       β̂ = (XᵀX)^(-1) Xᵀy
       
       Where:
       - Xᵀ = transpose of design matrix
       - (XᵀX)^(-1) = inverse of XᵀX (Gram matrix)
       - β̂ = optimal parameter vector
    
    5. Non-Negative Least Squares (NNLS):
       When positive=True, solve: min ||Xβ - y||² subject to β ≥ 0
       
       This uses coordinate descent algorithm instead of normal equation.
       Useful when coefficients represent physical quantities (prices, counts, etc.)
    
    6. Assumptions:
       - Linearity: Relationship between X and y is linear
       - Independence: Observations are independent
       - Homoscedasticity: Constant variance of residuals
       - No multicollinearity: Features are not perfectly correlated
       - Normality: Residuals are normally distributed (for inference)
    
    Algorithm:
    ==========
    If positive=False (default):
        1. Augment feature matrix X with bias column (column of 1s)
        2. Compute XᵀX (Gram matrix)
        3. Compute inverse of XᵀX
        4. Compute Xᵀy
        5. Solve: β̂ = (XᵀX)^(-1) Xᵀy
        6. Split β̂ into coefficients and intercept
    
    If positive=True:
        1. Use coordinate descent to solve NNLS
        2. Iteratively optimize each coefficient while keeping others fixed
        3. Project coefficients to non-negative orthant
    
    Advantages:
    - Closed-form solution (no iteration needed)
    - Deterministic results
    - Computationally efficient for small datasets
    
    Limitations:
    - Requires matrix inversion (fails if XᵀX is singular)
    - O(n³) complexity for matrix inversion
    - Sensitive to multicollinearity
    - No regularization (prone to overfitting)
    """

    def __init__(self, positive=False):
        """
        Initialize the OLS regression model.
        
        Args:
            positive (bool): If True, constrain coefficients to be non-negative.
                           Uses Non-Negative Least Squares (NNLS) algorithm.
                           Defaults to False (unconstrained OLS).
        """
        self.positive = positive
        self.coefficients = None
        self.intercept = 0.0
    
    def fit(self, X, y):
        """
        Fit the linear regression model using the Normal Equation or NNLS.
        
        Mathematical Algorithm:
        =======================
        
        If positive=False (Unconstrained OLS):
        -------------------------------------
        1. Design Matrix Construction:
           X_aug = [X, 1]  (append column of 1s for intercept)
           
           Example for 2 features, 3 samples:
           X = [[x₁¹, x₁²],     X_aug = [[x₁¹, x₁², 1],
                [x₂¹, x₂²],   →          [x₂¹, x₂², 1],
                [x₃¹, x₃²]]               [x₃¹, x₃², 1]]
        
        2. Target Vector:
           y_vec = [[y₁], [y₂], ..., [yₘ]]ᵀ  (column vector)
        
        3. Normal Equation Solution:
           β̂ = (X_augᵀ X_aug)^(-1) X_augᵀ y_vec
           
           Step-by-step:
           a) Xᵀ = transpose(X_aug)                    → (n+1 × m) matrix
           b) XᵀX = Xᵀ × X_aug                        → (n+1 × n+1) Gram matrix
           c) XᵀX_inv = inverse(XᵀX)                  → (n+1 × n+1) matrix
           d) Xᵀy = Xᵀ × y_vec                        → (n+1 × 1) vector
           e) β̂ = XᵀX_inv × Xᵀy                       → (n+1 × 1) parameter vector
        
        4. Parameter Extraction:
           coefficients = β̂[0:n]  (first n elements)
           intercept = β̂[n]       (last element)
        
        If positive=True (Non-Negative Least Squares):
        ---------------------------------------------
        Solve: min ||X_aug β - y||² subject to β ≥ 0
        
        Uses coordinate descent algorithm:
        1. Initialize β = 0
        2. For each coordinate j:
           a) Compute residual r = y - X_aug β (excluding coordinate j)
           b) Solve for β_j: β_j = max(0, X_aug[:,j]ᵀ r / ||X_aug[:,j]||²)
           c) Update β_j and residual
        3. Repeat until convergence
        
        Computational Considerations:
        =============================
        - Matrix inversion requires XᵀX to be full rank (non-singular) for OLS
        - Fails if features are perfectly multicollinear
        - O(m·n²) time for matrix operations (m=samples, n=features)
        - Memory usage: O(m·n) for design matrix
        
        Geometric Interpretation:
        ========================
        The normal equation finds the β that minimizes ||y - Xβ||²
        This is equivalent to projecting y onto the column space of X
        
        Args:
            X (list): Feature vectors of shape (n_samples, n_features).
            y (list): Target values of shape (n_samples,).
            
        Side Effects:
            Updates self.coefficients and self.intercept in-place.
            
        Raises:
            ValueError: If XᵀX is singular (multicollinear features) and positive=False.
            
        Examples:
            X_train = [[1, 2], [3, 4], [5, 6]]
            y_train = [3, 7, 11]  # y = x₁ + x₂ + 1
            model.fit(X_train, y_train)
            # Result: coefficients=[1, 1], intercept=1
        """
        if self.positive:
            self._fit_nnls(X, y)
        else:
            self._fit_ols(X, y)
    
    def _fit_ols(self, X, y):
        """
        Fit OLS using the normal equation: β = (XᵀX)⁻¹Xᵀy
        
        Mathematical Foundation:
        =======================
        The normal equation provides the closed-form solution that minimizes
        the sum of squared residuals: min_β ||y - Xβ||²
        
        Derivation:
        -----------
        L(β) = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)
        ∇_βL = -2Xᵀ(y - Xβ) = 0
        XᵀXβ = Xᵀy
        β = (XᵀX)⁻¹Xᵀy
        
        Computational Details:
        =====================
        - Matrix inversion: O(n³) for n features
        - Memory efficient for small to medium datasets
        - Numerically stable when XᵀX is well-conditioned
        
        Args:
            X (list): Feature matrix of shape (n_samples, n_features)
            y (list): Target vector of shape (n_samples,)
            
        Updates:
            self.coefficients: Regression coefficients (excluding intercept)
            self.intercept: Bias term
        """
        # Convert to Array objects for matrix operations
        X_arr = Array(X)
        # Create y as column vector: [[y1], [y2], ..., [yn]]
        y_arr = Array([[yi] for yi in y])
        
        # Get dimensions
        n_samples, n_features = X_arr.shape()
        
        # Create intercept column (ones)
        intercept_col = Array([[1.0] for _ in range(n_samples)])
        
        # Add intercept column to X
        X_with_intercept_data = []
        for i in range(n_samples):
            row = [1.0] + X_arr.data[i]  # Add intercept
            X_with_intercept_data.append(row)
        X_with_intercept = Array(X_with_intercept_data)
        
        # Solve normal equation: β = (XᵀX)⁻¹Xᵀy
        XtX = X_with_intercept.T().matmul(X_with_intercept)
        Xty = X_with_intercept.T().matmul(y_arr)
        
        # Check for singularity
        try:
            beta = XtX.inverse().matmul(Xty)
        except ValueError as e:
            if "singular" in str(e).lower():
                raise ValueError("Features are multicollinear. Cannot fit OLS model.")
            raise
        
        # Extract coefficients and intercept
        self.intercept = beta.data[0][0]
        self.coefficients = [beta.data[i][0] for i in range(1, len(beta.data))]    
    def predict(self, X):
        """
        Predict target values for given feature vectors.
        
        Mathematical Definition:
        =======================
        ŷ = X·β + b
        
        Where:
        - X = feature matrix of shape (n_samples, n_features)
        - β = learned coefficients vector of shape (n_features,)
        - b = learned intercept (bias term)
        - ŷ = predicted values of shape (n_samples,)
        
        Args:
            X (list): Feature vectors of shape (n_samples, n_features).
            
        Returns:
            list: Predicted target values of shape (n_samples,).
            
        Raises:
            ValueError: If model has not been fitted yet.
            
        Examples:
            X_test = [[1.0, 2.0], [3.0, 4.0]]
            predictions = model.predict(X_test)  # [some_value1, some_value2]
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        predictions = []
        for x in X:
            # Linear combination: sum(w_i * x_i) + b
            pred = sum(w * xi for w, xi in zip(self.coefficients, x)) + self.intercept
            predictions.append(pred)
        
        return predictions    
    def _fit_nnls(self, X, y):
        """
        Fit Non-Negative Least Squares using coordinate descent.
        
        Mathematical Foundation:
        =======================
        Solves: min_β ||y - Xβ||² subject to β ≥ 0
        
        Algorithm: Coordinate Descent
        =============================
        1. Initialize β = 0
        2. For each coordinate j:
           a. Compute partial residual: r = y - Xβ (excluding j)
           b. Solve for βⱼ: βⱼ = max(0, Xⱼᵀr / ||Xⱼ||²)
           c. Update βⱼ
        3. Repeat until convergence
        
        Convergence Criteria:
        ====================
        - Relative change in objective < tolerance
        - Maximum iterations reached
        
        Computational Complexity:
        ========================
        - O(n·m·max_iter) where n=features, m=samples
        - Memory: O(m·n) for design matrix
        
        Args:
            X (list): Feature matrix of shape (n_samples, n_features)
            y (list): Target vector of shape (n_samples,)
            
        Updates:
            self.coefficients: Non-negative regression coefficients
            self.intercept: Bias term (handled separately)
        """
        # Convert to Array objects
        X_arr = Array(X)
        y_arr = Array([[yi] for yi in y])
        
        n_samples, n_features = X_arr.shape()
        tolerance = 1e-6
        max_iter = 1000
        
        # Initialize coefficients (non-negative) as list of lists
        beta = [[0.0] for _ in range(n_features + 1)]  # +1 for intercept
        
        # Create X with intercept column
        X_with_intercept_data = []
        for i in range(n_samples):
            row = [1.0] + X_arr.data[i]  # Add intercept
            X_with_intercept_data.append(row)
        X_with_intercept = Array(X_with_intercept_data)
        
        # Coordinate descent
        for _ in range(max_iter):
            beta_old = [row[:] for row in beta]
            
            for j in range(n_features + 1):  # Include intercept
                # Compute residual excluding feature j
                residual_data = []
                for i in range(n_samples):
                    pred = 0.0
                    for k in range(n_features + 1):
                        if k != j:
                            pred += X_with_intercept.data[i][k] * beta[k][0]
                    residual_data.append([y_arr.data[i][0] - pred])
                residual = Array(residual_data)
                
                # Solve for coordinate j
                xj_data = [[X_with_intercept.data[i][j]] for i in range(n_samples)]
                xj = Array(xj_data)
                
                # Compute Xj^T * residual and ||Xj||²
                xj_t_residual = 0.0
                xj_norm_sq = 0.0
                for i in range(n_samples):
                    xj_t_residual += xj.data[i][0] * residual.data[i][0]
                    xj_norm_sq += xj.data[i][0] ** 2
                
                if xj_norm_sq > 1e-12:  # Avoid division by zero
                    beta_j_new = max(0.0, xj_t_residual / xj_norm_sq)
                else:
                    beta_j_new = 0.0
                
                beta[j][0] = beta_j_new
            
            # Check convergence (simple check)
            converged = True
            for i in range(len(beta)):
                if abs(beta[i][0] - beta_old[i][0]) > tolerance:
                    converged = False
                    break
            if converged:
                break
        
        # Extract coefficients and intercept
        self.intercept = beta[0][0]
        self.coefficients = [beta[i][0] for i in range(1, len(beta))]
