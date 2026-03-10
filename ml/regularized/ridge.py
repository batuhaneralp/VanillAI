import math
import random
from core.array import Array
from ml.common.classifier import Classifier


class RidgeRegression:
    """
    Ridge Regression (L2-Regularized Linear Regression).

    Ridge regression adds L2 regularization to ordinary least squares to prevent overfitting
    by penalizing large coefficient values. This is particularly useful when features are
    highly correlated (multicollinear).

    Mathematical Foundation:
    ========================

    1. Ridge Objective Function:
       J(β) = ||y - Xβ||² + α||β||²

       Where:
       - ||y - Xβ||² = sum of squared residuals (SSE)
       - ||β||² = L2 norm of coefficients (squared sum of coefficients)
       - α = regularization strength (λ in some notations)

    2. Matrix Formulation:
       J(β) = (y - Xβ)ᵀ(y - Xβ) + αβᵀβ

    3. Closed-Form Solution:
       β̂ = (XᵀX + αI)^(-1) Xᵀy

       Where:
       - I = identity matrix of size (n_features+1) × (n_features+1)
       - αI ensures the matrix is always invertible (even with multicollinear features)

    4. Bias-Variance Tradeoff:
       - α = 0: Equivalent to OLS (high variance, low bias)
       - α → ∞: Coefficients approach zero (low variance, high bias)
       - Optimal α balances bias and variance

    5. Geometric Interpretation:
       Ridge regression shrinks coefficients toward zero while OLS finds the exact
       minimum in the unregularized loss landscape.

    Algorithm:
    ==========
    1. Add intercept column to feature matrix X
    2. Compute β̂ = (XᵀX + αI)^(-1) Xᵀy
    3. Extract coefficients and intercept from β̂

    Computational Considerations:
    ============================
    - Matrix inversion: O((n_features+1)³)
    - Memory: O(n_samples × n_features)
    - Always numerically stable due to αI term
    - Scales well for moderate feature dimensions

    Hyperparameter Selection:
    ========================
    - α typically chosen via cross-validation
    - Common range: 0.01 to 100 (logarithmic scale)
    - Smaller α for less regularization, larger α for more

    Args:
        alpha (float): Regularization strength. Must be positive. Default 1.0.

    Attributes:
        coefficients (list): Learned regression coefficients (excluding intercept)
        intercept (float): Learned bias term
        alpha (float): Regularization strength used

    Examples:
        # Basic usage
        model = RidgeRegression(alpha=0.1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Effect of alpha on coefficients
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            model = RidgeRegression(alpha=alpha)
            model.fit(X, y)
            print(f"Alpha={alpha}: coefficients={model.coefficients}")
    """

    def __init__(self, alpha=1.0):
        """
        Initialize Ridge Regression model.

        Args:
            alpha (float): Regularization strength. Must be >= 0.
                         alpha=0 is equivalent to OLS. Default 1.0.

        Raises:
            ValueError: If alpha < 0.
        """
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.coefficients = None
        self.intercept = 0.0

    def fit(self, X, y):
        """
        Fit Ridge regression model using regularized normal equation.

        Mathematical Details:
        ====================
        The ridge normal equation adds regularization to prevent overfitting:

        β̂ = (XᵀX + αI)^(-1) Xᵀy

        This ensures numerical stability and reduces coefficient magnitudes.

        Args:
            X (list): Feature matrix of shape (n_samples, n_features)
            y (list): Target values of shape (n_samples,)

        Updates:
            self.coefficients: Regularized regression coefficients
            self.intercept: Regularized bias term

        Examples:
            X_train = [[1, 2], [3, 4], [5, 6]]
            y_train = [3, 7, 11]  # y ≈ x₁ + x₂ + 1
            model = RidgeRegression(alpha=0.1)
            model.fit(X_train, y_train)
            # Coefficients will be slightly shrunk from OLS values
        """
        # Convert to Array objects
        X_arr = Array(X)
        y_arr = Array([[yi] for yi in y])

        n_samples, n_features = X_arr.shape()

        # Create design matrix with intercept
        X_design_data = []
        for i in range(n_samples):
            row = [1.0] + X_arr.data[i]  # Add intercept column
            X_design_data.append(row)
        X_design = Array(X_design_data)

        # Ridge normal equation: β̂ = (XᵀX + αI)^(-1) Xᵀy
        XtX = X_design.T().matmul(X_design)

        # Add regularization: XtX + αI
        n_params = n_features + 1  # +1 for intercept
        for i in range(n_params):
            XtX.data[i][i] += self.alpha

        # Solve for coefficients
        Xty = X_design.T().matmul(y_arr)
        beta = XtX.inverse().matmul(Xty)

        # Extract coefficients and intercept
        self.intercept = beta.data[0][0]
        self.coefficients = [beta.data[i][0] for i in range(1, n_params)]

    def predict(self, X):
        """
        Predict target values for given features.

        Mathematical Definition:
        =======================
        ŷ = X·β + b

        Where predictions are regularized due to ridge coefficient shrinkage.

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


class RidgeClassifier(Classifier):
    """
    Ridge Classifier (L2-Regularized Linear Classification).

    Ridge classifier applies L2 regularization to linear classification, providing
    a regularized version of logistic regression or multiclass linear classification.
    For binary classification, it uses logistic regression with L2 penalty.
    For multiclass, it uses one-vs-rest approach.

    Mathematical Foundation:
    ========================

    1. Binary Classification (Logistic Regression):
       P(y=1|x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))

       Loss: L(w,b) = -(1/m) Σ[yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)] + (α/2)||w||²

       Where:
       - σ = sigmoid function
       - ŷᵢ = predicted probability for class 1
       - α = regularization strength
       - ||w||² = L2 penalty on weights (excluding bias)

    2. Multiclass Classification (One-vs-Rest):
       For K classes, train K binary ridge classifiers
       Each classifier predicts P(class_k | x) vs P(not class_k | x)
       Final prediction: argmax_k P(class_k | x)

    3. Regularized Gradient Updates:
       For binary classification, gradients include L2 penalty:
       ∂L/∂w_j = (1/m) Σ(ŷᵢ - yᵢ)·xᵢⱼ + α·w_j
       ∂L/∂b = (1/m) Σ(ŷᵢ - yᵢ) + α·b  (if bias regularization desired)

    Algorithm:
    ==========
    Binary Classification:
    1. Initialize weights w and bias b
    2. For each iteration:
       a. Compute predictions: ŷ = σ(X·w + b)
       b. Compute gradients with L2 regularization
       c. Update: w -= α·∇w_L, b -= α·∇b_L
    3. Use threshold 0.5 for binary predictions

    Multiclass Classification:
    1. For each class k:
       a. Create binary labels: y_binary = 1 if y==k else 0
       b. Train ridge classifier on binary labels
    2. For prediction: choose class with highest probability

    Computational Considerations:
    ============================
    - Binary: O(iterations × n_samples × n_features)
    - Multiclass: O(K × iterations × n_samples × n_features)
    - Memory: O(n_samples × n_features + n_features × K)
    - L2 regularization prevents overfitting in high dimensions

    Hyperparameter Selection:
    ========================
    - alpha: Regularization strength (0.01 to 100)
    - iterations: Number of gradient descent steps
    - learning_rate: Step size for gradient updates

    Args:
        alpha (float): Regularization strength. Default 1.0.
        learning_rate (float): Learning rate for gradient descent. Default 0.01.
        iterations (int): Maximum iterations for training. Default 1000.

    Attributes:
        coefficients (list or list of lists): Learned coefficients
            - Binary: [w1, w2, ..., wn] (excluding intercept)
            - Multiclass: [[w1_class1, ...], [w1_class2, ...], ...]
        intercept (float or list): Learned intercepts
            - Binary: single float
            - Multiclass: [b_class1, b_class2, ...]
        classes_ (list): Unique class labels (for multiclass)

    Examples:
        # Binary classification
        model = RidgeClassifier(alpha=0.1)
        model.fit(X_train, y_train)  # y_train = [0, 1, 0, 1, ...]
        predictions = model.predict(X_test)

        # Multiclass classification
        model = RidgeClassifier(alpha=1.0)
        model.fit(X_train, y_train)  # y_train = [0, 1, 2, 0, 1, ...]
        predictions = model.predict(X_test)
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, iterations=1000):
        """
        Initialize Ridge Classifier.

        Args:
            alpha (float): L2 regularization strength. Default 1.0.
            learning_rate (float): Learning rate for gradient descent. Default 0.01.
            iterations (int): Maximum training iterations. Default 1000.
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None
        self.intercept = None
        self.classes_ = None

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 0.0 if z < 0 else 1.0

    def _fit_binary(self, X, y):
        """
        Fit binary ridge classifier using regularized gradient descent.

        Args:
            X (list): Feature matrix
            y (list): Binary labels [0, 1]
        """
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0

        # Initialize parameters
        self.coefficients = [0.0] * n_features
        self.intercept = 0.0

        # Gradient descent with L2 regularization
        for _ in range(self.iterations):
            # Compute predictions
            predictions = []
            for x in X:
                z = sum(w * xi for w, xi in zip(self.coefficients, x)) + self.intercept
                predictions.append(self._sigmoid(z))

            # Compute gradients with L2 regularization
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = predictions[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j] + self.alpha * self.coefficients[j]
                db += error + self.alpha * self.intercept

            # Average gradients
            for j in range(n_features):
                dw[j] /= n_samples
            db /= n_samples

            # Update parameters
            for j in range(n_features):
                self.coefficients[j] -= self.learning_rate * dw[j]
            self.intercept -= self.learning_rate * db

    def _fit_multiclass(self, X, y):
        """
        Fit multiclass ridge classifier using one-vs-rest approach.

        Args:
            X (list): Feature matrix
            y (list): Class labels
        """
        self.classes_ = sorted(list(set(y)))
        n_classes = len(self.classes_)

        # Initialize lists for coefficients and intercepts
        self.coefficients = []
        self.intercept = []

        for target_class in self.classes_:
            # Create binary labels for this class
            y_binary = [1 if yi == target_class else 0 for yi in y]

            # Fit binary classifier for this class
            # We need to create a temporary classifier for each class
            temp_classifier = RidgeClassifier(
                alpha=self.alpha,
                learning_rate=self.learning_rate,
                iterations=self.iterations
            )
            temp_classifier._fit_binary(X, y_binary)

            # Store coefficients for this class
            self.coefficients.append(temp_classifier.coefficients[:])  # Copy coefficients
            self.intercept.append(temp_classifier.intercept)

    def fit(self, X, y):
        """
        Fit Ridge classifier.

        Automatically detects binary vs multiclass classification.

        Args:
            X (list): Feature matrix of shape (n_samples, n_features)
            y (list): Target labels of shape (n_samples,)

        Examples:
            # Binary classification
            X = [[1, 2], [3, 4], [5, 6]]
            y = [0, 1, 0]
            model.fit(X, y)

            # Multiclass classification
            y = [0, 1, 2]
            model.fit(X, y)
        """
        unique_classes = set(y)
        if len(unique_classes) == 2:
            # Binary classification
            self._fit_binary(X, y)
        else:
            # Multiclass classification
            self._fit_multiclass(X, y)

    def predict(self, X):
        """
        Predict class labels for given features.

        Args:
            X (list): Feature vectors of shape (n_samples, n_features)

        Returns:
            list: Predicted class labels

        Examples:
            X_test = [[1.0, 2.0], [3.0, 4.0]]
            predictions = model.predict(X_test)  # [0, 1] or [0, 1, 2] etc.
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions")

        if isinstance(self.coefficients[0], list):
            # Multiclass case
            return self._predict_multiclass(X)
        else:
            # Binary case
            return self._predict_binary(X)

    def _predict_binary(self, X):
        """Predict for binary classification."""
        predictions = []
        for x in X:
            z = sum(w * xi for w, xi in zip(self.coefficients, x)) + self.intercept
            prob = self._sigmoid(z)
            predictions.append(1 if prob >= 0.5 else 0)
        return predictions

    def _predict_multiclass(self, X):
        """Predict for multiclass classification."""
        predictions = []
        for x in X:
            # Compute probabilities for each class
            class_probs = []
            for class_idx in range(len(self.classes_)):
                z = sum(w * xi for w, xi in zip(self.coefficients[class_idx], x)) + self.intercept[class_idx]
                prob = self._sigmoid(z)
                class_probs.append(prob)

            # Choose class with highest probability
            predicted_class = self.classes_[class_probs.index(max(class_probs))]
            predictions.append(predicted_class)
        return predictions

    def eval(self, X, y):
        """
        Evaluate model accuracy on given data.

        Args:
            X (list): Feature matrix
            y (list): True labels

        Returns:
            float: Accuracy score (0.0 to 1.0)

        Examples:
            accuracy = model.eval(X_test, y_test)
            print(f"Test accuracy: {accuracy:.3f}")
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y) if len(y) > 0 else 0.0


def _k_fold_split(X, y, k=5, seed=None):
    """
    Split data into k folds for cross-validation.

    Args:
        X (list): Feature matrix
        y (list): Target values
        k (int): Number of folds
        seed (int): Random seed

    Returns:
        list: List of (X_train, X_val, y_train, y_val) tuples
    """
    if seed is not None:
        random.seed(seed)

    n_samples = len(X)
    indices = list(range(n_samples))
    random.shuffle(indices)

    fold_size = n_samples // k
    folds = []

    for i in range(k):
        start_idx = i * fold_size
        if i == k - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * fold_size

        val_indices = indices[start_idx:end_idx]
        train_indices = indices[:start_idx] + indices[end_idx:]

        X_train = [X[j] for j in train_indices]
        y_train = [y[j] for j in train_indices]
        X_val = [X[j] for j in val_indices]
        y_val = [y[j] for j in val_indices]

        folds.append((X_train, X_val, y_train, y_val))

    return folds


class RidgeRegressionCV:
    """
    Ridge Regression with built-in Cross-Validation for alpha selection.

    This class automatically selects the best regularization parameter (alpha)
    using k-fold cross-validation, then fits the final model with the optimal alpha.

    Mathematical Foundation:
    ========================

    1. Cross-Validation Score:
       For each alpha, compute CV score as average validation error across k folds:
       CV(α) = (1/k) Σ MSE_fold_i(α)

       Where MSE_fold_i = (1/n_val) Σ (y_val - ŷ_val)²

    2. Optimal Alpha Selection:
       α̂ = argmin_α CV(α)

    3. Final Model:
       Fit RidgeRegression with α̂ on full training data

    Algorithm:
    ==========
    1. For each candidate alpha:
       a. Split data into k folds
       b. For each fold: train on k-1 folds, validate on remaining fold
       c. Compute average validation MSE across folds
    2. Select alpha with lowest CV MSE
    3. Fit final model on full data with optimal alpha

    Args:
        alphas (list): List of alpha values to try. Default [0.01, 0.1, 1.0, 10.0, 100.0]
        cv (int): Number of cross-validation folds. Default 5.
        seed (int): Random seed for reproducible CV splits. Default None.

    Attributes:
        alpha_ (float): Selected optimal alpha value
        best_score_ (float): Best cross-validation score (MSE)
        cv_results_ (dict): Dictionary with 'alphas' and 'scores' from CV
        coefficients (list): Final model coefficients
        intercept (float): Final model intercept

    Examples:
        # Default alpha search
        model = RidgeRegressionCV()
        model.fit(X_train, y_train)
        print(f"Optimal alpha: {model.alpha_}")

        # Custom alpha range
        import numpy as np
        alphas = np.logspace(-6, 6, 13)
        model = RidgeRegressionCV(alphas=alphas, cv=10)
        model.fit(X_train, y_train)
    """

    def __init__(self, alphas=None, cv=5, seed=None):
        """
        Initialize Ridge Regression with CV.

        Args:
            alphas (list): Alpha values to try. If None, uses [0.01, 0.1, 1.0, 10.0, 100.0]
            cv (int): Number of CV folds. Default 5.
            seed (int): Random seed. Default None.
        """
        if alphas is None:
            self.alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            self.alphas = alphas
        self.cv = cv
        self.seed = seed

        self.alpha_ = None
        self.best_score_ = float('inf')
        self.cv_results_ = None
        self.coefficients = None
        self.intercept = None

    def _cross_validate(self, X, y):
        """
        Perform cross-validation to find optimal alpha.

        Args:
            X (list): Feature matrix
            y (list): Target values

        Returns:
            tuple: (best_alpha, best_score, cv_results)
        """
        cv_scores = []

        for alpha in self.alphas:
            fold_scores = []

            folds = _k_fold_split(X, y, k=self.cv, seed=self.seed)
            for X_train, X_val, y_train, y_val in folds:
                # Train model on fold
                model = RidgeRegression(alpha=alpha)
                model.fit(X_train, y_train)

                # Evaluate on validation fold
                predictions = model.predict(X_val)

                # Compute MSE
                mse = sum((pred - true) ** 2 for pred, true in zip(predictions, y_val)) / len(y_val)
                fold_scores.append(mse)

            # Average MSE across folds
            avg_score = sum(fold_scores) / len(fold_scores)
            cv_scores.append(avg_score)

        # Find best alpha
        best_idx = cv_scores.index(min(cv_scores))
        best_alpha = self.alphas[best_idx]
        best_score = cv_scores[best_idx]

        cv_results = {
            'alphas': self.alphas.copy(),
            'scores': cv_scores
        }

        return best_alpha, best_score, cv_results

    def fit(self, X, y):
        """
        Fit Ridge Regression with CV alpha selection.

        Args:
            X (list): Feature matrix
            y (list): Target values

        Examples:
            X_train = [[1, 2], [3, 4], [5, 6]]
            y_train = [3, 7, 11]
            model = RidgeRegressionCV()
            model.fit(X_train, y_train)
        """
        # Perform cross-validation
        self.alpha_, self.best_score_, self.cv_results_ = self._cross_validate(X, y)

        # Fit final model with optimal alpha
        final_model = RidgeRegression(alpha=self.alpha_)
        final_model.fit(X, y)

        self.coefficients = final_model.coefficients
        self.intercept = final_model.intercept

    def predict(self, X):
        """
        Predict using the fitted model.

        Args:
            X (list): Feature vectors

        Returns:
            list: Predicted values

        Raises:
            ValueError: If model hasn't been fitted
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before making predictions")

        return RidgeRegression.predict(self, X)


class RidgeClassifierCV(RidgeClassifier):
    """
    Ridge Classifier with built-in Cross-Validation for alpha selection.

    This class automatically selects the best regularization parameter (alpha)
    using k-fold cross-validation, then fits the final model with the optimal alpha.

    Mathematical Foundation:
    ========================

    1. Cross-Validation Score:
       For each alpha, compute CV score as average validation accuracy across k folds:
       CV(α) = (1/k) Σ accuracy_fold_i(α)

       Where accuracy_fold_i = (1/n_val) Σ I(ŷ_val == y_val)

    2. Optimal Alpha Selection:
       α̂ = argmax_α CV(α)

    3. Final Model:
       Fit RidgeClassifier with α̂ on full training data

    Algorithm:
    ==========
    1. For each candidate alpha:
       a. Split data into k folds
       b. For each fold: train on k-1 folds, validate on remaining fold
       c. Compute average validation accuracy across folds
    2. Select alpha with highest CV accuracy
    3. Fit final model on full data with optimal alpha

    Args:
        alphas (list): List of alpha values to try. Default [0.01, 0.1, 1.0, 10.0, 100.0]
        cv (int): Number of cross-validation folds. Default 5.
        seed (int): Random seed for reproducible CV splits. Default None.
        learning_rate (float): Learning rate for gradient descent. Default 0.01.
        iterations (int): Maximum iterations for training. Default 1000.

    Attributes:
        alpha_ (float): Selected optimal alpha value
        best_score_ (float): Best cross-validation score (accuracy)
        cv_results_ (dict): Dictionary with 'alphas' and 'scores' from CV
        coefficients: Final model coefficients
        intercept: Final model intercept
        classes_: Unique class labels

    Examples:
        # Default alpha search
        model = RidgeClassifierCV()
        model.fit(X_train, y_train)
        print(f"Optimal alpha: {model.alpha_}")

        # Custom alpha range
        import numpy as np
        alphas = np.logspace(-6, 6, 13)
        model = RidgeClassifierCV(alphas=alphas, cv=10)
        model.fit(X_train, y_train)
    """

    def __init__(self, alphas=None, cv=5, seed=None, learning_rate=0.01, iterations=1000):
        """
        Initialize Ridge Classifier with CV.

        Args:
            alphas (list): Alpha values to try. If None, uses [0.01, 0.1, 1.0, 10.0, 100.0]
            cv (int): Number of CV folds. Default 5.
            seed (int): Random seed. Default None.
            learning_rate (float): Learning rate. Default 0.01.
            iterations (int): Max iterations. Default 1000.
        """
        if alphas is None:
            self.alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            self.alphas = alphas
        self.cv = cv
        self.seed = seed

        self.alpha_ = None
        self.best_score_ = 0.0
        self.cv_results_ = None

        # Initialize parent class with default alpha (will be overridden in fit)
        super().__init__(alpha=1.0, learning_rate=learning_rate, iterations=iterations)

    def _cross_validate(self, X, y):
        """
        Perform cross-validation to find optimal alpha.

        Args:
            X (list): Feature matrix
            y (list): Target labels

        Returns:
            tuple: (best_alpha, best_score, cv_results)
        """
        cv_scores = []

        for alpha in self.alphas:
            fold_scores = []

            folds = _k_fold_split(X, y, k=self.cv, seed=self.seed)
            for X_train, X_val, y_train, y_val in folds:
                # Train model on fold
                model = RidgeClassifier(alpha=alpha, learning_rate=self.learning_rate, iterations=self.iterations)
                model.fit(X_train, y_train)

                # Evaluate on validation fold
                accuracy = model.eval(X_val, y_val)
                fold_scores.append(accuracy)

            # Average accuracy across folds
            avg_score = sum(fold_scores) / len(fold_scores)
            cv_scores.append(avg_score)

        # Find best alpha (highest accuracy)
        best_idx = cv_scores.index(max(cv_scores))
        best_alpha = self.alphas[best_idx]
        best_score = cv_scores[best_idx]

        cv_results = {
            'alphas': self.alphas.copy(),
            'scores': cv_scores
        }

        return best_alpha, best_score, cv_results

    def fit(self, X, y):
        """
        Fit Ridge Classifier with CV alpha selection.

        Args:
            X (list): Feature matrix
            y (list): Target labels

        Examples:
            X_train = [[1, 2], [3, 4], [5, 6]]
            y_train = [0, 1, 0]
            model = RidgeClassifierCV()
            model.fit(X_train, y_train)
        """
        # Perform cross-validation
        self.alpha_, self.best_score_, self.cv_results_ = self._cross_validate(X, y)

        # Fit final model with optimal alpha
        self.alpha = self.alpha_  # Set the alpha for the parent class
        super().fit(X, y)