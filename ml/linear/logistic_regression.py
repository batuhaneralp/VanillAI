import math
from ml.common.classifier import Classifier


class LogisticRegression(Classifier):
    """
    Logistic Regression classifier for binary classification.
    
    Logistic Regression models the probability of a binary outcome using the logistic function.
    It finds optimal weights that minimize the cross-entropy (log) loss using gradient descent.
    
    Mathematical Foundation:
    ========================
    
    1. Logistic Function (Sigmoid):
       σ(z) = 1 / (1 + e^(-z))
       
       Where z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w·x + b
       
       Output: σ(z) ∈ (0, 1) representing P(y=1|x)
    
    2. Cross-Entropy Loss Function:
       L(w, b) = -(1/m) Σ[yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
       
       Where:
       - m = number of samples
       - yᵢ ∈ {0, 1} = true label
       - ŷᵢ = σ(w·xᵢ + b) = predicted probability
    
    3. Gradient Descent Update Rule:
       w_new = w_old - α·(dL/dw)
       b_new = b_old - α·(dL/db)
       
       Where α = learning_rate
       
       Gradients (derived from chain rule):
       dL/dw_j = (1/m) Σ(ŷᵢ - yᵢ)·xᵢⱼ
       dL/db   = (1/m) Σ(ŷᵢ - yᵢ)
    
    4. Decision Boundary:
       Predict y = 1 if P(y=1|x) ≥ 0.5, else y = 0
       Equivalently: Predict y = 1 if w·x + b ≥ 0, else y = 0
    
    Algorithm:
    ==========
    Iterate for `iterations` steps:
        1. Compute predictions: ŷ = σ(w·X + b)
        2. Compute gradients using training data
        3. Update weights: w -= α·∇w_L, b -= α·∇b_L
    
    The model converges when gradients become small and loss stabilizes.
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent. Defaults to 0.01.
            iterations (int): Number of training iterations. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None
        self.intercept = 0.0
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function (logistic function).
        
        Mathematical Definition:
        ========================
        σ(z) = 1 / (1 + e^(-z))
        
        Properties:
        - Range: (0, 1) - output is always between 0 and 1
        - Monotonic increasing: σ'(z) > 0 for all z
        - Symmetry: σ(-z) = 1 - σ(z)
        - σ(0) = 0.5 (decision boundary)
        - σ(z) ≈ 0 for z << 0 (strongly negative)
        - σ(z) ≈ 1 for z >> 0 (strongly positive)
        
        Derivative (used in backprop):
        σ'(z) = σ(z)·(1 - σ(z))
        
        Args:
            z (float): Input value (logit/log-odds).
            
        Returns:
            float: Sigmoid(z) ∈ (0, 1)
            
        Examples:
            sigmoid(-10) ≈ 0      # Very confident class 0
            sigmoid(0)   = 0.5    # Uncertain
            sigmoid(10)  ≈ 1      # Very confident class 1
        """
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            # Handle overflow for very negative values
            return 0 if z < 0 else 1
    
    def _predict_proba(self, X):
        """
        Compute probability predictions for each sample.
        
        Mathematical Definition:
        ========================
        For each sample xᵢ:
        P(yᵢ = 1 | xᵢ) = σ(w·xᵢ + b) = 1 / (1 + e^(-(w·xᵢ + b)))
        
        Where:
        - w = learned coefficient vector
        - b = learned bias/intercept
        - σ = sigmoid function
        - · = dot product
        
        Interpretation:
        The output represents the model's confidence that the sample belongs to class 1.
        - Probability close to 0.5 → Model is uncertain
        - Probability close to 0 → Model predicts class 0 with high confidence
        - Probability close to 1 → Model predicts class 1 with high confidence
        
        Args:
            X (list): List of feature vectors of shape (n_samples, n_features).
            
        Returns:
            list: Probability predictions ∈ [0, 1] for each sample.
            
        Examples:
            X = [[1.0, 2.0], [3.0, 4.0]]
            proba = model._predict_proba(X)  # [0.3, 0.8]
        """
        return [
            self._sigmoid(sum(w * xi for w, xi in zip(self.coefficients, x)) + self.intercept)
            for x in X
        ]
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using batch gradient descent.
        
        Optimization Algorithm:
        =======================
        Batch Gradient Descent minimizes cross-entropy loss iteratively:
        
        For each iteration t = 1, 2, ..., iterations:
        
        1. Forward Pass (Prediction):
           ŷᵢ = σ(w·xᵢ + b) for all samples i
           
        2. Compute Loss Gradients:
           eᵢ = ŷᵢ - yᵢ  (prediction error)
           
           ∂L/∂w_j = (1/m) Σ eᵢ·xᵢⱼ
           ∂L/∂b   = (1/m) Σ eᵢ
           
           Where m = number of samples
        
        3. Update Parameters:
           w_j^(t+1) = w_j^(t) - α·(∂L/∂w_j)
           b^(t+1)   = b^(t) - α·(∂L/∂b)
           
           Where α = learning_rate (controls step size)
        
        Convergence:
        - Gradients ∇w → 0 and ∇b → 0 indicates convergence
        - Larger learning_rate → Faster convergence but risk of oscillation
        - Smaller learning_rate → Slower convergence but more stable
        - Loss function is convex → Guaranteed to find global optimum
        
        Args:
            X (list): Training feature vectors of shape (n_samples, n_features).
            y (list): Binary training labels of shape (n_samples,), values ∈ {0, 1}.
            
        Side Effects:
            Updates self.coefficients and self.intercept in-place.
            
        Examples:
            X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
            y_train = [0, 0, 1, 1]
            model.fit(X_train, y_train)
        """
        n_features = len(X[0])
        n_samples = len(X)
        
        # Initialize coefficients
        self.coefficients = [0.0] * n_features
        self.intercept = 0.0
        
        # Gradient descent
        for iteration in range(self.iterations):
            # Compute predictions
            predictions = self._predict_proba(X)
            
            # Compute gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = predictions[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            
            # Update parameters
            for j in range(n_features):
                self.coefficients[j] -= (self.learning_rate / n_samples) * dw[j]
            self.intercept -= (self.learning_rate / n_samples) * db
    
    def predict(self, X):
        """
        Make binary predictions on new data using the learned model.
        
        Decision Rule:
        ==============
        For each sample xᵢ:
        
        1. Compute prediction probability:
           ŷᵢ = σ(w·xᵢ + b)
        
        2. Apply threshold:
           ŷ_class = 1  if ŷᵢ ≥ 0.5
           ŷ_class = 0  if ŷᵢ < 0.5
        
        Equivalently on the logit scale:
           ŷ_class = 1  if w·xᵢ + b ≥ 0
           ŷ_class = 0  if w·xᵢ + b < 0
        
        The threshold 0.5 can be adjusted for different precision-recall tradeoffs:
        - Lower threshold → More samples predicted as class 1 (higher recall, lower precision)
        - Higher threshold → Fewer samples predicted as class 1 (lower recall, higher precision)
        
        Args:
            X (list): Feature vectors for prediction of shape (n_samples, n_features).
            
        Returns:
            list: Binary predictions ∈ {0, 1} for each sample.
            
        Requires:
            Model must be fitted first via fit(X, y).
            
        Examples:
            predictions = model.predict([[1, 2], [3, 4]])  # [0, 1]
        """
        probabilities = self._predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probabilities]
    
    def eval(self, X, y):
        """
        Evaluate the model using accuracy metric on labeled data.
        
        Performance Metric:
        ===================
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
                 = Number of Correct Predictions / Total Predictions
        
        Where:
        - TP (True Positive) = Predicted 1, Actual 1
        - TN (True Negative) = Predicted 0, Actual 0
        - FP (False Positive) = Predicted 1, Actual 0
        - FN (False Negative) = Predicted 0, Actual 1
        
        Mathematical Formula:
        Accuracy = (1/m) Σ 𝟙[ŷᵢ = yᵢ]
        
        Where:
        - m = number of samples
        - 𝟙[·] = indicator function (1 if condition true, 0 otherwise)
        - ŷᵢ = model prediction
        - yᵢ = true label
        
        Interpretation:
        - Accuracy = 1.0 → Perfect predictions
        - Accuracy = 0.5 → Random guessing (for balanced binary classification)
        - Accuracy = 0.0 → All predictions incorrect
        
        Limitations:
        Accuracy can be misleading on imbalanced datasets. For example:
        - If 99% of samples are class 0, a model that always predicts 0
          achieves 99% accuracy but is useless for detecting class 1.
        - Consider precision, recall, F1-score for imbalanced data.
        
        Args:
            X (list): Feature vectors of shape (n_samples, n_features).
            y (list): True binary labels of shape (n_samples,), values ∈ {0, 1}.
            
        Returns:
            float: Accuracy score ∈ [0, 1].
            
        Examples:
            accuracy = model.eval(X_test, y_test)  # 0.85
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
