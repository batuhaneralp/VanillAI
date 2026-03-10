"""
Regularized linear models module.

This module contains regularized versions of linear models including:
- Ridge Regression: L2-regularized linear regression
- Ridge Classification: L2-regularized linear classification
"""

from .ridge import RidgeRegression, RidgeClassifier

__all__ = ['RidgeRegression', 'RidgeClassifier']