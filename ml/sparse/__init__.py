"""
Sparse linear models module.

This module contains sparse regularized linear models including:
- Lasso: L1-regularized linear regression for feature selection
"""

from .lasso import LassoRegression

__all__ = ['LassoRegression']