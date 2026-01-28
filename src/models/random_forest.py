"""
Random Forest Model for Pediatric Type 2 Diabetes Risk Prediction

Predicts HOMA-IR (insulin resistance) using lifestyle and biomarker inputs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Dict
import joblib
from pathlib import Path


class PediatricSentinelModel:
    """
    Random Forest Regressor for predicting HOMA-IR.

    Model tests the three-tier cascade hypothesis:
    Input (Diet + Activity) → Mediator (Inflammation) → Output (Insulin Resistance)
    """

    def __init__(self, **hyperparams):
        """
        Initialize Random Forest model.

        Default hyperparameters optimized for small-medium datasets:
        - n_estimators=200: Balance between performance and speed
        - max_depth=20: Prevent overfitting on 886 samples
        - min_samples_split=5: Require 5 samples to split
        - min_samples_leaf=2: Require 2 samples per leaf
        - max_features='sqrt': Use sqrt(n_features) per split
        - random_state=42: Reproducibility
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }

        # Override defaults with user-provided params
        default_params.update(hyperparams)

        self.model = RandomForestRegressor(**default_params)
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix (DataFrame)
            y: Target variable (HOMA-IR values)

        Returns:
            self: Fitted model
        """
        self.feature_names_ = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix (DataFrame)

        Returns:
            Predicted HOMA-IR values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def predict_risk_category(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk categories instead of continuous HOMA-IR.

        Categories:
        - 0: Normal (<2.5)
        - 1: Moderate (2.5-5.0)
        - 2: Severe (>5.0)

        Args:
            X: Feature matrix

        Returns:
            Risk categories (0, 1, 2)
        """
        homa_ir_pred = self.predict(X)

        categories = np.zeros_like(homa_ir_pred, dtype=int)
        categories[(homa_ir_pred >= 2.5) & (homa_ir_pred < 5.0)] = 1
        categories[homa_ir_pred >= 5.0] = 2

        return categories

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained Random Forest.

        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, filepath: Path):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model (.pkl file)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[OK] Model saved: {filepath}")

    @staticmethod
    def load_model(filepath: Path) -> 'PediatricSentinelModel':
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model (.pkl file)

        Returns:
            Loaded model
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        print(f"[OK] Model loaded: {filepath}")
        return model


if __name__ == "__main__":
    # Quick test
    print("THE PEDIATRIC SENTINEL - Random Forest Model")
    print("=" * 70)

    # Create dummy data for testing
    X_test = pd.DataFrame({
        'comprehensive_inactivity_score': [30, 50, 80],
        'DR1TSUGR': [40, 80, 120],
        'DR1TFIBE': [25, 15, 8],
        'LBXHSCRP': [1.0, 3.0, 6.0]
    })

    y_test = pd.Series([1.5, 3.0, 6.5])

    # Train model
    model = PediatricSentinelModel()
    model.fit(X_test, y_test)

    # Predict
    predictions = model.predict(X_test)
    print(f"\nPredictions: {predictions}")

    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nFeature Importance:")
    print(importance)

    print("\n[OK] Model test complete!")
