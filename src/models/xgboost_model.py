"""
XGBoost Model for Pediatric Insulin Resistance Prediction

XGBoost (Extreme Gradient Boosting) often outperforms Random Forest due to:
1. Sequential boosting (learns from previous tree errors)
2. Built-in regularization (L1/L2)
3. Better handling of missing values
4. Tree-based feature selection at each split

Expected R^2 improvement: +0.10-0.20 over Random Forest
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, Optional


class XGBoostSentinelModel:
    """
    XGBoost Regressor for predicting HOMA-IR (insulin resistance).

    Optimized for pediatric population with three-tier cascade hypothesis:
    - Tier 1 (Input): Diet + Physical Activity
    - Tier 2 (Mediator): Inflammation (CRP)
    - Tier 3 (Output): Insulin Resistance (HOMA-IR)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize XGBoost model with optimized hyperparameters.

        Args:
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum tree depth (6 is typical for XGBoost)
            learning_rate: Step size shrinkage (lower = more conservative)
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            gamma: Minimum loss reduction for split (regularization)
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel threads (-1 = use all cores)
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            objective='reg:squarederror',
            tree_method='hist',  # Fast histogram-based algorithm
            **kwargs
        )

        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True):
        """
        Train the XGBoost model.

        Args:
            X: Training features
            y: Training target (HOMA-IR)
            verbose: Print training progress

        Returns:
            self
        """
        self.feature_names = list(X.columns)

        if verbose:
            self.model.set_params(verbose=1)
        else:
            self.model.set_params(verbose=0)

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Predicted HOMA-IR values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            importance_type: Type of importance metric
                - 'gain': Average gain across all splits (default)
                - 'weight': Number of times feature is used
                - 'cover': Average coverage (samples affected)

        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")

        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)

        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })

        # Normalize importance to sum to 1
        importance_df['importance'] = importance_df['importance'] / importance_df['importance'].sum()

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def save_model(self, filepath: Path):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model (e.g., 'models/xgboost_sentinel.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save model that hasn't been fitted")

        joblib.dump(self, filepath)
        print(f"[OK] Model saved: {filepath}")

    @classmethod
    def load_model(cls, filepath: Path) -> 'XGBoostSentinelModel':
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded XGBoostSentinelModel
        """
        model = joblib.load(filepath)
        print(f"[OK] Model loaded: {filepath}")
        return model

    def get_params(self) -> Dict:
        """Get model hyperparameters."""
        return self.model.get_params()

    def set_params(self, **params):
        """Set model hyperparameters."""
        self.model.set_params(**params)
        return self


if __name__ == "__main__":
    # Test XGBoost model
    print("THE PEDIATRIC SENTINEL - XGBoost Model Test")
    print("=" * 70)

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # Load modeling dataset
    data_file = Path(__file__).parent.parent.parent / "data" / "processed" / "modeling_dataset.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        print(f"\nPlease run: python scripts/engineer_features.py")
        sys.exit(1)

    df = pd.read_csv(data_file)
    print(f"\n[OK] Loaded: {data_file.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # Define features
    required_features = [
        'comprehensive_inactivity_score',
        'DR1TSUGR',
        'DR1TFIBE',
        'LBXHSCRP',
        'BMXBMI',
        'BMXWAIST',
        'LBXGH'
    ]

    # Prepare data
    from sklearn.model_selection import train_test_split

    X = df[required_features].copy()
    y = df['HOMA_IR'].copy()

    # Drop missing values
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # Train model
    print("\nTraining XGBoost model...")
    model = XGBoostSentinelModel(
        n_estimators=100,  # Small for quick test
        max_depth=6,
        learning_rate=0.1,
        verbose=True
    )

    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nTest R^2 Score: {r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Feature importance
    print("\nFeature Importance:")
    importance = model.get_feature_importance()
    for i, row in importance.iterrows():
        print(f"  {i+1}. {row['feature']:35s} {row['importance']:.4f}")

    print(f"\n[OK] XGBoost model test complete!")
