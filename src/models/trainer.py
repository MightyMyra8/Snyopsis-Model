"""
Model Training Pipeline

Handles data splitting, cross-validation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.constants import TARGET_COLUMN, RANDOM_STATE, N_FOLDS


def prepare_data(
    df: pd.DataFrame,
    required_features: list,
    target_column: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets.

    Split strategy:
    - 70% training (for model fitting)
    - 15% validation (for hyperparameter tuning)
    - 15% test (for final evaluation - never seen during training)

    Args:
        df: Feature-engineered dataset
        required_features: List of feature column names
        target_column: Name of target variable (default: HOMA_IR)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    # Check if all required features exist
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Select features and target
    X = df[required_features].copy()
    y = df[target_column].copy()

    print(f"\nDataset shape: {X.shape}")
    print(f"Target variable: {target_column}")
    print(f"Features: {len(required_features)}")
    print(f"  {', '.join(required_features)}")

    # Check for missing values
    missing_X = X.isnull().sum().sum()
    missing_y = y.isnull().sum()

    if missing_X > 0 or missing_y > 0:
        print(f"\n[WARN] Missing values detected:")
        print(f"  Features: {missing_X}")
        print(f"  Target: {missing_y}")
        print(f"  Dropping rows with missing values...")

        # Drop rows with any missing values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"  Remaining samples: {len(X)}")

    # Split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Split temp into validation (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(f"\n" + "-" * 70)
    print("TRAIN/VALIDATION/TEST SPLIT")
    print("-" * 70)
    print(f"Training set:   {len(X_train):4d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val):4d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test):4d} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Total:          {len(X):4d} samples")

    # Check target distribution in each set
    print(f"\n" + "-" * 70)
    print("TARGET DISTRIBUTION (HOMA-IR)")
    print("-" * 70)
    print(f"                Mean    Median  Std     Min     Max")
    print(f"Training:       {y_train.mean():6.2f}  {y_train.median():6.2f}  "
          f"{y_train.std():6.2f}  {y_train.min():6.2f}  {y_train.max():6.2f}")
    print(f"Validation:     {y_val.mean():6.2f}  {y_val.median():6.2f}  "
          f"{y_val.std():6.2f}  {y_val.min():6.2f}  {y_val.max():6.2f}")
    print(f"Test:           {y_test.mean():6.2f}  {y_test.median():6.2f}  "
          f"{y_test.std():6.2f}  {y_test.min():6.2f}  {y_test.max():6.2f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_with_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS
) -> Dict:
    """
    Train model with k-fold cross-validation.

    Cross-validation provides robust estimate of model performance
    by training on k different train/validation splits.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target variable
        n_folds: Number of CV folds (default: 5)

    Returns:
        Dictionary with CV results
    """
    print("\n" + "=" * 70)
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    # Perform cross-validation
    cv_scores = cross_val_score(
        model,
        X, y,
        cv=n_folds,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    # Calculate metrics
    mean_r2 = cv_scores.mean()
    std_r2 = cv_scores.std()

    print(f"\nCross-Validation R² Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")

    print(f"\n" + "-" * 70)
    print(f"Mean R² Score: {mean_r2:.4f} ± {std_r2:.4f}")
    print("-" * 70)

    # Check for overfitting (high variance across folds)
    if std_r2 > 0.10:
        print(f"[WARN] High variance across folds (std={std_r2:.4f})")
        print(f"       Model may be unstable or overfitting")
    else:
        print(f"[OK] Low variance across folds - stable model")

    return {
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'cv_scores': cv_scores,
        'min_r2': cv_scores.min(),
        'max_r2': cv_scores.max()
    }


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict = None,
    cv: int = 5,
    verbose: int = 1
) -> Dict:
    """
    Perform grid search for optimal hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target
        param_grid: Hyperparameter grid to search
        cv: Number of cross-validation folds
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)

    Returns:
        Dictionary with best parameters and scores
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations: {total_combinations}")
    print(f"CV folds: {cv}")
    print(f"Total fits: {total_combinations * cv}")
    print(f"\nSearching for best hyperparameters...")

    # Perform grid search
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # Extract results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\n" + "-" * 70)
    print("BEST HYPERPARAMETERS")
    print("-" * 70)
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV R² Score: {best_score:.4f}")

    # Show top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score']
    ]

    print(f"\n" + "-" * 70)
    print("TOP 5 PARAMETER COMBINATIONS")
    print("-" * 70)
    for i, row in top_5.iterrows():
        print(f"\n{i+1}. R² = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"   {row['params']}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': grid_search.cv_results_,
        'best_estimator': grid_search.best_estimator_
    }


if __name__ == "__main__":
    # Test data preparation
    print("THE PEDIATRIC SENTINEL - Training Pipeline Test")
    print("=" * 70)

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
        'LBXHSCRP'
    ]

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        df, required_features
    )

    # Test cross-validation
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    cv_results = train_with_cv(model, X_train, y_train, n_folds=5)

    print(f"\n[OK] Training pipeline test complete!")
