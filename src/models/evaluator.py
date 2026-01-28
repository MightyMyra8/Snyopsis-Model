"""
Model Evaluation and Validation

Calculate regression metrics and early detection performance.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
    set_name: str = "Test"
) -> Dict:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: Actual HOMA-IR values
        y_pred: Predicted HOMA-IR values
        set_name: Name of dataset (e.g., "Test", "Validation")

    Returns:
        Dictionary with regression metrics
    """
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Mean Absolute Percentage Error (handle division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100

    # Residuals
    residuals = y_true - y_pred
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    print(f"\n" + "=" * 70)
    print(f"{set_name.upper()} SET REGRESSION METRICS")
    print("=" * 70)
    print(f"R² Score:       {r2:.4f}")
    print(f"RMSE:           {rmse:.4f}")
    print(f"MAE:            {mae:.4f}")
    print(f"MAPE:           {mape:.2f}%")
    print(f"\nResiduals:")
    print(f"  Mean:         {mean_residual:.4f}")
    print(f"  Std Dev:      {std_residual:.4f}")

    # Check if target R² achieved
    target_r2 = 0.70
    if r2 >= target_r2:
        print(f"\n[OK] Target R² ({target_r2:.2f}) achieved!")
    else:
        print(f"\n[WARN] Target R² ({target_r2:.2f}) not achieved (current: {r2:.4f})")

    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'residuals': residuals
    }


def evaluate_early_detection(
    y_true: pd.Series,
    y_pred: np.ndarray,
    glucose: pd.Series,
    homa_ir_threshold: float = 2.5,
    glucose_threshold: float = 100.0
) -> Dict:
    """
    Evaluate early detection capability.

    Goal: Identify individuals with insulin resistance (HOMA-IR ≥ 2.5)
    but normal fasting glucose (<100 mg/dL). These are high-risk
    individuals who would be missed by glucose screening alone.

    Args:
        y_true: Actual HOMA-IR values
        y_pred: Predicted HOMA-IR values
        glucose: Fasting glucose levels (mg/dL)
        homa_ir_threshold: HOMA-IR cutoff for insulin resistance (default: 2.5)
        glucose_threshold: Normal glucose cutoff (default: 100 mg/dL)

    Returns:
        Dictionary with early detection metrics
    """
    print("\n" + "=" * 70)
    print("EARLY DETECTION ANALYSIS")
    print("=" * 70)

    # Identify groups
    normal_glucose = glucose < glucose_threshold
    high_risk_actual = y_true >= homa_ir_threshold
    high_risk_predicted = y_pred >= homa_ir_threshold

    # Early detection candidates: normal glucose + high HOMA-IR
    early_detection_candidates = normal_glucose & high_risk_actual
    early_detected = normal_glucose & high_risk_predicted & high_risk_actual
    early_missed = normal_glucose & ~high_risk_predicted & high_risk_actual

    # Calculate metrics
    n_candidates = early_detection_candidates.sum()
    n_detected = early_detected.sum()
    n_missed = early_missed.sum()

    early_detection_rate = n_detected / n_candidates if n_candidates > 0 else 0

    # Overall sensitivity and specificity
    true_positives = (high_risk_predicted & high_risk_actual).sum()
    false_positives = (high_risk_predicted & ~high_risk_actual).sum()
    true_negatives = (~high_risk_predicted & ~high_risk_actual).sum()
    false_negatives = (~high_risk_predicted & high_risk_actual).sum()

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0

    print(f"\nNormal Glucose (<{glucose_threshold} mg/dL): {normal_glucose.sum()} participants")
    print(f"High HOMA-IR (>={homa_ir_threshold}): {high_risk_actual.sum()} participants")
    print(f"\n" + "-" * 70)
    print("EARLY DETECTION PERFORMANCE")
    print("-" * 70)
    print(f"Early detection candidates:  {n_candidates}")
    print(f"  (normal glucose + high HOMA-IR)")
    print(f"\nSuccessfully detected:       {n_detected} ({early_detection_rate*100:.1f}%)")
    print(f"Missed (false negatives):    {n_missed} ({n_missed/n_candidates*100 if n_candidates > 0 else 0:.1f}%)")

    print(f"\n" + "-" * 70)
    print("OVERALL CLASSIFICATION PERFORMANCE")
    print("-" * 70)
    print(f"Sensitivity (recall):        {sensitivity:.3f}")
    print(f"Specificity:                 {specificity:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Low    Predicted High")
    print(f"Actual Low        {true_negatives:8d}         {false_positives:8d}")
    print(f"Actual High       {false_negatives:8d}         {true_positives:8d}")

    # Check if target sensitivity achieved
    target_sensitivity = 0.75
    if sensitivity >= target_sensitivity:
        print(f"\n[OK] Target sensitivity ({target_sensitivity:.2f}) achieved!")
    else:
        print(f"\n[WARN] Target sensitivity ({target_sensitivity:.2f}) not achieved (current: {sensitivity:.3f})")

    return {
        'early_detection_candidates': n_candidates,
        'early_detected': n_detected,
        'early_missed': n_missed,
        'early_detection_rate': early_detection_rate,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }


def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Path = None
) -> None:
    """
    Create scatter plot of predicted vs actual values.

    Args:
        y_true: Actual HOMA-IR values
        y_pred: Predicted HOMA-IR values
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=50)

    # Perfect prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # HOMA-IR risk thresholds
    ax.axvline(x=2.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk (2.5)')
    ax.axvline(x=5.0, color='red', linestyle='--', alpha=0.5, label='Severe Risk (5.0)')
    ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5)

    # Labels and title
    ax.set_xlabel('Actual HOMA-IR', fontsize=12)
    ax.set_ylabel('Predicted HOMA-IR', fontsize=12)
    ax.set_title('Predicted vs Actual HOMA-IR', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Path = None
) -> None:
    """
    Create residual plot to check for systematic errors.

    Args:
        y_true: Actual HOMA-IR values
        y_pred: Predicted HOMA-IR values
        save_path: Optional path to save plot
    """
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residual plot
    ax1.scatter(y_pred, residuals, alpha=0.5, s=50)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted HOMA-IR', fontsize=12)
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Residual distribution
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add normal curve
    from scipy import stats
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residuals.mean(), residuals.std())
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
    ax2_twin.set_ylabel('Probability Density', fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test evaluator
    print("THE PEDIATRIC SENTINEL - Model Evaluator Test")
    print("=" * 70)

    # Create dummy data
    np.random.seed(42)
    n_samples = 100

    y_true = pd.Series(np.random.uniform(0.5, 8.0, n_samples))
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add noise
    glucose = pd.Series(np.random.uniform(70, 130, n_samples))

    # Regression metrics
    metrics = evaluate_regression(y_true, y_pred, set_name="Test")

    # Early detection
    early_detection = evaluate_early_detection(y_true, y_pred, glucose)

    # Plots
    print("\nGenerating plots...")
    plot_predictions(y_true, y_pred)
    plot_residuals(y_true, y_pred)

    print("\n[OK] Evaluator test complete!")
