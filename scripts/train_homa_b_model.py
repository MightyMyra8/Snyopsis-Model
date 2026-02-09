"""
Train Model to Predict HOMA-B (Beta-Cell Function)

This script trains a Random Forest model to predict beta-cell function (HOMA-B)
from lifestyle and biomarker inputs.

Target: HOMA-B (%)
- Normal: 50-150%
- Dysfunction (<50%): Beta-cell exhaustion - CRITICAL!
- Compensatory (>150%): Pancreas overworking - at risk!

Purpose: Identify teenagers at risk of beta-cell exhaustion before diabetes develops
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def load_data():
    """Load the modeling dataset."""
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "processed" / "modeling_dataset.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        sys.exit(1)

    df = pd.read_csv(data_file)
    print(f"[OK] Loaded {len(df):,} participants")

    return df


def main():
    """Train HOMA-B prediction model."""

    print("\n" + "=" * 70)
    print("TRAIN HOMA-B (BETA-CELL FUNCTION) PREDICTION MODEL")
    print("=" * 70)

    # Load data
    df = load_data()

    # Check if HOMA_B is available
    if 'HOMA_B' not in df.columns:
        print("[FAIL] HOMA_B column not found in dataset!")
        print("  Run engineer_features.py first to calculate HOMA-B")
        sys.exit(1)

    # Filter to cases with HOMA-B
    df_complete = df.dropna(subset=['HOMA_B'])
    print(f"\nCases with HOMA-B: {len(df_complete):,}")

    # =========================================================================
    # 1. PREPARE FEATURES
    # =========================================================================
    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    # Use same features as HOMA-IR model for consistency
    feature_cols = [
        # Tier 1: Lifestyle
        'comprehensive_inactivity_score',
        'DR1TSUGR',
        'DR1TFIBE',
        # Tier 2: Inflammation
        'LBXHSCRP',
        'nlr',
        # Metabolic markers
        'BMXBMI',
        'BMXWAIST',
        'LBXGH',
        'BPXSY2',
        'BPXDI2',
        'carb_percent',
        # Demographics
        'RIDAGEYR',
        'RIAGENDR',
        # Interactions
        'sugar_inactivity_interaction',
        'bmi_inactivity_interaction',
        'sugar_crp_interaction',
        'crp_bmi_interaction',
        'bmi_squared',
        'crp_squared'
    ]

    # Filter to complete cases
    required_cols = feature_cols + ['HOMA_B']
    df_model = df_complete[required_cols].dropna()

    print(f"\nComplete cases (all features): {len(df_model):,}")

    X = df_model[feature_cols]
    y = df_model['HOMA_B']

    print(f"\nTarget variable (HOMA-B):")
    print(f"  Mean: {y.mean():.1f}%")
    print(f"  Median: {y.median():.1f}%")
    print(f"  Std: {y.std():.1f}%")
    print(f"  Range: {y.min():.1f}% - {y.max():.1f}%")

    # Categorize risk
    dysfunction = (y < 50).sum()
    normal = ((y >= 50) & (y <= 150)).sum()
    compensatory = (y > 150).sum()

    print(f"\nBeta-Cell Function Distribution:")
    print(f"  Dysfunction (<50%): {dysfunction} ({dysfunction/len(y)*100:.1f}%)")
    print(f"  Normal (50-150%): {normal} ({normal/len(y)*100:.1f}%)")
    print(f"  Compensatory (>150%): {compensatory} ({compensatory/len(y)*100:.1f}%)")

    # =========================================================================
    # 2. TRAIN/VAL/TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAIN/VAL/TEST SPLIT")
    print("=" * 70)

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"\nTrain set: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val set:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    # =========================================================================
    # 3. TRAIN RANDOM FOREST MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    print("\nFitting model...")
    model.fit(X_train, y_train)
    print("[OK] Model trained")

    # =========================================================================
    # 4. CROSS-VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION (5-FOLD)")
    print("=" * 70)

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    print(f"\nCV R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # =========================================================================
    # 5. EVALUATE ON TEST SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    y_pred_test = model.predict(X_test)

    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\nTest Set Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  MAE: {mae:.2f}%")

    # =========================================================================
    # 6. FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")

    # =========================================================================
    # 7. DYSFUNCTION DETECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("BETA-CELL DYSFUNCTION DETECTION")
    print("=" * 70)

    # Predict dysfunction (<50%)
    dysfunction_actual = (y_test < 50)
    dysfunction_predicted = (y_pred_test < 50)

    if dysfunction_actual.sum() > 0:
        tp = (dysfunction_actual & dysfunction_predicted).sum()
        fn = (dysfunction_actual & ~dysfunction_predicted).sum()
        fp = (~dysfunction_actual & dysfunction_predicted).sum()
        tn = (~dysfunction_actual & ~dysfunction_predicted).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"\nDysfunction Detection (HOMA-B < 50%):")
        print(f"  Actual cases: {dysfunction_actual.sum()}")
        print(f"  Predicted cases: {dysfunction_predicted.sum()}")
        print(f"  Sensitivity: {sensitivity:.2%}")
        print(f"  Specificity: {specificity:.2%}")
        print(f"  PPV: {ppv:.2%}")
    else:
        print("\n[NOTE] No dysfunction cases in test set")

    # =========================================================================
    # 8. COMPENSATORY DETECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPENSATORY PHASE DETECTION")
    print("=" * 70)

    # Predict compensatory (>150%)
    compensatory_actual = (y_test > 150)
    compensatory_predicted = (y_pred_test > 150)

    if compensatory_actual.sum() > 0:
        tp_comp = (compensatory_actual & compensatory_predicted).sum()
        fn_comp = (compensatory_actual & ~compensatory_predicted).sum()
        fp_comp = (~compensatory_actual & compensatory_predicted).sum()
        tn_comp = (~compensatory_actual & ~compensatory_predicted).sum()

        sensitivity_comp = tp_comp / (tp_comp + fn_comp) if (tp_comp + fn_comp) > 0 else 0
        specificity_comp = tn_comp / (tn_comp + fp_comp) if (tn_comp + fp_comp) > 0 else 0
        ppv_comp = tp_comp / (tp_comp + fp_comp) if (tp_comp + fp_comp) > 0 else 0

        print(f"\nCompensatory Detection (HOMA-B > 150%):")
        print(f"  Actual cases: {compensatory_actual.sum()}")
        print(f"  Predicted cases: {compensatory_predicted.sum()}")
        print(f"  Sensitivity: {sensitivity_comp:.2%}")
        print(f"  Specificity: {specificity_comp:.2%}")
        print(f"  PPV: {ppv_comp:.2%}")
    else:
        print("\n[NOTE] No compensatory cases in test set")

    # =========================================================================
    # 9. SAVE MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "models" / "beta_cell"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / "homa_b_model.pkl"
    joblib.dump(model, model_file)

    print(f"\n[OK] Saved model: {model_file.name}")

    # Save feature importance
    importance_file = output_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_file, index=False)

    print(f"[OK] Saved feature importance: {importance_file.name}")

    # Save metrics
    metrics = {
        'model_type': 'Random Forest Regressor',
        'target': 'HOMA-B (Beta-Cell Function %)',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_r2': r2,
        'test_rmse': rmse,
        'test_mae': mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_file = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    print(f"[OK] Saved metrics: {metrics_file.name}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    print(f"\n[OK] HOMA-B Prediction Model Trained!")
    print(f"\nPerformance:")
    print(f"  Test R² Score: {r2:.4f}")
    print(f"  CV R² Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Test RMSE: {rmse:.2f}%")

    print(f"\nTop 3 Predictors:")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.1%}")

    print(f"\nModel saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
