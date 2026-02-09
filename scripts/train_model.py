"""
Model Training Script

Train Random Forest model to predict HOMA-IR (insulin resistance) and
test the three-tier cascade hypothesis:
    High-sugar diet + Low physical activity → Inflammation → Insulin resistance

Usage:
    python scripts/train_model.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.random_forest import PediatricSentinelModel
from src.models.trainer import prepare_data, train_with_cv, hyperparameter_tuning
from src.models.evaluator import (
    evaluate_regression,
    evaluate_early_detection,
    plot_predictions,
    plot_residuals
)
from config.constants import RANDOM_STATE


def main():
    """Main training function."""

    print("\n" + "=" * 70)
    print("THE PEDIATRIC SENTINEL - MODEL TRAINING")
    print("=" * 70)
    print("\nTesting Three-Tier Cascade Hypothesis:")
    print("  Input:    High-sugar diet + Low physical activity")
    print("  Mediator: Inflammation (CRP)")
    print("  Output:   Insulin resistance (HOMA-IR)")
    print("=" * 70)

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "processed" / "modeling_dataset.csv"

    if not data_file.exists():
        print(f"\n[FAIL] Data file not found: {data_file}")
        print(f"\nPlease run: python scripts/engineer_features.py")
        sys.exit(1)

    print(f"\n[OK] Loading: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # =========================================================================
    # 2. DEFINE FEATURES
    # =========================================================================
    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    # Core features (4 required inputs for original model)
    core_features = [
        'comprehensive_inactivity_score',  # Physical activity (Tier 1 - Input)
        'DR1TSUGR',                        # Sugar intake (Tier 1 - Input)
        'DR1TFIBE',                        # Fiber intake (Tier 1 - Input)
        'LBXHSCRP'                         # CRP inflammation (Tier 2 - Mediator)
    ]

    # NEW: Metabolic features for R² improvement (86-100% coverage)
    metabolic_features = []

    # Anthropometric features
    if 'BMXBMI' in df.columns:
        metabolic_features.append('BMXBMI')  # Body Mass Index (95%)
    if 'BMXWAIST' in df.columns:
        metabolic_features.append('BMXWAIST')  # Waist circumference (92%)

    # Glycemic control
    if 'LBXGH' in df.columns:
        metabolic_features.append('LBXGH')  # HbA1c (86%)

    # Blood pressure
    if 'BPXSY2' in df.columns:
        metabolic_features.append('BPXSY2')  # Systolic BP (90.5%)
    if 'BPXDI2' in df.columns:
        metabolic_features.append('BPXDI2')  # Diastolic BP (90.5%)

    # Diet composition
    if 'carb_percent' in df.columns:
        metabolic_features.append('carb_percent')  # % calories from carbs (100%)

    # Additional features (improve model performance)
    additional_features = []

    # Add NLR (Neutrophil-to-Lymphocyte Ratio) if available
    if 'nlr' in df.columns:
        additional_features.append('nlr')  # Tier 2 - Immune activation marker

    # Add age, gender if available
    if 'RIDAGEYR' in df.columns:
        additional_features.append('RIDAGEYR')
    if 'RIAGENDR' in df.columns:
        additional_features.append('RIAGENDR')

    # NEW: Interaction features for capturing non-linear relationships
    interaction_features = []

    # Tier 1 × Tier 1: Diet × Activity
    if 'sugar_inactivity_interaction' in df.columns:
        interaction_features.append('sugar_inactivity_interaction')
    if 'bmi_inactivity_interaction' in df.columns:
        interaction_features.append('bmi_inactivity_interaction')

    # Tier 1 → Tier 2: Environmental stress → Inflammation
    if 'sugar_crp_interaction' in df.columns:
        interaction_features.append('sugar_crp_interaction')

    # Tier 2 × Confounders: Inflammation feedback loops
    if 'crp_bmi_interaction' in df.columns:
        interaction_features.append('crp_bmi_interaction')

    # Polynomial features
    if 'bmi_squared' in df.columns:
        interaction_features.append('bmi_squared')
    if 'crp_squared' in df.columns:
        interaction_features.append('crp_squared')

    # Combine all features
    all_features = core_features + metabolic_features + additional_features + interaction_features

    print(f"\nCore features (n={len(core_features)}):")
    for i, feature in enumerate(core_features, 1):
        valid_count = df[feature].notna().sum()
        print(f"  {i}. {feature}: {valid_count:,} valid values")

    if metabolic_features:
        print(f"\nMetabolic features (n={len(metabolic_features)}) - NEW for R² improvement:")
        for i, feature in enumerate(metabolic_features, 1):
            valid_count = df[feature].notna().sum()
            print(f"  {i}. {feature}: {valid_count:,} valid values")

    if additional_features:
        print(f"\nAdditional features (n={len(additional_features)}):")
        for i, feature in enumerate(additional_features, 1):
            valid_count = df[feature].notna().sum()
            print(f"  {i}. {feature}: {valid_count:,} valid values")

    if interaction_features:
        print(f"\nInteraction features (n={len(interaction_features)}) - NEW for non-linear relationships:")
        for i, feature in enumerate(interaction_features, 1):
            valid_count = df[feature].notna().sum()
            print(f"  {i}. {feature}: {valid_count:,} valid values")

    print(f"\nTotal features: {len(all_features)}")
    print(f"  Core: {len(core_features)}")
    print(f"  Metabolic: {len(metabolic_features)}")
    print(f"  Additional: {len(additional_features)}")
    print(f"  Interactions: {len(interaction_features)}")

    # =========================================================================
    # 3. PREPARE DATA (Train/Val/Test Split)
    # =========================================================================
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        df,
        required_features=all_features,
        target_column='HOMA_IR'
    )

    # Get glucose for early detection analysis
    glucose_train = df.loc[y_train.index, 'LBXGLU']
    glucose_val = df.loc[y_val.index, 'LBXGLU']
    glucose_test = df.loc[y_test.index, 'LBXGLU']

    # =========================================================================
    # 4. CROSS-VALIDATION (Check model stability)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: CROSS-VALIDATION ON TRAINING SET")
    print("=" * 70)

    # Create model with default hyperparameters
    model_cv = PediatricSentinelModel(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )

    # Perform 5-fold cross-validation
    cv_results = train_with_cv(model_cv.model, X_train, y_train, n_folds=5)

    # =========================================================================
    # 5. HYPERPARAMETER TUNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [20, 30, 40, None],  # Try deeper trees and unrestricted depth
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
        'min_impurity_decrease': [0.0, 0.001, 0.01]
    }

    print("\nThis may take 10-30 minutes depending on your CPU...")
    print("Grid search will test multiple hyperparameter combinations.")

    # Run grid search
    tuning_results = hyperparameter_tuning(
        X_train, y_train,
        param_grid=param_grid,
        cv=5,
        verbose=2
    )

    best_params = tuning_results['best_params']

    # =========================================================================
    # 6. TRAIN FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING FINAL MODEL (OPTIMIZED)")
    print("=" * 70)

    model = PediatricSentinelModel(**best_params)

    print("\nTraining Random Forest with optimized hyperparameters...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(all_features)}")
    print(f"  Hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")

    model.fit(X_train, y_train)

    print("\n[OK] Model training complete!")

    # =========================================================================
    # 6. FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    feature_importance = model.get_feature_importance()

    print("\nTop 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

    # Verify three-tier cascade hypothesis
    tier1_features = ['comprehensive_inactivity_score', 'DR1TSUGR', 'DR1TFIBE']
    tier2_features = ['LBXHSCRP', 'nlr']

    tier1_importance = feature_importance[
        feature_importance['feature'].isin(tier1_features)
    ]['importance'].sum()

    tier2_importance = feature_importance[
        feature_importance['feature'].isin(tier2_features)
    ]['importance'].sum()

    print(f"\n" + "-" * 70)
    print("THREE-TIER CASCADE IMPORTANCE")
    print("-" * 70)
    print(f"Tier 1 (Diet + Activity):  {tier1_importance:.4f} ({tier1_importance*100:.1f}%)")
    print(f"Tier 2 (Inflammation):     {tier2_importance:.4f} ({tier2_importance*100:.1f}%)")

    # =========================================================================
    # 7. EVALUATE ON VALIDATION SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: VALIDATION SET EVALUATION")
    print("=" * 70)

    y_pred_val = model.predict(X_val)

    val_metrics = evaluate_regression(y_val, y_pred_val, set_name="Validation")
    val_early_detection = evaluate_early_detection(
        y_val, y_pred_val, glucose_val,
        homa_ir_threshold=2.5,
        glucose_threshold=100.0
    )

    # =========================================================================
    # 8. EVALUATE ON TEST SET (Final evaluation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: TEST SET EVALUATION (FINAL)")
    print("=" * 70)

    y_pred_test = model.predict(X_test)

    test_metrics = evaluate_regression(y_test, y_pred_test, set_name="Test")
    test_early_detection = evaluate_early_detection(
        y_test, y_pred_test, glucose_test,
        homa_ir_threshold=2.5,
        glucose_threshold=100.0
    )

    # =========================================================================
    # 9. SAVE MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    models_dir = project_root / "models" / "final"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "pediatric_sentinel_model.pkl"
    model.save_model(model_path)

    # Save feature names
    feature_names_path = models_dir / "feature_names.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(all_features))
    print(f"[OK] Feature names saved: {feature_names_path.name}")

    # Save training metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'hyperparameter_optimization': 'GridSearchCV',
        'optimized_hyperparameters': best_params,
        'n_features': len(all_features),
        'features': all_features,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'cv_mean_r2': cv_results['mean_r2'],
        'cv_std_r2': cv_results['std_r2'],
        'tuning_best_cv_r2': tuning_results['best_score'],
        'val_r2': val_metrics['r2_score'],
        'val_rmse': val_metrics['rmse'],
        'test_r2': test_metrics['r2_score'],
        'test_rmse': test_metrics['rmse'],
        'test_sensitivity': test_early_detection['sensitivity'],
        'test_specificity': test_early_detection['specificity'],
        'early_detection_rate': test_early_detection['early_detection_rate']
    }

    metadata_path = models_dir / "model_metadata.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for key, value in metadata.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"[OK] Metadata saved: {metadata_path.name}")

    # =========================================================================
    # 10. SAVE PLOTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plots_dir = project_root / "models" / "final" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Prediction plot
    plot_predictions(
        y_test,
        y_pred_test,
        save_path=plots_dir / "predictions_test.png"
    )

    # Residual plot
    plot_residuals(
        y_test,
        y_pred_test,
        save_path=plots_dir / "residuals_test.png"
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    print(f"\nModel Performance Summary:")
    print(f"  Cross-Validation R²:  {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
    print(f"  Validation R²:        {val_metrics['r2_score']:.4f}")
    print(f"  Test R²:              {test_metrics['r2_score']:.4f}")
    print(f"  Test RMSE:            {test_metrics['rmse']:.4f}")

    print(f"\nEarly Detection Performance:")
    print(f"  Sensitivity:          {test_early_detection['sensitivity']:.3f}")
    print(f"  Specificity:          {test_early_detection['specificity']:.3f}")
    print(f"  Early Detection Rate: {test_early_detection['early_detection_rate']:.1%}")

    # Check success criteria
    target_r2 = 0.70
    target_sensitivity = 0.75

    success = True

    if test_metrics['r2_score'] >= target_r2:
        print(f"\n[OK] R^2 target ({target_r2:.2f}) achieved!")
    else:
        print(f"\n[WARN] R^2 target ({target_r2:.2f}) not achieved (current: {test_metrics['r2_score']:.4f})")
        success = False

    if test_early_detection['sensitivity'] >= target_sensitivity:
        print(f"[OK] Sensitivity target ({target_sensitivity:.2f}) achieved!")
    else:
        print(f"[WARN] Sensitivity target ({target_sensitivity:.2f}) not achieved (current: {test_early_detection['sensitivity']:.3f})")
        success = False

    if success:
        print(f"\n" + "=" * 70)
        print("SUCCESS! All targets achieved!")
        print("=" * 70)
    else:
        print(f"\n" + "=" * 70)
        print("PARTIAL SUCCESS - Some targets not met")
        print("=" * 70)
        print("\nConsider:")
        print("  - Hyperparameter tuning: python -c \"from src.models.trainer import hyperparameter_tuning; ...\"")
        print("  - Feature engineering: Add interaction terms (sugar × activity)")
        print("  - More data: Download additional NHANES cycles")

    print(f"\nModel saved: {model_path}")
    print(f"\nNext steps:")
    print(f"  1. Review feature importance and SHAP analysis")
    print(f"  2. Build Streamlit application: app/streamlit_app.py")
    print(f"  3. Test predictions: model = PediatricSentinelModel.load_model('{model_path.name}')")


if __name__ == "__main__":
    main()
