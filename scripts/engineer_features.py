"""
Feature Engineering Master Script

Orchestrates all feature engineering steps:
1. Clinical features (HOMA-IR)
2. Biological features (synthetic miRNA-155)
3. Nutritional features (NSI)
4. Physical activity features

Usage:
    python scripts/engineer_features.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.clinical import calculate_all_clinical_features
from src.features.biological import calculate_all_biological_features
from src.features.nutritional import calculate_all_nutritional_features
from src.features.physical import calculate_all_physical_features
from src.features.interactions import create_interaction_features


def engineer_all_features(
    input_file: Path,
    output_file: Path,
    mirna_method: str = "log_linear"
) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.

    Args:
        input_file: Path to input CSV (multi_cycle_pediatric.csv)
        output_file: Path to output CSV
        mirna_method: Method for synthetic miRNA calculation

    Returns:
        DataFrame with all engineered features
    """
    print("\n" + "=" * 70)
    print("THE PEDIATRIC SENTINEL - FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Load data
    print(f"\n[OK] Loading: {input_file.name}")
    df = pd.read_csv(input_file)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    initial_columns = df.shape[1]

    # 1. Clinical Features
    df = calculate_all_clinical_features(df)

    # 2. Biological Features
    df = calculate_all_biological_features(df, mirna_method=mirna_method)

    # 3. Nutritional Features
    df = calculate_all_nutritional_features(df)

    # 4. Physical Activity Features
    df = calculate_all_physical_features(df)

    # Summary of engineered features
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    new_columns = df.shape[1] - initial_columns
    print(f"\nOriginal columns: {initial_columns}")
    print(f"New features: {new_columns}")
    print(f"Total columns: {df.shape[1]}")

    # List new feature columns
    print(f"\nKey engineered features:")

    key_features = [
        'HOMA_IR',
        'HOMA_IR_category',
        'synthetic_mirna155',
        'inflammatory_index',
        'nutritional_stress_index',
        'nsi_category',
        'physical_activity_score',
        'activity_category'
    ]

    for feature in key_features:
        if feature in df.columns:
            valid_count = df[feature].notna().sum()
            print(f"  - {feature}: {valid_count:,} valid values")

    # Check completeness for modeling
    print(f"\n" + "=" * 70)
    print("DATA COMPLETENESS FOR MODELING")
    print("=" * 70)

    # Required features for model
    required_features = [
        'HOMA_IR',              # Target variable
        'LBXHSCRP',             # CRP (biomarker)
        'DR1TSUGR',             # Sugar intake
        'DR1TFIBE',             # Fiber intake
        'comprehensive_inactivity_score'  # Comprehensive physical inactivity score (70%+ coverage)
    ]

    complete_cases = df[required_features].notna().all(axis=1)
    complete_count = complete_cases.sum()

    print(f"\nRequired features for modeling:")
    for feature in required_features:
        if feature in df.columns:
            valid = df[feature].notna().sum()
            pct = valid / len(df) * 100
            print(f"  {feature}: {valid:,}/{len(df):,} ({pct:.1f}%)")

    print(f"\nComplete cases (all required features): {complete_count:,}/{len(df):,} "
          f"({complete_count/len(df)*100:.1f}%)")

    # HOMA-IR distribution (target variable)
    if 'HOMA_IR' in df.columns:
        homa_ir_valid = df['HOMA_IR'].dropna()
        if len(homa_ir_valid) > 0:
            print(f"\nHOMA-IR Distribution (Target Variable):")
            print(f"  Count: {len(homa_ir_valid):,}")
            print(f"  Mean: {homa_ir_valid.mean():.2f}")
            print(f"  Median: {homa_ir_valid.median():.2f}")
            print(f"  Std Dev: {homa_ir_valid.std():.2f}")
            print(f"  Min: {homa_ir_valid.min():.2f}")
            print(f"  Max: {homa_ir_valid.max():.2f}")

            # Risk distribution
            normal = (homa_ir_valid < 2.5).sum()
            moderate = ((homa_ir_valid >= 2.5) & (homa_ir_valid < 5.0)).sum()
            severe = (homa_ir_valid >= 5.0).sum()

            print(f"\n  Risk Categories:")
            print(f"    Normal (<2.5): {normal} ({normal/len(homa_ir_valid)*100:.1f}%)")
            print(f"    Moderate (2.5-5.0): {moderate} ({moderate/len(homa_ir_valid)*100:.1f}%)")
            print(f"    Severe (>5.0): {severe} ({severe/len(homa_ir_valid)*100:.1f}%)")

    # Save engineered dataset
    print(f"\n" + "=" * 70)
    print("SAVING ENGINEERED DATASET")
    print("=" * 70)

    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved: {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    print(f"\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Explore features in notebook: notebooks/02_feature_engineering.ipynb")
    print(f"  2. Train model: python scripts/train_model.py")

    return df


def main():
    """Main function to run feature engineering."""

    # File paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "processed" / "multi_cycle_pediatric.csv"
    output_file = project_root / "data" / "processed" / "feature_engineered.csv"

    # Check if input file exists
    if not input_file.exists():
        print(f"[FAIL] Input file not found: {input_file}")
        print(f"\nPlease run: python scripts/download_multi_cycle.py")
        sys.exit(1)

    # Run feature engineering
    df = engineer_all_features(
        input_file=input_file,
        output_file=output_file,
        mirna_method="log_linear"
    )

    # Create interaction features for model improvement
    print("\n" + "=" * 70)
    print("CREATING INTERACTION FEATURES")
    print("=" * 70)
    df = create_interaction_features(df, verbose=True)

    # Save updated feature-engineered dataset with interactions
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Updated feature_engineered.csv with interaction features")

    # Create filtered dataset with only complete cases for modeling
    required_features = [
        # Identifiers
        'SEQN',
        'NHANES_CYCLE',

        # Demographics & Confounders
        'RIDAGEYR',
        'RIAGENDR',
        'BMXBMI',           # NEW: Body Mass Index (95% coverage)
        'BMXWAIST',         # NEW: Waist circumference (92% coverage)
        'bmi_category',     # NEW: BMI classification

        # Raw biomarkers
        'LBXGLU',
        'LBXIN',
        'LBXHSCRP',
        'LBXGH',            # NEW: HbA1c (86% coverage)
        'hba1c_category',   # NEW: HbA1c classification

        # Blood pressure
        'BPXSY2',           # NEW: Systolic BP (90.5% coverage)
        'BPXDI2',           # NEW: Diastolic BP (90.5% coverage)

        # Dietary features
        'DR1TSUGR',
        'DR1TFIBE',
        'carb_percent',     # NEW: % calories from carbs (100% coverage)
        'protein_percent',  # NEW: % calories from protein
        'fat_percent',      # NEW: % calories from fat

        # Physical activity
        'PAD680',
        'PAQ706',
        'PAD733',

        # Target & engineered clinical
        'HOMA_IR',
        'HOMA_B',           # NEW: Beta-cell function (secondary target)
        'HOMA_IR_category',
        'synthetic_mirna155',
        'inflammatory_index',
        'nutritional_stress_index',
        'nsi_category',

        # Physical activity scores
        'physical_activity_score',
        'activity_category',
        'sedentary_score',
        'active_score',
        'comprehensive_inactivity_score',

        # Interaction features
        'sugar_inactivity_interaction',
        'nsi_inactivity_interaction',
        'bmi_inactivity_interaction',
        'waist_inactivity_interaction',
        'sugar_crp_interaction',
        'crp_bmi_interaction',
        'crp_waist_interaction',
        'bmi_squared',
        'crp_squared',
        'sugar_squared',
        'hba1c_bmi_interaction'
    ]

    # Keep only columns that exist
    available_features = [col for col in required_features if col in df.columns]

    # Filter to complete cases on critical features
    # Core 4 features + key metabolic markers for improved R²
    critical_features = [
        'HOMA_IR',                          # Target variable
        'LBXHSCRP',                         # CRP (Tier 2 mediator)
        'DR1TSUGR',                         # Sugar intake (Tier 1)
        'DR1TFIBE',                         # Fiber intake (Tier 1)
        'comprehensive_inactivity_score',   # Physical activity (Tier 1)
        'BMXBMI',                           # BMI (confounder - 95% coverage)
        'carb_percent',                     # Diet composition (100% coverage)
    ]
    critical_available = [col for col in critical_features if col in df.columns]

    print(f"\n" + "=" * 70)
    print("FILTERING FOR COMPLETE CASES")
    print("=" * 70)
    print(f"Critical features for modeling (all required):")
    for feat in critical_available:
        coverage = df[feat].notna().sum()
        pct = coverage / len(df) * 100
        print(f"  {feat}: {coverage:,}/{len(df):,} ({pct:.1f}%)")

    df_complete = df[df[critical_available].notna().all(axis=1)].copy()
    df_complete = df_complete[available_features]

    # Save complete cases dataset
    complete_file = project_root / "data" / "processed" / "modeling_dataset.csv"
    df_complete.to_csv(complete_file, index=False)

    file_size_mb = complete_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved modeling dataset: {complete_file.name}")
    print(f"  Rows: {len(df_complete):,} (complete cases only)")
    print(f"  Columns: {df_complete.shape[1]} (selected features)")
    print(f"  File size: {file_size_mb:.2f} MB")

    print(f"\n" + "=" * 70)
    print("READY FOR MODEL TRAINING!")
    print("=" * 70)


if __name__ == "__main__":
    main()
