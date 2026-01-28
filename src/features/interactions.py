"""
Feature Interactions Module

Creates interaction and polynomial features to capture non-linear relationships
in the three-tier cascade model for insulin resistance prediction.

Interaction Types:
1. Tier 1 × Tier 1: Environmental factors amplifying each other
2. Tier 1 × Tier 2: Environmental stress triggering biological response
3. Tier 2 × Confounders: Biological feedback loops
4. Polynomial: Non-linear dose-response relationships
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_interaction_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create interaction and polynomial features for improved model performance.

    Interactions capture synergistic effects where the combined impact of two
    features exceeds their individual contributions.

    Args:
        df: DataFrame with base features
        verbose: Print feature creation progress

    Returns:
        DataFrame with added interaction features
    """
    if verbose:
        print("\n" + "=" * 70)
        print("INTERACTION FEATURE ENGINEERING")
        print("=" * 70)

    initial_cols = df.shape[1]
    features_created = []

    # =========================================================================
    # TIER 1 × TIER 1: Diet × Physical Activity Interactions
    # =========================================================================

    # Interaction 1: Sugar × Inactivity
    # High sugar intake is more harmful when combined with sedentary behavior
    if 'DR1TSUGR' in df.columns and 'comprehensive_inactivity_score' in df.columns:
        df['sugar_inactivity_interaction'] = (
            df['DR1TSUGR'] * df['comprehensive_inactivity_score'] / 100
        )
        features_created.append('sugar_inactivity_interaction')

        if verbose:
            valid_count = df['sugar_inactivity_interaction'].notna().sum()
            mean_val = df['sugar_inactivity_interaction'].mean()
            print(f"\n[OK] Sugar × Inactivity interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Dietary stress amplified by sedentary behavior")

    # Interaction 2: NSI × Inactivity (alternative to sugar × inactivity)
    # Nutritional Stress Index already combines sugar/fiber ratio
    if 'nutritional_stress_index' in df.columns and 'comprehensive_inactivity_score' in df.columns:
        df['nsi_inactivity_interaction'] = (
            df['nutritional_stress_index'] * df['comprehensive_inactivity_score'] / 100
        )
        features_created.append('nsi_inactivity_interaction')

        if verbose:
            valid_count = df['nsi_inactivity_interaction'].notna().sum()
            mean_val = df['nsi_inactivity_interaction'].mean()
            print(f"\n[OK] NSI × Inactivity interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Poor diet quality amplified by inactivity")

    # =========================================================================
    # TIER 1 × CONFOUNDERS: Obesity × Lifestyle Interactions
    # =========================================================================

    # Interaction 3: BMI × Inactivity
    # Obesity combined with sedentary behavior compounds insulin resistance risk
    if 'BMXBMI' in df.columns and 'comprehensive_inactivity_score' in df.columns:
        df['bmi_inactivity_interaction'] = (
            df['BMXBMI'] * df['comprehensive_inactivity_score'] / 100
        )
        features_created.append('bmi_inactivity_interaction')

        if verbose:
            valid_count = df['bmi_inactivity_interaction'].notna().sum()
            mean_val = df['bmi_inactivity_interaction'].mean()
            print(f"\n[OK] BMI × Inactivity interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Obesity + sedentary compound metabolic risk")

    # Interaction 4: Waist × Inactivity
    # Central obesity (waist circumference) is strongly linked to insulin resistance
    if 'BMXWAIST' in df.columns and 'comprehensive_inactivity_score' in df.columns:
        df['waist_inactivity_interaction'] = (
            df['BMXWAIST'] * df['comprehensive_inactivity_score'] / 100
        )
        features_created.append('waist_inactivity_interaction')

        if verbose:
            valid_count = df['waist_inactivity_interaction'].notna().sum()
            mean_val = df['waist_inactivity_interaction'].mean()
            print(f"\n[OK] Waist × Inactivity interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Visceral fat + sedentary synergy")

    # =========================================================================
    # TIER 1 → TIER 2: Environmental Stress Triggering Inflammation
    # =========================================================================

    # Interaction 5: Sugar × CRP
    # Dietary stress (high sugar) triggers inflammatory response (elevated CRP)
    if 'DR1TSUGR' in df.columns and 'LBXHSCRP' in df.columns:
        df['sugar_crp_interaction'] = df['DR1TSUGR'] * df['LBXHSCRP']
        features_created.append('sugar_crp_interaction')

        if verbose:
            valid_count = df['sugar_crp_interaction'].notna().sum()
            mean_val = df['sugar_crp_interaction'].mean()
            print(f"\n[OK] Sugar × CRP interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Dietary stress -> inflammation cascade")

    # =========================================================================
    # TIER 2 × CONFOUNDERS: Inflammation Feedback Loops
    # =========================================================================

    # Interaction 6: CRP × BMI
    # Adipose tissue (BMI) secretes inflammatory cytokines → CRP feedback loop
    if 'LBXHSCRP' in df.columns and 'BMXBMI' in df.columns:
        df['crp_bmi_interaction'] = df['LBXHSCRP'] * df['BMXBMI']
        features_created.append('crp_bmi_interaction')

        if verbose:
            valid_count = df['crp_bmi_interaction'].notna().sum()
            mean_val = df['crp_bmi_interaction'].mean()
            print(f"\n[OK] CRP × BMI interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Inflammation-obesity feedback loop")

    # Interaction 7: CRP × Waist
    # Visceral fat is more inflammatory than subcutaneous fat
    if 'LBXHSCRP' in df.columns and 'BMXWAIST' in df.columns:
        df['crp_waist_interaction'] = df['LBXHSCRP'] * df['BMXWAIST']
        features_created.append('crp_waist_interaction')

        if verbose:
            valid_count = df['crp_waist_interaction'].notna().sum()
            mean_val = df['crp_waist_interaction'].mean()
            print(f"\n[OK] CRP × Waist interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Visceral fat inflammation amplification")

    # =========================================================================
    # POLYNOMIAL FEATURES: Non-linear Dose-Response Relationships
    # =========================================================================

    # Polynomial 1: BMI²
    # BMI effect is non-linear; severely obese individuals have disproportionate risk
    if 'BMXBMI' in df.columns:
        df['bmi_squared'] = df['BMXBMI'] ** 2
        features_created.append('bmi_squared')

        if verbose:
            valid_count = df['bmi_squared'].notna().sum()
            mean_val = df['bmi_squared'].mean()
            print(f"\n[OK] BMI² (polynomial): {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Non-linear obesity threshold effects")

    # Polynomial 2: CRP²
    # CRP effect is non-linear; very high inflammation has exponential impact
    if 'LBXHSCRP' in df.columns:
        df['crp_squared'] = df['LBXHSCRP'] ** 2
        features_created.append('crp_squared')

        if verbose:
            valid_count = df['crp_squared'].notna().sum()
            mean_val = df['crp_squared'].mean()
            print(f"\n[OK] CRP² (polynomial): {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Non-linear inflammation escalation")

    # Polynomial 3: Sugar² (if high outliers exist)
    if 'DR1TSUGR' in df.columns:
        df['sugar_squared'] = df['DR1TSUGR'] ** 2
        features_created.append('sugar_squared')

        if verbose:
            valid_count = df['sugar_squared'].notna().sum()
            mean_val = df['sugar_squared'].mean()
            print(f"\n[OK] Sugar² (polynomial): {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Extreme sugar intake threshold effects")

    # =========================================================================
    # METABOLIC MARKERS: HbA1c Interactions
    # =========================================================================

    # Interaction 8: HbA1c × BMI
    # Long-term glucose control worsens with obesity
    if 'LBXGH' in df.columns and 'BMXBMI' in df.columns:
        df['hba1c_bmi_interaction'] = df['LBXGH'] * df['BMXBMI']
        features_created.append('hba1c_bmi_interaction')

        if verbose:
            valid_count = df['hba1c_bmi_interaction'].notna().sum()
            mean_val = df['hba1c_bmi_interaction'].mean()
            print(f"\n[OK] HbA1c × BMI interaction: {valid_count:,} values")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Captures: Chronic glucose dysregulation + obesity")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    if verbose:
        print("\n" + "=" * 70)
        print("INTERACTION FEATURES COMPLETE")
        print("=" * 70)
        print(f"Original columns: {initial_cols}")
        print(f"Interaction features created: {len(features_created)}")
        print(f"Total columns: {df.shape[1]}")
        print(f"\nFeatures created:")
        for i, feature in enumerate(features_created, 1):
            print(f"  {i}. {feature}")

    return df


def get_interaction_feature_names() -> List[str]:
    """
    Return list of all possible interaction feature names.

    Useful for feature selection and model training.

    Returns:
        List of interaction feature column names
    """
    return [
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


def select_high_impact_interactions(
    df: pd.DataFrame,
    target_col: str = 'HOMA_IR',
    top_n: int = 6
) -> List[str]:
    """
    Select top N interaction features based on correlation with target.

    Args:
        df: DataFrame with interaction features and target
        target_col: Target variable column name
        top_n: Number of top interactions to select

    Returns:
        List of top N interaction feature names
    """
    interaction_features = get_interaction_feature_names()
    available_interactions = [f for f in interaction_features if f in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Calculate correlations
    correlations = {}
    for feature in available_interactions:
        if df[feature].notna().sum() > 0:
            corr = abs(df[[feature, target_col]].corr().iloc[0, 1])
            correlations[feature] = corr

    # Sort by correlation strength
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Return top N
    top_features = [f[0] for f in sorted_features[:top_n]]

    print(f"\nTop {top_n} Interaction Features (by correlation with {target_col}):")
    for i, (feature, corr) in enumerate(sorted_features[:top_n], 1):
        print(f"  {i}. {feature}: {corr:.4f}")

    return top_features


if __name__ == "__main__":
    # Test interaction feature creation
    print("THE PEDIATRIC SENTINEL - Interaction Feature Engineering Test")
    print("=" * 70)

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # Load feature-engineered data
    data_file = Path(__file__).parent.parent.parent / "data" / "processed" / "feature_engineered.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        print(f"\nPlease run: python scripts/engineer_features.py")
        sys.exit(1)

    df = pd.read_csv(data_file, low_memory=False)
    print(f"\n[OK] Loaded: {data_file.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # Create interaction features
    df = create_interaction_features(df, verbose=True)

    # Select top interactions
    if 'HOMA_IR' in df.columns:
        top_interactions = select_high_impact_interactions(df, 'HOMA_IR', top_n=6)

    print(f"\n[OK] Interaction feature test complete!")
