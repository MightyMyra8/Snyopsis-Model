"""
Individual Behavior Analysis: Real Examples of Lifestyle and Diabetes Risk

This script explores individual-level data to show REAL PEOPLE and how their
daily behaviors (physical activity, diet) relate to insulin resistance.

Purpose: Create concrete examples for science fair presentation
         - Show "high risk" vs "low risk" profiles
         - Demonstrate physical activity effects
         - Make abstract statistics REAL
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def load_data():
    """Load the modeling dataset with all features."""
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "processed" / "modeling_dataset.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        sys.exit(1)

    df = pd.read_csv(data_file)
    print(f"[OK] Loaded {len(df):,} participants")

    return df


def categorize_activity(score):
    """Categorize physical activity level."""
    if score < 33:
        return "Highly Active"
    elif score < 66:
        return "Moderately Active"
    else:
        return "Inactive"


def categorize_diet(sugar, fiber):
    """Categorize diet quality."""
    sugar_ratio = sugar / (fiber + 1)

    if sugar_ratio < 5:
        return "Excellent Diet"
    elif sugar_ratio < 10:
        return "Good Diet"
    elif sugar_ratio < 15:
        return "Poor Diet"
    else:
        return "Very Poor Diet"


def categorize_risk(homa_ir):
    """Categorize diabetes risk."""
    if homa_ir < 2.5:
        return "Low Risk"
    elif homa_ir < 5.0:
        return "Moderate Risk"
    else:
        return "High Risk"


def main():
    """Explore individual behaviors and their diabetes risk."""

    print("\n" + "=" * 70)
    print("INDIVIDUAL BEHAVIOR ANALYSIS")
    print("=" * 70)
    print("\nExploring real people's behaviors and diabetes risk")

    # Load data
    df = load_data()

    # Filter to complete cases with all key features
    required_cols = [
        'comprehensive_inactivity_score',
        'DR1TSUGR',
        'DR1TFIBE',
        'LBXHSCRP',
        'BMXBMI',
        'HOMA_IR',
        'LBXGLU',
        'RIDAGEYR',
        'RIAGENDR'
    ]

    df_complete = df[required_cols].dropna()
    print(f"\nComplete cases: {len(df_complete):,}")

    # Add categorical variables
    df_complete['activity_category'] = df_complete['comprehensive_inactivity_score'].apply(categorize_activity)
    df_complete['diet_category'] = df_complete.apply(
        lambda row: categorize_diet(row['DR1TSUGR'], row['DR1TFIBE']), axis=1
    )
    df_complete['risk_category'] = df_complete['HOMA_IR'].apply(categorize_risk)

    # =========================================================================
    # 1. SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nActivity Levels:")
    print(df_complete['activity_category'].value_counts())

    print(f"\nDiet Quality:")
    print(df_complete['diet_category'].value_counts())

    print(f"\nDiabetes Risk:")
    print(df_complete['risk_category'].value_counts())

    # =========================================================================
    # 2. EXTREME CASES: Best vs Worst Behaviors
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXTREME CASES: Best vs Worst Lifestyles")
    print("=" * 70)

    # Best case: Highly active + Excellent diet
    best_cases = df_complete[
        (df_complete['activity_category'] == 'Highly Active') &
        (df_complete['diet_category'] == 'Excellent Diet')
    ]

    # Worst case: Inactive + Very poor diet
    worst_cases = df_complete[
        (df_complete['activity_category'] == 'Inactive') &
        (df_complete['diet_category'] == 'Very Poor Diet')
    ]

    print(f"\nBEST LIFESTYLE (Highly Active + Excellent Diet):")
    print(f"  N = {len(best_cases)}")
    if len(best_cases) > 0:
        print(f"  Average HOMA-IR: {best_cases['HOMA_IR'].mean():.2f}")
        print(f"  Average BMI: {best_cases['BMXBMI'].mean():.2f}")
        print(f"  Average CRP: {best_cases['LBXHSCRP'].mean():.2f}")
        print(f"  High Risk: {(best_cases['risk_category'] == 'High Risk').sum()}/{len(best_cases)} ({(best_cases['risk_category'] == 'High Risk').mean():.1%})")

    print(f"\nWORST LIFESTYLE (Inactive + Very Poor Diet):")
    print(f"  N = {len(worst_cases)}")
    if len(worst_cases) > 0:
        print(f"  Average HOMA-IR: {worst_cases['HOMA_IR'].mean():.2f}")
        print(f"  Average BMI: {worst_cases['BMXBMI'].mean():.2f}")
        print(f"  Average CRP: {worst_cases['LBXHSCRP'].mean():.2f}")
        print(f"  High Risk: {(worst_cases['risk_category'] == 'High Risk').sum()}/{len(worst_cases)} ({(worst_cases['risk_category'] == 'High Risk').mean():.1%})")

    # =========================================================================
    # 3. ACTIVITY LEVELS vs DIABETES RISK
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHYSICAL ACTIVITY vs DIABETES RISK")
    print("=" * 70)

    activity_risk = df_complete.groupby('activity_category')['HOMA_IR'].agg([
        ('N', 'count'),
        ('Mean_HOMA_IR', 'mean'),
        ('Std_HOMA_IR', 'std'),
        ('Mean_BMI', lambda x: df_complete.loc[x.index, 'BMXBMI'].mean()),
        ('High_Risk_%', lambda x: (df_complete.loc[x.index, 'risk_category'] == 'High Risk').mean() * 100)
    ])

    print("\n" + activity_risk.to_string())

    # =========================================================================
    # 4. DIET QUALITY vs DIABETES RISK
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIET QUALITY vs DIABETES RISK")
    print("=" * 70)

    diet_risk = df_complete.groupby('diet_category')['HOMA_IR'].agg([
        ('N', 'count'),
        ('Mean_HOMA_IR', 'mean'),
        ('Std_HOMA_IR', 'std'),
        ('Mean_BMI', lambda x: df_complete.loc[x.index, 'BMXBMI'].mean()),
        ('High_Risk_%', lambda x: (df_complete.loc[x.index, 'risk_category'] == 'High Risk').mean() * 100)
    ])

    print("\n" + diet_risk.to_string())

    # =========================================================================
    # 5. REAL INDIVIDUAL EXAMPLES
    # =========================================================================
    print("\n" + "=" * 70)
    print("REAL INDIVIDUAL EXAMPLES")
    print("=" * 70)

    print("\nFinding representative cases...")

    # Case 1: Low risk (highly active)
    low_risk_active = df_complete[
        (df_complete['activity_category'] == 'Highly Active') &
        (df_complete['risk_category'] == 'Low Risk')
    ].sample(n=1, random_state=42)

    # Case 2: Moderate risk (moderately active)
    mod_risk_mod_active = df_complete[
        (df_complete['activity_category'] == 'Moderately Active') &
        (df_complete['risk_category'] == 'Moderate Risk')
    ].sample(n=1, random_state=42)

    # Case 3: High risk (inactive)
    high_risk_inactive = df_complete[
        (df_complete['activity_category'] == 'Inactive') &
        (df_complete['risk_category'] == 'High Risk')
    ].sample(n=1, random_state=42)

    # Display cases
    cases = [
        ("CASE 1: Low Risk Teen", low_risk_active),
        ("CASE 2: Moderate Risk Teen", mod_risk_mod_active),
        ("CASE 3: High Risk Teen", high_risk_inactive)
    ]

    for title, case in cases:
        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)

        row = case.iloc[0]

        print(f"\nDemographics:")
        print(f"  Age: {int(row['RIDAGEYR'])} years")
        print(f"  Gender: {'Male' if row['RIAGENDR'] == 1 else 'Female'}")

        print(f"\nPhysical Activity:")
        print(f"  Inactivity Score: {row['comprehensive_inactivity_score']:.1f}")
        print(f"  Category: {row['activity_category']}")

        print(f"\nDiet:")
        print(f"  Sugar Intake: {row['DR1TSUGR']:.1f} grams/day")
        print(f"  Fiber Intake: {row['DR1TFIBE']:.1f} grams/day")
        print(f"  Sugar/Fiber Ratio: {row['DR1TSUGR']/(row['DR1TFIBE']+1):.1f}")
        print(f"  Diet Quality: {row['diet_category']}")

        print(f"\nHealth Markers:")
        print(f"  BMI: {row['BMXBMI']:.1f}")
        print(f"  CRP (Inflammation): {row['LBXHSCRP']:.2f} mg/L")
        print(f"  Fasting Glucose: {row['LBXGLU']:.1f} mg/dL")

        print(f"\nDiabetes Risk:")
        print(f"  HOMA-IR: {row['HOMA_IR']:.2f}")
        print(f"  Risk Category: {row['risk_category']}")

    # =========================================================================
    # 6. CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    from scipy.stats import pearsonr, spearmanr

    correlations = []

    vars_to_test = [
        ('Physical Inactivity', 'comprehensive_inactivity_score'),
        ('Sugar Intake', 'DR1TSUGR'),
        ('Fiber Intake', 'DR1TFIBE'),
        ('CRP (Inflammation)', 'LBXHSCRP'),
        ('BMI', 'BMXBMI')
    ]

    print(f"\n{'Variable':<25} {'Correlation with HOMA-IR':<30} {'P-value':<15} {'Significance'}")
    print("-" * 80)

    for var_name, var_col in vars_to_test:
        r, p = pearsonr(df_complete[var_col], df_complete['HOMA_IR'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        print(f"{var_name:<25} r = {r:+.3f}                      {p:<15.4f} {sig}")

    # =========================================================================
    # 7. SAVE INDIVIDUAL EXAMPLES TO FILE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING EXAMPLES")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "models" / "final" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save 10 representative cases from each risk category
    examples = []

    for risk_cat in ['Low Risk', 'Moderate Risk', 'High Risk']:
        subset = df_complete[df_complete['risk_category'] == risk_cat].sample(
            n=min(10, len(df_complete[df_complete['risk_category'] == risk_cat])),
            random_state=42
        )

        for idx, row in subset.iterrows():
            examples.append({
                'Risk_Category': risk_cat,
                'Age': int(row['RIDAGEYR']),
                'Gender': 'Male' if row['RIAGENDR'] == 1 else 'Female',
                'Inactivity_Score': row['comprehensive_inactivity_score'],
                'Activity_Level': row['activity_category'],
                'Sugar_Intake_g': row['DR1TSUGR'],
                'Fiber_Intake_g': row['DR1TFIBE'],
                'Diet_Quality': row['diet_category'],
                'BMI': row['BMXBMI'],
                'CRP_mg_L': row['LBXHSCRP'],
                'Glucose_mg_dL': row['LBXGLU'],
                'HOMA_IR': row['HOMA_IR']
            })

    examples_df = pd.DataFrame(examples)
    examples_file = output_dir / "individual_examples.csv"
    examples_df.to_csv(examples_file, index=False)

    print(f"\n[OK] Saved {len(examples)} individual examples to:")
    print(f"     {examples_file}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nKey Findings:")
    print(f"\n1. Activity Matters:")
    print(f"   - Highly Active:      Avg HOMA-IR = {activity_risk.loc['Highly Active', 'Mean_HOMA_IR']:.2f}")
    print(f"   - Inactive:           Avg HOMA-IR = {activity_risk.loc['Inactive', 'Mean_HOMA_IR']:.2f}")
    print(f"   - Difference:         {(activity_risk.loc['Inactive', 'Mean_HOMA_IR'] - activity_risk.loc['Highly Active', 'Mean_HOMA_IR']):.2f} higher")

    print(f"\n2. Diet Matters:")
    excellent_diet_homa = diet_risk.loc['Excellent Diet', 'Mean_HOMA_IR'] if 'Excellent Diet' in diet_risk.index else 0
    poor_diet_homa = diet_risk.loc['Poor Diet', 'Mean_HOMA_IR'] if 'Poor Diet' in diet_risk.index else 0

    if excellent_diet_homa > 0 and poor_diet_homa > 0:
        print(f"   - Excellent Diet:     Avg HOMA-IR = {excellent_diet_homa:.2f}")
        print(f"   - Poor Diet:          Avg HOMA-IR = {poor_diet_homa:.2f}")
        print(f"   - Difference:         {(poor_diet_homa - excellent_diet_homa):.2f} higher")

    print(f"\n3. Lifestyle Effects REAL:")
    print(f"   - Worst lifestyle -> {worst_cases['HOMA_IR'].mean():.2f} avg HOMA-IR" if len(worst_cases) > 0 else "")
    print(f"   - Best lifestyle -> {best_cases['HOMA_IR'].mean():.2f} avg HOMA-IR" if len(best_cases) > 0 else "")
    if len(worst_cases) > 0 and len(best_cases) > 0:
        print(f"   - Difference: {(worst_cases['HOMA_IR'].mean() - best_cases['HOMA_IR'].mean()):.2f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nUse these examples for science fair presentation!")
    print(f"Individual data saved: {examples_file}")


if __name__ == "__main__":
    main()
