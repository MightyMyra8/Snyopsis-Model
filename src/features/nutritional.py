"""
Nutritional Feature Engineering

Calculates dietary features from NHANES Dietary Interview data,
including Nutritional Stress Index (NSI) based on sugar/fiber ratio.

High sugar + low fiber intake creates metabolic stress and increases
diabetes risk.
"""

import pandas as pd
import numpy as np


class NutritionalFeatureCalculator:
    """Calculates nutritional features from dietary data."""

    def __init__(self):
        """Initialize nutritional feature calculator."""
        # AHA guidelines for added sugars (grams/day)
        self.sugar_limit_female = 25.0  # ~6 teaspoons
        self.sugar_limit_male = 36.0    # ~9 teaspoons

        # Dietary Guidelines for fiber (grams/day)
        self.fiber_target = 25.0  # For adolescents

    def calculate_nutritional_stress_index(
        self,
        sugar: pd.Series,
        fiber: pd.Series
    ) -> pd.Series:
        """
        Calculate Nutritional Stress Index based on sugar/fiber ratio.

        High NSI indicates poor diet quality (high sugar, low fiber).

        Formula: (sugar / (fiber + 1)) × 10
        Normalized to 0-100 scale.

        Args:
            sugar: Total sugars (g/day) - DR1TSUGR
            fiber: Dietary fiber (g/day) - DR1TFIBE

        Returns:
            Nutritional Stress Index (0-100 scale)
        """
        if sugar.isna().all() or fiber.isna().all():
            print("[WARN] All sugar or fiber values are missing")
            return pd.Series([np.nan] * len(sugar), index=sugar.index)

        # Calculate ratio (+1 to avoid division by zero)
        nsi = (sugar / (fiber + 1)) * 10

        # Clip to 0-100 range
        nsi = np.clip(nsi, 0, 100)

        # Summary statistics
        valid_values = nsi.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Nutritional Stress Index calculated: {len(valid_values)} values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

            # Risk categories
            low_risk = (valid_values < 33).sum()
            medium_risk = ((valid_values >= 33) & (valid_values < 66)).sum()
            high_risk = (valid_values >= 66).sum()

            print(f"  Risk Distribution:")
            print(f"    Low risk (<33): {low_risk} "
                  f"({low_risk/len(valid_values)*100:.1f}%)")
            print(f"    Medium risk (33-66): {medium_risk} "
                  f"({medium_risk/len(valid_values)*100:.1f}%)")
            print(f"    High risk (>66): {high_risk} "
                  f"({high_risk/len(valid_values)*100:.1f}%)")

        return nsi

    def categorize_nsi(
        self,
        nsi: pd.Series
    ) -> pd.Series:
        """
        Categorize NSI into risk categories.

        Categories:
        - 0: Low risk (<33)
        - 1: Medium risk (33-66)
        - 2: High risk (>66)

        Args:
            nsi: Nutritional Stress Index values

        Returns:
            NSI risk categories
        """
        categories = pd.cut(
            nsi,
            bins=[-np.inf, 33, 66, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        )
        return categories.astype(float)

    def assess_sugar_intake(
        self,
        sugar: pd.Series,
        gender: pd.Series
    ) -> pd.Series:
        """
        Assess sugar intake against AHA guidelines.

        AHA recommendations:
        - Females: <25g/day
        - Males: <36g/day

        Args:
            sugar: Total sugars (g/day)
            gender: Gender (1=Male, 2=Female)

        Returns:
            Boolean series indicating excessive sugar intake
        """
        # Create gender-specific thresholds
        thresholds = gender.map({1: self.sugar_limit_male, 2: self.sugar_limit_female})

        excessive_sugar = sugar > thresholds

        valid_excessive = excessive_sugar.dropna()
        if len(valid_excessive) > 0:
            count = valid_excessive.sum()
            print(f"[OK] Excessive sugar intake: {count} "
                  f"({count/len(valid_excessive)*100:.1f}%)")

        return excessive_sugar

    def assess_fiber_intake(
        self,
        fiber: pd.Series
    ) -> pd.Series:
        """
        Assess fiber intake against Dietary Guidelines.

        Target: 25g/day for adolescents

        Args:
            fiber: Dietary fiber (g/day)

        Returns:
            Boolean series indicating inadequate fiber intake
        """
        inadequate_fiber = fiber < self.fiber_target

        valid_inadequate = inadequate_fiber.dropna()
        if len(valid_inadequate) > 0:
            count = valid_inadequate.sum()
            print(f"[OK] Inadequate fiber intake: {count} "
                  f"({count/len(valid_inadequate)*100:.1f}%)")

        return inadequate_fiber

    def calculate_macronutrient_ratios(
        self,
        carbs: pd.Series,
        protein: pd.Series,
        fat: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate macronutrient ratios from total intake.

        Args:
            carbs: Total carbohydrates (g)
            protein: Total protein (g)
            fat: Total fat (g)

        Returns:
            DataFrame with carb%, protein%, fat%
        """
        total = carbs + protein + fat

        ratios = pd.DataFrame({
            'carb_percent': (carbs / total * 100).fillna(0),
            'protein_percent': (protein / total * 100).fillna(0),
            'fat_percent': (fat / total * 100).fillna(0)
        })

        valid_ratios = ratios.dropna()
        if len(valid_ratios) > 0:
            print(f"[OK] Macronutrient ratios calculated: {len(valid_ratios)} values")
            print(f"  Mean Carb%: {ratios['carb_percent'].mean():.1f}%")
            print(f"  Mean Protein%: {ratios['protein_percent'].mean():.1f}%")
            print(f"  Mean Fat%: {ratios['fat_percent'].mean():.1f}%")

        return ratios


def calculate_all_nutritional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all nutritional features from NHANES dataset.

    Args:
        df: DataFrame with dietary columns (DR1TSUGR, DR1TFIBE, etc.)

    Returns:
        DataFrame with added nutritional features
    """
    print("\n" + "=" * 70)
    print("NUTRITIONAL FEATURE ENGINEERING")
    print("=" * 70)

    calculator = NutritionalFeatureCalculator()

    # Check required columns
    required_cols = ['DR1TSUGR', 'DR1TFIBE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[FAIL] Missing required columns: {missing_cols}")
        return df

    print(f"\nDataset: {len(df):,} rows")
    print(f"Sugar data available: {df['DR1TSUGR'].notna().sum():,}")
    print(f"Fiber data available: {df['DR1TFIBE'].notna().sum():,}")

    # Calculate Nutritional Stress Index
    df['nutritional_stress_index'] = calculator.calculate_nutritional_stress_index(
        df['DR1TSUGR'], df['DR1TFIBE']
    )

    # Categorize NSI
    df['nsi_category'] = calculator.categorize_nsi(df['nutritional_stress_index'])

    # Assess sugar intake (needs gender)
    if 'RIAGENDR' in df.columns:
        df['excessive_sugar'] = calculator.assess_sugar_intake(
            df['DR1TSUGR'], df['RIAGENDR']
        ).astype(float)
    else:
        print("[WARN] Gender column (RIAGENDR) not found - skipping sugar assessment")

    # Assess fiber intake
    df['inadequate_fiber'] = calculator.assess_fiber_intake(df['DR1TFIBE']).astype(float)

    # Calculate macronutrient ratios if available
    macro_cols = ['DR1TCARB', 'DR1TPROT', 'DR1TTFAT']
    if all(col in df.columns for col in macro_cols):
        ratios = calculator.calculate_macronutrient_ratios(
            df['DR1TCARB'], df['DR1TPROT'], df['DR1TTFAT']
        )
        df['carb_percent'] = ratios['carb_percent']
        df['protein_percent'] = ratios['protein_percent']
        df['fat_percent'] = ratios['fat_percent']
    else:
        print(f"[WARN] Macronutrient columns not all found - skipping ratio calculation")

    print("\n" + "=" * 70)
    print("NUTRITIONAL FEATURES COMPLETE")
    print("=" * 70)
    print(f"Core features: nutritional_stress_index, nsi_category, "
          f"excessive_sugar, inadequate_fiber")
    print(f"Valid NSI values: "
          f"{df['nutritional_stress_index'].notna().sum():,}/{len(df):,}")

    return df


if __name__ == "__main__":
    # Test with multi-cycle pediatric data
    print("THE PEDIATRIC SENTINEL - Nutritional Feature Engineering")
    print("=" * 70)

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # Load multi-cycle pediatric data
    data_file = Path(__file__).parent.parent.parent / "data" / "processed" / "multi_cycle_pediatric.csv"

    if not data_file.exists():
        print(f"[FAIL] Data file not found: {data_file}")
        sys.exit(1)

    df = pd.read_csv(data_file)
    print(f"\n[OK] Loaded: {data_file.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")

    # Calculate nutritional features
    df = calculate_all_nutritional_features(df)

    # Save result
    output_file = data_file.parent / "pediatric_with_nutritional_features.csv"
    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved: {output_file.name}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  New columns: {df.shape[1]}")
