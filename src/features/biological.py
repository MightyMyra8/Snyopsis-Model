"""
Biological Feature Engineering

Creates synthetic miRNA-155 layer from CRP (C-Reactive Protein) levels.
Since real miRNA data is unavailable in NHANES, we create a proxy based on
the well-documented correlation between CRP and miRNA-155 in inflammation.

Scientific Basis:
- CRP is a marker of systemic inflammation
- miRNA-155 is upregulated in inflammatory conditions
- Both are elevated in insulin resistance and Type 2 Diabetes
"""

import pandas as pd
import numpy as np
from typing import Literal


class BiologicalFeatureCalculator:
    """Creates synthetic biological features from biomarker data."""

    def __init__(self):
        """Initialize biological feature calculator."""
        self.crp_normal = 3.0  # mg/L - normal threshold
        self.crp_high = 10.0   # mg/L - high inflammation threshold

    def create_synthetic_mirna(
        self,
        crp: pd.Series,
        method: Literal["log_linear", "sigmoid", "power"] = "log_linear"
    ) -> pd.Series:
        """
        Generate synthetic miRNA-155 proxy from CRP levels.

        Based on literature showing correlation between CRP and miRNA-155
        in inflammatory conditions and insulin resistance.

        Methods:
        - log_linear: np.log1p(crp) * weight_factor
          Best for capturing exponential relationship
        - sigmoid: 1 / (1 + np.exp(-k * (crp - threshold)))
          Best for modeling threshold effects
        - power: crp ** exponent
          Best for simple power-law relationship

        Args:
            crp: High-sensitivity CRP in mg/L (LBXHSCRP)
            method: Calculation method

        Returns:
            Synthetic miRNA-155 values (normalized to 0-10 scale)
        """
        if crp.isna().all():
            print("[WARN] All CRP values are missing")
            return pd.Series([np.nan] * len(crp), index=crp.index)

        if method == "log_linear":
            # Logarithmic transformation with weight factor
            # Based on TODAY Study proxy weight of 1.2
            weight_factor = 1.2
            mirna = np.log1p(crp) * weight_factor

        elif method == "sigmoid":
            # Sigmoid transformation around CRP threshold
            k = 0.5  # Steepness parameter
            threshold = self.crp_normal
            mirna = 10 / (1 + np.exp(-k * (crp - threshold)))

        elif method == "power":
            # Power-law transformation
            exponent = 0.6
            mirna = crp ** exponent

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize to 0-10 scale
        valid_mirna = mirna.dropna()
        if len(valid_mirna) > 0:
            min_val = valid_mirna.min()
            max_val = valid_mirna.max()
            if max_val > min_val:
                mirna = ((mirna - min_val) / (max_val - min_val)) * 10

        # Summary statistics
        valid_values = mirna.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Synthetic miRNA-155 calculated ({method}): "
                  f"{len(valid_values)} values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

        return mirna

    def categorize_crp(
        self,
        crp: pd.Series
    ) -> pd.Series:
        """
        Categorize CRP levels into inflammation risk categories.

        Categories:
        - 0: Normal (<3 mg/L)
        - 1: Moderate (3-10 mg/L)
        - 2: High (>10 mg/L)

        Args:
            crp: CRP levels (mg/L)

        Returns:
            CRP risk categories
        """
        categories = pd.cut(
            crp,
            bins=[-np.inf, self.crp_normal, self.crp_high, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        )

        valid_categories = categories.dropna()
        if len(valid_categories) > 0:
            normal = (valid_categories == 0).sum()
            moderate = (valid_categories == 1).sum()
            high = (valid_categories == 2).sum()

            print(f"[OK] CRP categorized: {len(valid_categories)} values")
            print(f"  Normal (<{self.crp_normal}): {normal} "
                  f"({normal/len(valid_categories)*100:.1f}%)")
            print(f"  Moderate ({self.crp_normal}-{self.crp_high}): {moderate} "
                  f"({moderate/len(valid_categories)*100:.1f}%)")
            print(f"  High (>{self.crp_high}): {high} "
                  f"({high/len(valid_categories)*100:.1f}%)")

        return categories.astype(float)

    def calculate_inflammatory_index(
        self,
        crp: pd.Series,
        synthetic_mirna: pd.Series
    ) -> pd.Series:
        """
        Calculate composite inflammatory index from CRP and synthetic miRNA.

        Combines CRP and miRNA signals into single inflammation score.

        Args:
            crp: CRP levels (mg/L)
            synthetic_mirna: Synthetic miRNA-155 values

        Returns:
            Inflammatory index (0-100 scale)
        """
        # Normalize CRP to 0-10 scale
        crp_normalized = crp.clip(0, 20) / 2  # Cap at 20 mg/L

        # Composite: weighted average
        crp_weight = 0.6  # CRP has stronger clinical validation
        mirna_weight = 0.4

        inflammatory_index = (
            crp_normalized * crp_weight +
            synthetic_mirna * mirna_weight
        )

        # Scale to 0-100
        inflammatory_index = inflammatory_index * 10

        valid_values = inflammatory_index.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Inflammatory index calculated: {len(valid_values)} values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")

        return inflammatory_index


def calculate_all_biological_features(
    df: pd.DataFrame,
    mirna_method: Literal["log_linear", "sigmoid", "power"] = "log_linear"
) -> pd.DataFrame:
    """
    Calculate all biological features from NHANES dataset.

    Args:
        df: DataFrame with LBXHSCRP column
        mirna_method: Method for synthetic miRNA calculation

    Returns:
        DataFrame with added biological features
    """
    print("\n" + "=" * 70)
    print("BIOLOGICAL FEATURE ENGINEERING")
    print("=" * 70)

    calculator = BiologicalFeatureCalculator()

    # Check required columns
    if 'LBXHSCRP' not in df.columns:
        print("[FAIL] Missing required column: LBXHSCRP (CRP)")
        return df

    print(f"\nDataset: {len(df):,} rows")
    print(f"CRP available: {df['LBXHSCRP'].notna().sum():,}")

    # Create synthetic miRNA-155
    df['synthetic_mirna155'] = calculator.create_synthetic_mirna(
        df['LBXHSCRP'], method=mirna_method
    )

    # Categorize CRP
    df['crp_category'] = calculator.categorize_crp(df['LBXHSCRP'])

    # Calculate inflammatory index
    df['inflammatory_index'] = calculator.calculate_inflammatory_index(
        df['LBXHSCRP'], df['synthetic_mirna155']
    )

    print("\n" + "=" * 70)
    print("BIOLOGICAL FEATURES COMPLETE")
    print("=" * 70)
    print(f"New features added: synthetic_mirna155, crp_category, inflammatory_index")
    print(f"Valid synthetic miRNA values: "
          f"{df['synthetic_mirna155'].notna().sum():,}/{len(df):,}")

    return df


if __name__ == "__main__":
    # Test with multi-cycle pediatric data
    print("THE PEDIATRIC SENTINEL - Biological Feature Engineering")
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

    # Calculate biological features
    df = calculate_all_biological_features(df, mirna_method="log_linear")

    # Save result
    output_file = data_file.parent / "pediatric_with_biological_features.csv"
    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved: {output_file.name}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  New columns: {df.shape[1]}")
