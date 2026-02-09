"""
Biological Feature Engineering

Calculates immune and inflammatory features from NHANES biomarker data:
- NLR (Neutrophil-to-Lymphocyte Ratio) from CBC dataset
- CRP categorization from HSCRP dataset
- Composite inflammatory index combining CRP and NLR

Scientific Basis:
- CRP is a protein-based marker of systemic inflammation
- NLR is a cell-based marker of immune activation
- Both are independently elevated in insulin resistance and Type 2 Diabetes
- Using both provides two genuinely independent immune signals
"""

import pandas as pd
import numpy as np


class BiologicalFeatureCalculator:
    """Creates biological features from biomarker data."""

    def __init__(self):
        """Initialize biological feature calculator."""
        self.crp_normal = 3.0  # mg/L - normal threshold
        self.crp_high = 10.0   # mg/L - high inflammation threshold
        self.nlr_normal = 3.0  # NLR above this = elevated immune activation
        self.nlr_high = 6.0    # NLR above this = high inflammation

    def calculate_nlr(
        self,
        neutrophils: pd.Series,
        lymphocytes: pd.Series
    ) -> pd.Series:
        """
        Calculate Neutrophil-to-Lymphocyte Ratio (NLR) from CBC data.

        NLR is a clinically validated marker of systemic immune activation,
        independently associated with insulin resistance and metabolic syndrome.

        Args:
            neutrophils: Neutrophil count in 1000 cells/uL (LBDNENO)
            lymphocytes: Lymphocyte count in 1000 cells/uL (LBDLYMNO)

        Returns:
            NLR values (typical range: 0.5-8.0)
        """
        if neutrophils.isna().all() or lymphocytes.isna().all():
            print("[WARN] All neutrophil or lymphocyte values are missing")
            return pd.Series([np.nan] * len(neutrophils), index=neutrophils.index)

        # Avoid division by zero: replace zero lymphocytes with NaN
        safe_lymphocytes = lymphocytes.replace(0, np.nan)

        nlr = neutrophils / safe_lymphocytes

        # Summary statistics
        valid_values = nlr.dropna()
        if len(valid_values) > 0:
            print(f"[OK] NLR calculated: {len(valid_values)} values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

            # Distribution
            normal = (valid_values < self.nlr_normal).sum()
            elevated = ((valid_values >= self.nlr_normal) & (valid_values < self.nlr_high)).sum()
            high = (valid_values >= self.nlr_high).sum()
            print(f"  Normal (<{self.nlr_normal}): {normal} ({normal/len(valid_values)*100:.1f}%)")
            print(f"  Elevated ({self.nlr_normal}-{self.nlr_high}): {elevated} ({elevated/len(valid_values)*100:.1f}%)")
            print(f"  High (>{self.nlr_high}): {high} ({high/len(valid_values)*100:.1f}%)")

        return nlr

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
        nlr: pd.Series
    ) -> pd.Series:
        """
        Calculate composite inflammatory index from CRP and NLR.

        Combines protein-based (CRP) and cell-based (NLR) inflammation signals
        into a single score.

        Args:
            crp: CRP levels (mg/L)
            nlr: Neutrophil-to-Lymphocyte Ratio

        Returns:
            Inflammatory index (0-100 scale)
        """
        # Normalize CRP to 0-10 scale (cap at 20 mg/L)
        crp_normalized = crp.clip(0, 20) / 2

        # Normalize NLR to 0-10 scale (cap at 10)
        nlr_normalized = nlr.clip(0, 10)

        # Composite: weighted average (CRP has stronger clinical validation)
        crp_weight = 0.6
        nlr_weight = 0.4

        inflammatory_index = (
            crp_normalized * crp_weight +
            nlr_normalized * nlr_weight
        )

        # Scale to 0-100
        inflammatory_index = inflammatory_index * 10

        valid_values = inflammatory_index.dropna()
        if len(valid_values) > 0:
            print(f"[OK] Inflammatory index calculated: {len(valid_values)} values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")

        return inflammatory_index


def calculate_all_biological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all biological features from NHANES dataset.

    Requires CRP (LBXHSCRP) and CBC columns (LBDNENO, LBDLYMNO).

    Args:
        df: DataFrame with LBXHSCRP, LBDNENO, LBDLYMNO columns

    Returns:
        DataFrame with added biological features (nlr, crp_category, inflammatory_index)
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

    # Calculate NLR from CBC data
    if 'LBDNENO' in df.columns and 'LBDLYMNO' in df.columns:
        print(f"Neutrophils available: {df['LBDNENO'].notna().sum():,}")
        print(f"Lymphocytes available: {df['LBDLYMNO'].notna().sum():,}")
        df['nlr'] = calculator.calculate_nlr(df['LBDNENO'], df['LBDLYMNO'])
    else:
        print("[WARN] CBC columns (LBDNENO, LBDLYMNO) not found - NLR will be NaN")
        print("  Make sure CBC dataset is included in download")
        df['nlr'] = np.nan

    # Categorize CRP
    df['crp_category'] = calculator.categorize_crp(df['LBXHSCRP'])

    # Calculate inflammatory index (using CRP + NLR)
    df['inflammatory_index'] = calculator.calculate_inflammatory_index(
        df['LBXHSCRP'], df['nlr']
    )

    print("\n" + "=" * 70)
    print("BIOLOGICAL FEATURES COMPLETE")
    print("=" * 70)
    print(f"New features added: nlr, crp_category, inflammatory_index")
    print(f"Valid NLR values: {df['nlr'].notna().sum():,}/{len(df):,}")

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
    df = calculate_all_biological_features(df)

    # Save result
    output_file = data_file.parent / "pediatric_with_biological_features.csv"
    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved: {output_file.name}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  New columns: {df.shape[1]}")
