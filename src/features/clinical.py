"""
Clinical Feature Engineering

Calculates clinical biomarkers, primarily HOMA-IR (Homeostatic Model
Assessment of Insulin Resistance) - the primary target variable for
predicting Type 2 Diabetes risk.
"""

import pandas as pd
import numpy as np
from typing import Optional


class ClinicalFeatureCalculator:
    """Calculates clinical biomarkers from NHANES data."""

    def __init__(self):
        """Initialize clinical feature calculator."""
        self.homa_ir_normal = 2.5
        self.homa_ir_severe = 5.0

    def calculate_homa_ir(
        self,
        glucose: pd.Series,
        insulin: pd.Series
    ) -> pd.Series:
        """
        Calculate HOMA-IR: Homeostatic Model Assessment of Insulin Resistance.

        Formula: (Fasting Glucose [mg/dL] × Fasting Insulin [μU/mL]) / 405

        Interpretation:
        - Normal: <2.5
        - Insulin Resistance: 2.5-5.0
        - Severe Insulin Resistance: >5.0

        Args:
            glucose: Fasting glucose in mg/dL (LBXGLU)
            insulin: Fasting insulin in μU/mL (LBXIN)

        Returns:
            HOMA-IR values
        """
        # Validate inputs
        if glucose.isna().all() or insulin.isna().all():
            print("[WARN] All glucose or insulin values are missing")
            return pd.Series([np.nan] * len(glucose), index=glucose.index)

        # Calculate HOMA-IR
        homa_ir = (glucose * insulin) / 405

        # Validation: check physiological ranges
        valid_glucose = (glucose >= 50) & (glucose <= 300)
        valid_insulin = (insulin >= 1) & (insulin <= 200)
        valid_homa_ir = (homa_ir >= 0) & (homa_ir <= 50)

        invalid_count = (~(valid_glucose & valid_insulin & valid_homa_ir)).sum()

        if invalid_count > 0:
            print(f"[WARN] {invalid_count} HOMA-IR values outside "
                  f"physiological range (set to NaN)")
            homa_ir = homa_ir.where(valid_glucose & valid_insulin & valid_homa_ir, np.nan)

        # Summary statistics
        valid_values = homa_ir.dropna()
        if len(valid_values) > 0:
            print(f"[OK] HOMA-IR calculated: {len(valid_values)} valid values")
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
            print(f"  Median: {valid_values.median():.2f}")

            # Risk category distribution
            normal = (valid_values < self.homa_ir_normal).sum()
            moderate = ((valid_values >= self.homa_ir_normal) &
                       (valid_values < self.homa_ir_severe)).sum()
            severe = (valid_values >= self.homa_ir_severe).sum()

            print(f"  Risk Distribution:")
            print(f"    Normal (<{self.homa_ir_normal}): {normal} "
                  f"({normal/len(valid_values)*100:.1f}%)")
            print(f"    Moderate ({self.homa_ir_normal}-{self.homa_ir_severe}): "
                  f"{moderate} ({moderate/len(valid_values)*100:.1f}%)")
            print(f"    Severe (>{self.homa_ir_severe}): {severe} "
                  f"({severe/len(valid_values)*100:.1f}%)")

        return homa_ir

    def categorize_homa_ir(
        self,
        homa_ir: pd.Series
    ) -> pd.Series:
        """
        Categorize HOMA-IR into risk categories.

        Args:
            homa_ir: HOMA-IR values

        Returns:
            Risk categories (0=Normal, 1=Moderate, 2=Severe)
        """
        categories = pd.cut(
            homa_ir,
            bins=[-np.inf, self.homa_ir_normal, self.homa_ir_severe, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        )
        return categories.astype(float)

    def calculate_glucose_insulin_ratio(
        self,
        glucose: pd.Series,
        insulin: pd.Series
    ) -> pd.Series:
        """
        Calculate glucose-to-insulin ratio.

        Lower ratios indicate insulin resistance.

        Args:
            glucose: Fasting glucose (mg/dL)
            insulin: Fasting insulin (μU/mL)

        Returns:
            Glucose/Insulin ratio
        """
        ratio = glucose / (insulin + 0.1)  # +0.1 to avoid division by zero
        print(f"[OK] Glucose/Insulin ratio calculated: {ratio.notna().sum()} values")
        return ratio

    def identify_early_detection_candidates(
        self,
        glucose: pd.Series,
        homa_ir: pd.Series,
        glucose_threshold: float = 100.0
    ) -> pd.Series:
        """
        Identify individuals with insulin resistance but normal glucose.

        This is the key for early detection: catching high-risk individuals
        before they develop elevated fasting glucose.

        Args:
            glucose: Fasting glucose (mg/dL)
            homa_ir: HOMA-IR values
            glucose_threshold: Normal glucose threshold (default: 100 mg/dL)

        Returns:
            Boolean series indicating early detection candidates
        """
        normal_glucose = glucose < glucose_threshold
        high_risk = homa_ir >= self.homa_ir_normal

        early_detection = normal_glucose & high_risk

        print(f"\n[OK] Early Detection Analysis:")
        print(f"  Normal glucose (<{glucose_threshold}): {normal_glucose.sum()}")
        print(f"  High HOMA-IR (>={self.homa_ir_normal}): {high_risk.sum()}")
        print(f"  Early detection candidates: {early_detection.sum()} "
              f"({early_detection.sum()/len(early_detection)*100:.1f}%)")

        return early_detection


def calculate_all_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all clinical features from NHANES dataset.

    Args:
        df: DataFrame with LBXGLU and LBXIN columns

    Returns:
        DataFrame with added clinical features
    """
    print("\n" + "=" * 70)
    print("CLINICAL FEATURE ENGINEERING")
    print("=" * 70)

    calculator = ClinicalFeatureCalculator()

    # Check required columns
    required_cols = ['LBXGLU', 'LBXIN']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[FAIL] Missing required columns: {missing_cols}")
        return df

    print(f"\nDataset: {len(df):,} rows")
    print(f"Glucose available: {df['LBXGLU'].notna().sum():,}")
    print(f"Insulin available: {df['LBXIN'].notna().sum():,}")

    # Calculate HOMA-IR (primary target variable)
    df['HOMA_IR'] = calculator.calculate_homa_ir(df['LBXGLU'], df['LBXIN'])

    # Calculate HOMA-IR risk category (for classification)
    df['HOMA_IR_category'] = calculator.categorize_homa_ir(df['HOMA_IR'])

    # Calculate glucose/insulin ratio
    df['glucose_insulin_ratio'] = calculator.calculate_glucose_insulin_ratio(
        df['LBXGLU'], df['LBXIN']
    )

    # Identify early detection candidates
    df['early_detection_candidate'] = calculator.identify_early_detection_candidates(
        df['LBXGLU'], df['HOMA_IR']
    )

    print("\n" + "=" * 70)
    print("CLINICAL FEATURES COMPLETE")
    print("=" * 70)
    print(f"New features added: HOMA_IR, HOMA_IR_category, "
          f"glucose_insulin_ratio, early_detection_candidate")
    print(f"Valid HOMA-IR values: {df['HOMA_IR'].notna().sum():,}/{len(df):,}")

    return df


if __name__ == "__main__":
    # Test with multi-cycle pediatric data
    print("THE PEDIATRIC SENTINEL - Clinical Feature Engineering")
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

    # Calculate clinical features
    df = calculate_all_clinical_features(df)

    # Save result
    output_file = data_file.parent / "pediatric_with_clinical_features.csv"
    df.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved: {output_file.name}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  New columns: {df.shape[1]}")
