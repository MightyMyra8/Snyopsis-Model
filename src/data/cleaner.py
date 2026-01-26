"""
NHANES Data Cleaner

Cleans NHANES datasets: filters age range, handles missing values, removes outliers.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.constants import (
    MIN_AGE,
    MAX_AGE,
    MAX_MISSING_PCT,
    IQR_MULTIPLIER
)


class NHANESCleaner:
    """Cleans NHANES datasets."""

    def __init__(self, min_age: int = MIN_AGE, max_age: int = MAX_AGE):
        """
        Initialize cleaner.

        Args:
            min_age: Minimum age (inclusive)
            max_age: Maximum age (inclusive)
        """
        self.min_age = min_age
        self.max_age = max_age

    def filter_age_range(
        self,
        df: pd.DataFrame,
        age_column: str = 'RIDAGEYR'
    ) -> pd.DataFrame:
        """
        Filter DataFrame to specified age range.

        Args:
            df: DataFrame with age column
            age_column: Name of age column (default: RIDAGEYR)

        Returns:
            Filtered DataFrame
        """
        if age_column not in df.columns:
            print(f"[WARN] Age column '{age_column}' not found")
            return df

        initial_rows = len(df)
        df_filtered = df[
            (df[age_column] >= self.min_age) &
            (df[age_column] <= self.max_age)
        ].copy()

        removed = initial_rows - len(df_filtered)
        print(f"[OK] Age filter ({self.min_age}-{self.max_age}): "
              f"{len(df_filtered):,} rows (removed {removed:,})")

        return df_filtered

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'drop',
        threshold: float = MAX_MISSING_PCT,
        critical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame
            strategy: Strategy ('drop', 'drop_columns', 'keep')
            threshold: Max missing % for columns (if drop_columns)
            critical_columns: Columns that cannot have missing values

        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)

        # Drop rows with missing critical columns
        if critical_columns:
            df = df.dropna(subset=critical_columns)
            removed = initial_rows - len(df)
            if removed > 0:
                print(f"[OK] Dropped {removed:,} rows with missing critical values")

        # Handle other missing values
        if strategy == 'drop_columns':
            # Drop columns with too many missing values
            missing_pct = (df.isnull().sum() / len(df)) * 100
            cols_to_drop = missing_pct[missing_pct > threshold * 100].index.tolist()

            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                print(f"[OK] Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing")

        elif strategy == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
            removed = initial_rows - len(df)
            print(f"[OK] Dropped {removed:,} rows with missing values")

        # Report final missing values
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
            print(f"[WARN] Remaining missing values: {total_missing:,} ({missing_pct:.1f}%)")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        multiplier: float = IQR_MULTIPLIER
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR method.

        Args:
            df: DataFrame
            columns: Columns to check for outliers
            method: Method ('iqr', 'zscore')
            multiplier: IQR multiplier (default: 1.5)

        Returns:
            DataFrame with outliers removed
        """
        initial_rows = len(df)

        if method == 'iqr':
            # IQR method
            for col in columns:
                if col not in df.columns:
                    continue

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                df = df[
                    (df[col] >= lower_bound) &
                    (df[col] <= upper_bound)
                ]

        elif method == 'zscore':
            # Z-score method
            for col in columns:
                if col not in df.columns:
                    continue

                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]  # Keep values within 3 standard deviations

        removed = initial_rows - len(df)
        if removed > 0:
            print(f"[OK] Removed {removed:,} outliers ({method} method)")

        return df

    def standardize_columns(
        self,
        df: pd.DataFrame,
        lowercase: bool = False
    ) -> pd.DataFrame:
        """
        Standardize column names.

        Args:
            df: DataFrame
            lowercase: Convert to lowercase

        Returns:
            DataFrame with standardized columns
        """
        if lowercase:
            df.columns = df.columns.str.lower()
            print("[OK] Converted column names to lowercase")

        # Remove leading/trailing whitespace
        df.columns = df.columns.str.strip()

        return df

    def clean_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean demographics dataset (P_DEMO).

        Args:
            df: Demographics DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning demographics (P_DEMO)...")

        # Filter age range
        df = self.filter_age_range(df, 'RIDAGEYR')

        # Keep only essential columns if memory is a concern
        essential_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'RIDRETH3']
        available_cols = [col for col in essential_cols if col in df.columns]
        # Note: Not dropping other columns yet, just noting essentials

        # Handle missing BMI (important for analysis)
        if 'BMXBMI' in df.columns:
            missing_bmi = df['BMXBMI'].isnull().sum()
            if missing_bmi > 0:
                print(f"[WARN] {missing_bmi:,} rows missing BMI")

        return df

    def clean_biochemistry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean biochemistry dataset (P_BIOPRO).

        Args:
            df: Biochemistry DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning biochemistry (P_BIOPRO)...")

        # Critical columns for HOMA-IR
        critical = ['SEQN', 'LBXGLU', 'LBXIN']
        available_critical = [col for col in critical if col in df.columns]

        if len(available_critical) == len(critical):
            initial_rows = len(df)
            df = df.dropna(subset=['LBXGLU', 'LBXIN'])
            removed = initial_rows - len(df)
            if removed > 0:
                print(f"[OK] Dropped {removed:,} rows with missing glucose/insulin")

        # Remove physiologically impossible values
        if 'LBXGLU' in df.columns:
            # Glucose: 0-600 mg/dL is realistic range
            df = df[(df['LBXGLU'] > 0) & (df['LBXGLU'] < 600)]

        if 'LBXIN' in df.columns:
            # Insulin: 0-300 μU/mL is realistic range
            df = df[(df['LBXIN'] > 0) & (df['LBXIN'] < 300)]

        return df

    def clean_all_datasets(
        self,
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Clean all datasets.

        Args:
            datasets: Dictionary mapping dataset codes to DataFrames

        Returns:
            Dictionary of cleaned DataFrames
        """
        print("\n" + "=" * 70)
        print("CLEANING ALL DATASETS")
        print("=" * 70)

        cleaned = {}

        for code, df in datasets.items():
            print(f"\nProcessing {code}...")

            if code == 'P_DEMO':
                cleaned[code] = self.clean_demographics(df)
            elif code == 'P_BIOPRO':
                cleaned[code] = self.clean_biochemistry(df)
            else:
                # Generic cleaning for other datasets
                # Just ensure SEQN exists
                if 'SEQN' in df.columns:
                    cleaned[code] = df.copy()
                else:
                    print(f"[WARN] {code} missing SEQN column - skipping")

        print("\n" + "=" * 70)
        print(f"Cleaned {len(cleaned)}/{len(datasets)} datasets")
        print("=" * 70)

        return cleaned


# Convenience function
def clean_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Clean all datasets."""
    cleaner = NHANESCleaner()
    return cleaner.clean_all_datasets(datasets)


if __name__ == "__main__":
    # Test: Would need loaded datasets
    print("THE PEDIATRIC SENTINEL - NHANES Data Cleaner")
    print("=" * 70)
    print("Usage: cleaner.clean_all_datasets(loaded_datasets)")
