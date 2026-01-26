"""
NHANES Data Merger

Merges multiple NHANES datasets on SEQN (sequence number).
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.constants import (
    PROCESSED_DATA_DIR,
    MERGED_DATA_FILE
)


class NHANESMerger:
    """Merges NHANES datasets on SEQN."""

    def __init__(self, merge_key: str = 'SEQN'):
        """
        Initialize merger.

        Args:
            merge_key: Column name to merge on (default: SEQN)
        """
        self.merge_key = merge_key

    def merge_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        how: str = 'left',
        base_dataset: str = 'P_DEMO'
    ) -> pd.DataFrame:
        """
        Merge all datasets on SEQN.

        Args:
            datasets: Dictionary mapping dataset codes to DataFrames
            how: Type of merge ('left', 'inner', 'outer')
            base_dataset: Dataset to use as base (default: P_DEMO)

        Returns:
            Merged DataFrame
        """
        print("\n" + "=" * 70)
        print(f"MERGING {len(datasets)} DATASETS ON {self.merge_key}")
        print("=" * 70)

        if not datasets:
            print("[FAIL] No datasets provided")
            return pd.DataFrame()

        # Start with base dataset (demographics)
        if base_dataset not in datasets:
            base_dataset = list(datasets.keys())[0]
            print(f"[WARN] Using {base_dataset} as base dataset")

        merged = datasets[base_dataset].copy()
        print(f"\nBase dataset: {base_dataset}")
        print(f"  Rows: {len(merged):,}")
        print(f"  Columns: {merged.shape[1]}")

        # Merge other datasets
        merge_stats = {}

        for code, df in datasets.items():
            if code == base_dataset:
                continue

            # Check if merge key exists
            if self.merge_key not in df.columns:
                print(f"\n[FAIL] {code}: Missing {self.merge_key} column - skipping")
                continue

            # Perform merge
            before_cols = merged.shape[1]
            before_rows = len(merged)

            merged = merged.merge(
                df,
                on=self.merge_key,
                how=how,
                suffixes=('', f'_{code}')
            )

            after_cols = merged.shape[1]
            after_rows = len(merged)

            # Calculate merge statistics
            new_cols = after_cols - before_cols
            matched = df[self.merge_key].isin(merged[self.merge_key]).sum()
            match_rate = (matched / len(df)) * 100 if len(df) > 0 else 0

            merge_stats[code] = {
                'matched': matched,
                'total': len(df),
                'match_rate': match_rate,
                'new_columns': new_cols
            }

            print(f"\n{code}:")
            print(f"  Matched: {matched:,}/{len(df):,} ({match_rate:.1f}%)")
            print(f"  Added columns: {new_cols}")
            print(f"  Merged shape: {after_rows:,} rows × {after_cols} columns")

        # Final summary
        print("\n" + "=" * 70)
        print("MERGE SUMMARY")
        print("=" * 70)
        print(f"Final dataset: {len(merged):,} rows × {merged.shape[1]} columns")

        # Check SEQN uniqueness
        seqn_duplicates = merged[self.merge_key].duplicated().sum()
        if seqn_duplicates > 0:
            print(f"[WARN] SEQN duplicates: {seqn_duplicates}")
        else:
            print(f"[OK] All SEQN values are unique")

        # Missing values summary
        total_values = merged.shape[0] * merged.shape[1]
        missing_values = merged.isnull().sum().sum()
        missing_pct = (missing_values / total_values) * 100
        print(f"Missing values: {missing_values:,} ({missing_pct:.1f}%)")

        return merged

    def calculate_merge_statistics(
        self,
        merged: pd.DataFrame,
        original_datasets: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Calculate detailed merge statistics.

        Args:
            merged: Merged DataFrame
            original_datasets: Original datasets before merging

        Returns:
            Dictionary with merge statistics
        """
        stats = {
            'final_rows': len(merged),
            'final_columns': merged.shape[1],
            'unique_seqn': merged[self.merge_key].nunique(),
            'total_missing': merged.isnull().sum().sum(),
            'missing_pct': (merged.isnull().sum().sum() /
                           (merged.shape[0] * merged.shape[1])) * 100,
            'datasets_merged': len(original_datasets),
        }

        # Per-dataset coverage
        coverage = {}
        for code, df in original_datasets.items():
            if self.merge_key in df.columns:
                matched = df[self.merge_key].isin(merged[self.merge_key]).sum()
                coverage[code] = {
                    'matched': matched,
                    'total': len(df),
                    'rate': (matched / len(df)) * 100 if len(df) > 0 else 0
                }

        stats['coverage'] = coverage

        return stats

    def save_merged_data(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save merged dataset to CSV.

        Args:
            df: Merged DataFrame
            output_path: Path to save file (default: MERGED_DATA_FILE)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = MERGED_DATA_FILE

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"\n[OK] Saved merged data to {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")

        return output_path

    def merge_and_save(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_path: Optional[Path] = None,
        how: str = 'left'
    ) -> pd.DataFrame:
        """
        Merge datasets and save to file.

        Args:
            datasets: Dictionary of datasets
            output_path: Path to save merged data
            how: Type of merge

        Returns:
            Merged DataFrame
        """
        # Merge datasets
        merged = self.merge_datasets(datasets, how=how)

        # Save to file
        if output_path is None:
            output_path = MERGED_DATA_FILE

        self.save_merged_data(merged, output_path)

        # Calculate and print statistics
        stats = self.calculate_merge_statistics(merged, datasets)

        print("\n" + "=" * 70)
        print("MERGE STATISTICS")
        print("=" * 70)
        print(f"Datasets merged: {stats['datasets_merged']}")
        print(f"Final rows: {stats['final_rows']:,}")
        print(f"Final columns: {stats['final_columns']}")
        print(f"Unique SEQN: {stats['unique_seqn']:,}")
        print(f"Missing values: {stats['total_missing']:,} ({stats['missing_pct']:.1f}%)")

        print("\nDataset Coverage:")
        for code, cov in stats['coverage'].items():
            print(f"  {code}: {cov['matched']:,}/{cov['total']:,} ({cov['rate']:.1f}%)")

        print("=" * 70)

        return merged


# Convenience function
def merge_datasets(
    datasets: Dict[str, pd.DataFrame],
    save: bool = True
) -> pd.DataFrame:
    """Merge all datasets."""
    merger = NHANESMerger()

    if save:
        return merger.merge_and_save(datasets)
    else:
        return merger.merge_datasets(datasets)


if __name__ == "__main__":
    # Test: Would need loaded and cleaned datasets
    print("THE PEDIATRIC SENTINEL - NHANES Data Merger")
    print("=" * 70)
    print("Usage: merger.merge_and_save(cleaned_datasets)")
