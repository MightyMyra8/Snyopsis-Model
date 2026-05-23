("""
NHANES Data Loader

Loads NHANES .xpt (SAS transport) files into pandas DataFrames.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.constants import (
    RAW_DATA_DIR,
    DATASETS,
    ENHANCED_DATASETS,
    ALL_DATASETS)
)


class NHANESLoader:
    """Loads NHANES .xpt files into pandas DataFrames."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing .xpt files. Defaults to RAW_DATA_DIR.
        """
        self.data_dir = data_dir or RAW_DATA_DIR

    def load_xpt_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a single .xpt file into a DataFrame.

        Args:
            file_path: Path to .xpt file

        Returns:
            DataFrame, or None if loading failed
        """
        if not file_path.exists():
            print(f"[FAIL] File not found: {file_path}")
            return None

        try:
            # Suppress DtypeWarnings for mixed types
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_sas(file_path, format='xport', encoding='utf-8')

            print(f"[OK] Loaded {file_path.name}: {df.shape[0]} rows × {df.shape[1]} columns")
            return df

        except Exception as e:
            print(f"[FAIL] Failed to load {file_path.name}: {e}")
            return None

    def load_dataset(
        self,
        dataset_code: str,
        datasets_dict: Dict[str, str]
    ) -> Optional[pd.DataFrame]:
        """
        Load a dataset by code.

        Args:
            dataset_code: Dataset code (e.g., "P_DEMO")
            datasets_dict: Dictionary mapping codes to filenames

        Returns:
            DataFrame, or None if loading failed
        """
        if dataset_code not in datasets_dict:
            print(f"[FAIL] Unknown dataset code: {dataset_code}")
            return None

        filename = datasets_dict[dataset_code]
        file_path = self.data_dir / filename

        return self.load_xpt_file(file_path)

    def load_core_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 5 core NHANES datasets.

        Returns:
            Dictionary mapping dataset codes to DataFrames
        """
        print("\n" + "=" * 70)
        print("LOADING CORE DATASETS (5 datasets)")
        print("=" * 70)

        results = {}
        for code in DATASETS.keys():
            df = self.load_dataset(code, DATASETS)
            if df is not None:
                results[code] = df

        self._print_summary(results, len(DATASETS))
        return results

    def load_enhanced_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 7 enhanced NHANES datasets.

        Returns:
            Dictionary mapping dataset codes to DataFrames
        """
        print("\n" + "=" * 70)
        print("LOADING ENHANCED DATASETS (7 datasets)")
        print("=" * 70)

        results = {}
        for code in ENHANCED_DATASETS.keys():
            df = self.load_dataset(code, ENHANCED_DATASETS)
            if df is not None:
                results[code] = df

        self._print_summary(results, len(ENHANCED_DATASETS))
        return results

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 12 NHANES datasets (5 core + 7 enhanced).

        Returns:
            Dictionary mapping dataset codes to DataFrames
        """
        print("\n" + "=" * 70)
        print("LOADING ALL DATASETS (12 total)")
        print("=" * 70)

        results = {}

        # Load core datasets
        for code in DATASETS.keys():
            df = self.load_dataset(code, ALL_DATASETS)
            if df is not None:
                results[code] = df

        # Load enhanced datasets
        for code in ENHANCED_DATASETS.keys():
            df = self.load_dataset(code, ALL_DATASETS)
            if df is not None:
                results[code] = df

        self._print_summary(results, len(ALL_DATASETS))
        return results

    def get_dataset_info(self, df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """
        Get information about a dataset.

        Args:
            df: DataFrame
            name: Dataset name for display

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "name": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "duplicates": df.duplicated().sum(),
        }

        # Check if SEQN column exists
        if 'SEQN' in df.columns:
            info["unique_seqn"] = df['SEQN'].nunique()
            info["seqn_duplicates"] = df['SEQN'].duplicated().sum()

        return info

    def print_dataset_info(self, datasets: Dict[str, pd.DataFrame]):
        """
        Print detailed information about loaded datasets.

        Args:
            datasets: Dictionary mapping codes to DataFrames
        """
        print("\n" + "=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)

        for code, df in datasets.items():
            info = self.get_dataset_info(df, code)

            print(f"\n{code}:")
            print(f"  Rows: {info['rows']:,}")
            print(f"  Columns: {info['columns']}")
            print(f"  Memory: {info['memory_mb']:.2f} MB")
            print(f"  Missing: {info['missing_pct']:.1f}%")
            print(f"  Duplicates: {info['duplicates']}")

            if 'unique_seqn' in info:
                print(f"  Unique SEQN: {info['unique_seqn']:,}")
                if info['seqn_duplicates'] > 0:
                    print(f"  ⚠ SEQN duplicates: {info['seqn_duplicates']}")

        print("\n" + "=" * 70)

    def get_common_columns(self, datasets: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Get columns that appear in all datasets.

        Args:
            datasets: Dictionary mapping codes to DataFrames

        Returns:
            List of common column names
        """
        if not datasets:
            return []

        # Start with columns from first dataset
        common = set(list(datasets.values())[0].columns)

        # Intersect with columns from all other datasets
        for df in datasets.values():
            common &= set(df.columns)

        return sorted(list(common))

    @staticmethod
    def _print_summary(results: Dict[str, pd.DataFrame], total: int):
        """Print loading summary."""
        success_count = len(results)
        failed_count = total - success_count

        total_rows = sum(df.shape[0] for df in results.values())
        total_cols = sum(df.shape[1] for df in results.values())

        print("\n" + "-" * 70)
        print(f"Loading Summary: {success_count}/{total} datasets loaded")
        print(f"Total rows: {total_rows:,}")
        print(f"Total columns: {total_cols}")
        if failed_count > 0:
            print(f"Failed to load: {failed_count} datasets")
        print("-" * 70)


# Convenience functions
def load_core() -> Dict[str, pd.DataFrame]:
    """Load core 5 datasets."""
    loader = NHANESLoader()
    return loader.load_core_datasets()


def load_enhanced() -> Dict[str, pd.DataFrame]:
    """Load enhanced 7 datasets."""
    loader = NHANESLoader()
    return loader.load_enhanced_datasets()


def load_all() -> Dict[str, pd.DataFrame]:
    """Load all 12 datasets."""
    loader = NHANESLoader()
    return loader.load_all_datasets()


if __name__ == "__main__":
    # Test: Load all datasets
    print("THE PEDIATRIC SENTINEL - NHANES Data Loader")
    print("=" * 70)

    loader = NHANESLoader()

    # Load all datasets
    datasets = loader.load_all_datasets()

    # Print detailed info
    loader.print_dataset_info(datasets)

    # Check common columns (should include SEQN)
    common = loader.get_common_columns(datasets)
    print(f"\nCommon columns across all datasets: {common}")
