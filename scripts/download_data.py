"""
Download NHANES Data Script

Downloads, loads, cleans, and merges all 12 NHANES datasets.
Run this script to prepare data for The Pediatric Sentinel project.

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.downloader import NHANESDownloader
from src.data.loader import NHANESLoader
from src.data.cleaner import NHANESCleaner
from src.data.merger import NHANESMerger
from config.constants import ALL_DATASETS


def main():
    """Main pipeline: Download → Load → Clean → Merge."""

    print("\n" + "=" * 70)
    print("THE PEDIATRIC SENTINEL - DATA PIPELINE")
    print("=" * 70)
    print("This script will:")
    print("  1. Download 12 NHANES datasets from CDC")
    print("  2. Load .xpt files into DataFrames")
    print("  3. Clean data (filter age 12-19, handle missing values)")
    print("  4. Merge datasets on SEQN")
    print("  5. Save merged dataset for analysis")
    print("=" * 70)

    # Get user confirmation
    response = input("\nProceed with download? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return

    # ========================================================================
    # Step 1: Download Data
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 70)

    downloader = NHANESDownloader()
    downloaded = downloader.download_all_datasets(overwrite=False)

    if len(downloaded) < len(ALL_DATASETS):
        print("\n[WARN] Some datasets failed to download")
        proceed = input("Continue with available datasets? (yes/no): ")
        if proceed.lower() not in ['yes', 'y']:
            print("Aborted.")
            return

    # Verify downloads
    verification = downloader.verify_downloads(ALL_DATASETS)

    # ========================================================================
    # Step 2: Load Data
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATA")
    print("=" * 70)

    loader = NHANESLoader()
    datasets = loader.load_all_datasets()

    if not datasets:
        print("[FAIL] No datasets loaded. Exiting.")
        return

    # Print dataset info
    loader.print_dataset_info(datasets)

    # ========================================================================
    # Step 3: Clean Data
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: CLEANING DATA")
    print("=" * 70)

    cleaner = NHANESCleaner()
    cleaned_datasets = cleaner.clean_all_datasets(datasets)

    # ========================================================================
    # Step 4: Merge Data
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: MERGING DATA")
    print("=" * 70)

    merger = NHANESMerger()
    merged_data = merger.merge_and_save(cleaned_datasets)

    # ========================================================================
    # Step 5: Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("DATA PIPELINE COMPLETE!")
    print("=" * 70)

    print(f"\n[OK] Downloaded: {len(downloaded)} datasets")
    print(f"[OK] Loaded: {len(datasets)} datasets")
    print(f"[OK] Cleaned: {len(cleaned_datasets)} datasets")
    print(f"[OK] Merged dataset: {len(merged_data):,} rows × {merged_data.shape[1]} columns")

    # Age distribution
    if 'RIDAGEYR' in merged_data.columns:
        print(f"\nAge distribution:")
        print(merged_data['RIDAGEYR'].describe())

    # Missing values
    missing_pct = (merged_data.isnull().sum().sum() /
                   (merged_data.shape[0] * merged_data.shape[1])) * 100
    print(f"\nMissing values: {missing_pct:.1f}%")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  2. Engineer features: python src/features/...")
    print("  3. Train model: python scripts/train_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
