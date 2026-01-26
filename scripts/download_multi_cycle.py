"""
Download Multiple NHANES Cycles

Downloads data from multiple NHANES cycles to increase dataset size.
Combines 2015-2016 and 2017-2018 for ~1,000+ pediatric participants.

Usage:
    python scripts/download_multi_cycle.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.downloader import NHANESDownloader
from src.data.loader import NHANESLoader
from config.constants import NHANES_CYCLES, DATASET_NAMES, ENHANCED_DATASET_NAMES


def download_cycle(cycle_name: str, cycle_info: dict) -> dict:
    """
    Download all datasets for a specific cycle.

    Args:
        cycle_name: Cycle name (e.g., "2015-2016")
        cycle_info: Cycle configuration dict

    Returns:
        Dict mapping dataset codes to file paths
    """
    print("\n" + "=" * 70)
    print(f"DOWNLOADING NHANES {cycle_name} CYCLE")
    print("=" * 70)

    suffix = cycle_info["suffix"]
    base_url = cycle_info["base_url"]

    # Generate dataset list for this cycle
    datasets = {}
    for name in DATASET_NAMES.keys():
        code = f"{name}{suffix}"
        datasets[code] = f"{code}.xpt"

    for name in ENHANCED_DATASET_NAMES.keys():
        code = f"{name}{suffix}"
        datasets[code] = f"{code}.xpt"

    print(f"\nDatasets to download: {len(datasets)}")
    print(f"Base URL: {base_url}")

    # Download datasets
    downloader = NHANESDownloader(base_url=base_url)
    downloaded = {}

    for code, filename in datasets.items():
        result = downloader.download_file(code, filename, overwrite=False)
        if result:
            downloaded[code] = result

    print(f"\n[OK] Downloaded {len(downloaded)}/{len(datasets)} datasets for {cycle_name}")

    return downloaded


def load_cycle_data(cycle_name: str, cycle_info: dict) -> pd.DataFrame:
    """
    Load and merge all datasets for a specific cycle.

    Args:
        cycle_name: Cycle name
        cycle_info: Cycle configuration

    Returns:
        Merged DataFrame for the cycle
    """
    print("\n" + "=" * 70)
    print(f"LOADING {cycle_name} DATA")
    print("=" * 70)

    suffix = cycle_info["suffix"]
    loader = NHANESLoader()

    # Load datasets
    datasets = {}
    for name in DATASET_NAMES.keys():
        code = f"{name}{suffix}"
        filename = f"{code}.xpt"
        file_path = loader.data_dir / filename

        if file_path.exists():
            df = loader.load_xpt_file(file_path)
            if df is not None:
                datasets[code] = df

    for name in ENHANCED_DATASET_NAMES.keys():
        code = f"{name}{suffix}"
        filename = f"{code}.xpt"
        file_path = loader.data_dir / filename

        if file_path.exists():
            df = loader.load_xpt_file(file_path)
            if df is not None:
                datasets[code] = df

    print(f"\n[OK] Loaded {len(datasets)} datasets for {cycle_name}")

    # Merge on SEQN
    if not datasets:
        print("[FAIL] No datasets loaded")
        return pd.DataFrame()

    # Start with demographics
    demo_code = f"DEMO{suffix}"
    if demo_code not in datasets:
        print(f"[FAIL] Demographics dataset {demo_code} not found")
        return pd.DataFrame()

    merged = datasets[demo_code].copy()
    merged['NHANES_CYCLE'] = cycle_name  # Add cycle identifier

    print(f"\nBase: {demo_code} ({len(merged):,} rows)")

    # Merge other datasets
    for code, df in datasets.items():
        if code == demo_code:
            continue

        if 'SEQN' not in df.columns:
            print(f"[WARN] Skipping {code} - no SEQN column")
            continue

        before = merged.shape[1]
        merged = merged.merge(df, on='SEQN', how='left', suffixes=('', f'_{code}'))
        added = merged.shape[1] - before

        matched = df['SEQN'].isin(merged['SEQN']).sum()
        print(f"  {code}: {matched:,}/{len(df):,} matched, +{added} columns")

    print(f"\n[OK] Merged {cycle_name}: {len(merged):,} rows × {merged.shape[1]} columns")

    return merged


def main():
    """Main function to download and combine multiple cycles."""

    print("=" * 70)
    print("MULTI-CYCLE NHANES DATA DOWNLOAD")
    print("=" * 70)
    print(f"Cycles: {', '.join(NHANES_CYCLES.keys())}")
    print(f"Expected outcome: ~1,000-1,500 pediatric participants (ages 12-19)")
    print("=" * 70)

    # Download all cycles
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOADING ALL CYCLES")
    print("=" * 70)

    for cycle_name, cycle_info in NHANES_CYCLES.items():
        download_cycle(cycle_name, cycle_info)

    # Load and merge each cycle
    print("\n" + "=" * 70)
    print("STEP 2: LOADING AND MERGING CYCLES")
    print("=" * 70)

    cycle_dataframes = []
    for cycle_name, cycle_info in NHANES_CYCLES.items():
        df = load_cycle_data(cycle_name, cycle_info)
        if not df.empty:
            cycle_dataframes.append(df)

    if not cycle_dataframes:
        print("[FAIL] No data loaded from any cycle")
        return

    # Combine all cycles
    print("\n" + "=" * 70)
    print("STEP 3: COMBINING CYCLES")
    print("=" * 70)

    combined = pd.concat(cycle_dataframes, ignore_index=True)
    print(f"\n[OK] Combined dataset: {len(combined):,} rows × {combined.shape[1]} columns")

    # Check cycle distribution
    print("\nCycle distribution:")
    print(combined['NHANES_CYCLE'].value_counts())

    # Save combined data
    output_file = Path(__file__).parent.parent / "data" / "processed" / "multi_cycle_merged.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)

    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"\n[OK] Saved multi-cycle data: {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Filter to pediatric population
    print("\n" + "=" * 70)
    print("STEP 4: PEDIATRIC POPULATION (AGES 12-19)")
    print("=" * 70)

    pediatric = combined[(combined['RIDAGEYR'] >= 12) & (combined['RIDAGEYR'] <= 19)].copy()

    print(f"\nPediatric dataset: {len(pediatric):,} participants")
    print("\nAge distribution:")
    print(pediatric['RIDAGEYR'].value_counts().sort_index())

    print("\nCycle distribution (pediatric):")
    print(pediatric['NHANES_CYCLE'].value_counts())

    # Check glucose & insulin availability
    has_both = pediatric['LBXGLU'].notna() & pediatric['LBXIN'].notna()
    print(f"\nWith glucose & insulin: {has_both.sum():,}/{len(pediatric):,} ({has_both.sum()/len(pediatric)*100:.1f}%)")

    # Save pediatric data
    pediatric_file = Path(__file__).parent.parent / "data" / "processed" / "multi_cycle_pediatric.csv"
    pediatric.to_csv(pediatric_file, index=False)

    print(f"\n[OK] Saved pediatric data: {pediatric_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("MULTI-CYCLE DOWNLOAD COMPLETE!")
    print("=" * 70)

    print(f"\nTotal participants: {len(combined):,}")
    print(f"Pediatric participants (12-19): {len(pediatric):,}")
    print(f"With glucose & insulin: {has_both.sum():,}")

    comparison_2015 = 506  # Previous single-cycle count
    improvement = ((has_both.sum() - comparison_2015) / comparison_2015) * 100

    print(f"\nImprovement over single cycle:")
    print(f"  Previous (2015-2016 only): {comparison_2015}")
    print(f"  Current (multi-cycle): {has_both.sum():,}")
    print(f"  Increase: +{has_both.sum() - comparison_2015:,} ({improvement:+.1f}%)")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Use 'multi_cycle_pediatric.csv' for feature engineering")
    print("  2. Expect better model performance with larger dataset")
    print("  3. Train model: python scripts/train_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
