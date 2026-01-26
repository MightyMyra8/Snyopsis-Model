"""
NHANES Data Downloader

Downloads NHANES .xpt files from the CDC website.
Supports both core datasets and enhanced datasets.
"""

import requests
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.constants import (
    NHANES_BASE_URL,
    RAW_DATA_DIR,
    DATASETS,
    ENHANCED_DATASETS,
    ALL_DATASETS
)


class NHANESDownloader:
    """Downloads NHANES datasets from CDC."""

    def __init__(self, save_dir: Optional[Path] = None, base_url: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            save_dir: Directory to save downloaded files. Defaults to RAW_DATA_DIR.
            base_url: Base URL for downloads. Defaults to NHANES_BASE_URL.
        """
        self.save_dir = save_dir or RAW_DATA_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url or NHANES_BASE_URL

    def download_file(
        self,
        dataset_code: str,
        filename: str,
        overwrite: bool = False
    ) -> Optional[Path]:
        """
        Download a single NHANES .xpt file.

        Args:
            dataset_code: Dataset code (e.g., "P_DEMO")
            filename: Filename (e.g., "P_DEMO.XPT")
            overwrite: If True, redownload even if file exists

        Returns:
            Path to downloaded file, or None if failed
        """
        file_path = self.save_dir / filename

        # Check if file already exists
        if file_path.exists() and not overwrite:
            print(f"[OK] {filename} already exists (skipping)")
            return file_path

        # Construct URL
        url = f"{self.base_url}{filename}"

        print(f"Downloading {dataset_code} from {url}...")

        try:
            # Send request with timeout
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"[OK] Downloaded {filename} ({self._format_size(total_size)})")
            return file_path

        except requests.exceptions.RequestException as e:
            print(f"[FAIL] Failed to download {filename}: {e}")
            return None

    def download_dataset(
        self,
        dataset_code: str,
        datasets_dict: Dict[str, str],
        overwrite: bool = False,
        retry: int = 3
    ) -> Optional[Path]:
        """
        Download a dataset with retry logic.

        Args:
            dataset_code: Dataset code (e.g., "P_DEMO")
            datasets_dict: Dictionary mapping codes to filenames
            overwrite: If True, redownload even if file exists
            retry: Number of retry attempts

        Returns:
            Path to downloaded file, or None if failed
        """
        if dataset_code not in datasets_dict:
            print(f"[FAIL] Unknown dataset code: {dataset_code}")
            return None

        filename = datasets_dict[dataset_code]

        for attempt in range(1, retry + 1):
            result = self.download_file(dataset_code, filename, overwrite)

            if result is not None:
                return result

            if attempt < retry:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds... (attempt {attempt}/{retry})")
                time.sleep(wait_time)

        print(f"[FAIL] Failed to download {dataset_code} after {retry} attempts")
        return None

    def download_core_datasets(self, overwrite: bool = False) -> Dict[str, Path]:
        """
        Download all 5 core NHANES datasets.

        Args:
            overwrite: If True, redownload even if files exist

        Returns:
            Dictionary mapping dataset codes to file paths
        """
        print("\n" + "=" * 70)
        print("DOWNLOADING CORE DATASETS (5 datasets)")
        print("=" * 70)

        results = {}
        for code in DATASETS.keys():
            file_path = self.download_dataset(code, DATASETS, overwrite)
            if file_path:
                results[code] = file_path

        self._print_summary(results, len(DATASETS))
        return results

    def download_enhanced_datasets(self, overwrite: bool = False) -> Dict[str, Path]:
        """
        Download all 7 enhanced NHANES datasets.

        Args:
            overwrite: If True, redownload even if files exist

        Returns:
            Dictionary mapping dataset codes to file paths
        """
        print("\n" + "=" * 70)
        print("DOWNLOADING ENHANCED DATASETS (7 datasets)")
        print("=" * 70)

        results = {}
        for code in ENHANCED_DATASETS.keys():
            file_path = self.download_dataset(code, ENHANCED_DATASETS, overwrite)
            if file_path:
                results[code] = file_path

        self._print_summary(results, len(ENHANCED_DATASETS))
        return results

    def download_all_datasets(self, overwrite: bool = False) -> Dict[str, Path]:
        """
        Download all 12 NHANES datasets (5 core + 7 enhanced).

        Args:
            overwrite: If True, redownload even if files exist

        Returns:
            Dictionary mapping dataset codes to file paths
        """
        print("\n" + "=" * 70)
        print("DOWNLOADING ALL DATASETS (12 total)")
        print("=" * 70)

        results = {}

        # Download core datasets
        for code in DATASETS.keys():
            file_path = self.download_dataset(code, ALL_DATASETS, overwrite)
            if file_path:
                results[code] = file_path

        # Download enhanced datasets
        for code in ENHANCED_DATASETS.keys():
            file_path = self.download_dataset(code, ALL_DATASETS, overwrite)
            if file_path:
                results[code] = file_path

        self._print_summary(results, len(ALL_DATASETS))
        return results

    def verify_downloads(self, datasets_dict: Dict[str, str]) -> Dict[str, bool]:
        """
        Verify that all datasets have been downloaded.

        Args:
            datasets_dict: Dictionary mapping codes to filenames

        Returns:
            Dictionary mapping dataset codes to verification status
        """
        print("\n" + "=" * 70)
        print("VERIFYING DOWNLOADED FILES")
        print("=" * 70)

        verification = {}
        for code, filename in datasets_dict.items():
            file_path = self.save_dir / filename
            exists = file_path.exists()
            size = file_path.stat().st_size if exists else 0

            verification[code] = exists

            status = "[OK]" if exists else "[FAIL]"
            size_str = self._format_size(size) if exists else "missing"
            print(f"{status} {code:12} | {filename:20} | {size_str}")

        success_count = sum(verification.values())
        total_count = len(datasets_dict)
        print(f"\nVerified: {success_count}/{total_count} datasets downloaded")

        return verification

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    @staticmethod
    def _print_summary(results: Dict[str, Path], total: int):
        """Print download summary."""
        success_count = len(results)
        failed_count = total - success_count

        print("\n" + "-" * 70)
        print(f"Download Summary: {success_count}/{total} successful")
        if failed_count > 0:
            print(f"Failed downloads: {failed_count}")
        print("-" * 70)


# Convenience functions
def download_core(overwrite: bool = False) -> Dict[str, Path]:
    """Download core 5 datasets."""
    downloader = NHANESDownloader()
    return downloader.download_core_datasets(overwrite)


def download_enhanced(overwrite: bool = False) -> Dict[str, Path]:
    """Download enhanced 7 datasets."""
    downloader = NHANESDownloader()
    return downloader.download_enhanced_datasets(overwrite)


def download_all(overwrite: bool = False) -> Dict[str, Path]:
    """Download all 12 datasets."""
    downloader = NHANESDownloader()
    return downloader.download_all_datasets(overwrite)


def verify_all() -> Dict[str, bool]:
    """Verify all datasets downloaded."""
    downloader = NHANESDownloader()
    return downloader.verify_downloads(ALL_DATASETS)


if __name__ == "__main__":
    # Test: Download all datasets
    print("THE PEDIATRIC SENTINEL - NHANES Data Downloader")
    print("=" * 70)

    downloader = NHANESDownloader()

    # Download all datasets
    results = downloader.download_all_datasets(overwrite=False)

    # Verify downloads
    verification = downloader.verify_downloads(ALL_DATASETS)

    # Final status
    if all(verification.values()):
        print("\n[OK] ALL DATASETS DOWNLOADED SUCCESSFULLY!")
    else:
        missing = [code for code, verified in verification.items() if not verified]
        print(f"\n[FAIL] Missing datasets: {', '.join(missing)}")
