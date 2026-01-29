"""
Run the Streamlit Risk Calculator App

This script launches the Pediatric Sentinel web application.

Usage:
    python scripts/run_app.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app."""

    project_root = Path(__file__).parent.parent
    app_file = project_root / "app" / "streamlit_app.py"

    if not app_file.exists():
        print(f"[FAIL] App file not found: {app_file}")
        sys.exit(1)

    print("=" * 70)
    print("THE PEDIATRIC SENTINEL - Risk Calculator")
    print("=" * 70)
    print(f"\nLaunching Streamlit app from: {app_file}")
    print("\nThe app will open in your default web browser.")
    print("Press Ctrl+C to stop the server.\n")
    print("=" * 70)

    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_file),
        "--server.port=8501",
        "--server.address=localhost"
    ])


if __name__ == "__main__":
    main()
