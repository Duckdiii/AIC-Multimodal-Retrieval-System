"""
export_aic_core.py
==================
Packaging and synchronization utility for aic_core module.
Packages aic_core into a standalone zip / wheel for seamless installation on Google Colab:
!pip install aic_core.zip
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def package_aic_core(output_zip: str = "aic_core.zip"):
    root_dir = Path(__file__).parent
    source_dir = root_dir / "aic_core"
    dest_zip = root_dir / output_zip

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    print(f"Packaging {source_dir} into {dest_zip}...")
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in source_dir.rglob("*"):
            if "__pycache__" in file.parts or file.suffix == ".pyc":
                continue
            arcname = file.relative_to(root_dir)
            zf.write(file, arcname)
            print(f"  + Added {arcname}")

    print(f"Successfully created {dest_zip} ({dest_zip.stat().st_size / 1024:.1f} KB)")
    print("\nHow to use on Google Colab:")
    print("1. Upload aic_core.zip to your Colab workspace or Google Drive.")
    print("2. In your notebook, run:")
    print("   !unzip -q aic_core.zip -d .")
    print("   from aic_core.pipeline import AicOnlinePipeline")


if __name__ == "__main__":
    package_aic_core()
