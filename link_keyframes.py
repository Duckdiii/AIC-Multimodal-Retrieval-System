"""
link_keyframes.py
=================
Links all 873 keyframe video directories from D:\Data\AIC into the project's keyframes/ folder
using Windows NTFS Directory Junctions (instant, 0 extra disk space, no copying).
"""

import sys
import os
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def link_all_keyframes(source_root: str = r"D:\Data\AIC", target_folder: str = "keyframes"):
    src_base = Path(source_root)
    if not src_base.exists():
        print(f"❌ Source folder not found: {source_root}")
        return

    dst_base = Path(target_folder)
    dst_base.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning video folders in {src_base}...")
    found_videos = {}
    for batch_dir in src_base.glob("Keyframes_*"):
        kf_dir = batch_dir / "keyframes"
        if kf_dir.exists():
            for v_dir in kf_dir.iterdir():
                if v_dir.is_dir():
                    found_videos[v_dir.name] = v_dir

    print(f"📊 Found {len(found_videos)} video directories.")

    linked_count = 0
    skipped_count = 0

    for v_name, v_path in found_videos.items():
        junction_path = dst_base / v_name
        if junction_path.exists():
            skipped_count += 1
            continue

        # Create NTFS directory junction
        cmd = f'cmd /c mklink /J "{junction_path}" "{v_path}"'
        res = subprocess.run(cmd, shell=True, capture_output=True)
        if res.returncode == 0:
            linked_count += 1
        else:
            print(f"  ⚠️ Failed for {v_name}: {res.stderr.decode('utf-8', errors='ignore')}")

    print(f"✅ Finished linking!")
    print(f"   - Newly linked: {linked_count}")
    print(f"   - Already linked: {skipped_count}")
    print(f"   - Total in {dst_base}: {len(list(dst_base.iterdir()))}")


if __name__ == "__main__":
    link_all_keyframes()
