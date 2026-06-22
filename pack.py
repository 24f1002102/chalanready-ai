"""
ChalanReady AI — Submission Packager
=====================================
Creates a clean, judge-ready ZIP archive of the project.

Usage:
    python pack.py                   # Creates chalanready_submit.zip
    python pack.py --out my_sub.zip  # Custom output name
    python pack.py --verify          # Verify zip contents only

What's included:
    backend/         Full Python backend (FastAPI, pipeline, detection rules)
    frontend/        Dashboard HTML
    sample_data/     Synthetic video generator + evaluator
    requirements.txt, requirements-local.txt
    README.md
    CONCEPT_NOTE.md (if present)
    run.py / run.sh / run.bat

What's excluded:
    __pycache__, *.pyc, *.pyo
    .git, .gitignore
    node_modules
    *.sqlite3, *.db          (run-time data)
    sample_data/uploads/*    (user uploads)
    sample_data/outputs/*    (generated videos)
    sample_data/eval_output/ (generated metrics)
    .env, secrets.*
    *.mp4 > 5 MB             (large demo videos)

Run: python pack.py
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Files / dirs to always exclude ────────────────────────────────────────
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".github", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
    "sample_data/outputs", "sample_data/uploads", "sample_data/eval_output",
    "sample_data/analysis_tmp",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite3", ".egg-info", ".zip"}
EXCLUDE_FILENAMES = {".DS_Store", "Thumbs.db", ".env", ".env.local"}
MAX_MP4_MB = 5  # exclude mp4 files larger than this


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    rel_str = rel.as_posix()

    # Check excluded dirs
    for part in rel.parts:
        if part in EXCLUDE_DIRS:
            return False
    for excl_dir in EXCLUDE_DIRS:
        if rel_str.startswith(excl_dir):
            return False

    # Check excluded suffixes
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False

    # Check excluded filenames
    if path.name in EXCLUDE_FILENAMES:
        return False

    # Exclude large video files (keep small synthetic demo clips if present)
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_MP4_MB:
            print(f"  [SKIP] {rel_str} ({size_mb:.1f} MB — too large)")
            return False

    return True


def build_zip(output_path: Path) -> list[str]:
    """Build the submission ZIP and return list of included files."""
    included = []
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(ROOT.rglob("*")):
            if not file_path.is_file():
                continue
            if not should_include(file_path):
                continue
            arcname = file_path.relative_to(ROOT).as_posix()
            zf.write(file_path, arcname)
            included.append(arcname)
            print(f"  + {arcname}")
    return included


def verify_zip(zip_path: Path) -> None:
    """List and verify the contents of a ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        total_size = sum(i.file_size for i in infos)
        print(f"\nZIP: {zip_path.name}")
        print(f"Files: {len(infos)}")
        print(f"Uncompressed size: {total_size/1024/1024:.1f} MB")
        print(f"Compressed size:   {zip_path.stat().st_size/1024/1024:.1f} MB")
        print("\nKey files:")
        key_patterns = ["README", "requirements", "run.", "backend/main.py",
                        "frontend/dashboard.html", "CONCEPT_NOTE", "evaluate.py"]
        for pattern in key_patterns:
            for info in infos:
                if pattern.lower() in info.filename.lower():
                    print(f"  [OK] {info.filename}")
                    break
            else:
                print(f"  [MISSING] {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChalanReady AI submission packager")
    parser.add_argument("--out", default="chalanready_submit.zip",
                        help="Output zip filename (default: chalanready_submit.zip)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing zip contents only")
    args = parser.parse_args()

    out_path = ROOT / args.out

    if args.verify:
        if not out_path.exists():
            print(f"ERROR: {out_path} not found. Run without --verify to build it.")
            sys.exit(1)
        verify_zip(out_path)
        return

    print(f"\n{'='*60}")
    print(f"  ChalanReady AI — Submission Packager")
    print(f"{'='*60}")
    print(f"  Project root : {ROOT}")
    print(f"  Output file  : {out_path.name}\n")
    print("Including:")

    included = build_zip(out_path)

    print(f"\n{'='*60}")
    print(f"  Done! {len(included)} files packaged.")
    print(f"  Output: {out_path}")
    print(f"  Size  : {out_path.stat().st_size/1024/1024:.2f} MB")
    print(f"{'='*60}\n")

    # Quick sanity
    print("Verifying...")
    verify_zip(out_path)

    print("\n✅ Submission zip is ready. Upload chalanready_submit.zip to the hackathon portal.")


if __name__ == "__main__":
    main()
