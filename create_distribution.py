#!/usr/bin/env python3
"""
Create a distribution zip of BactAID for sharing.

This script creates a clean zip file containing only the necessary files,
excluding development files, caches, and large unnecessary dependencies.
"""

import os
import zipfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_NAME = f"BactAID_v1.0_{datetime.now().strftime('%Y%m%d')}"

# Patterns to exclude
EXCLUDE_PATTERNS = {
    # Directories to skip entirely
    '.git',
    '__pycache__',
    '.pytest_cache',
    'node_modules',
    '.venv',
    'venv',
    'build',
    'dist',
    '.idea',
    '.vscode',
    # Files to skip
    '.gitignore',
    '.gitattributes',
    '.env',
    '*.pyc',
    '*.pyo',
    '*.egg-info',
    '.DS_Store',
    'Thumbs.db',
    # Large unnecessary files
    '*.log',
    # Build files
    'bactaid.spec',
    'build_windows.py',
    'create_distribution.py',
}

# Files/folders that must be included
REQUIRED_ITEMS = [
    'backend/',
    'engine/',
    'rag/',
    'scoring/',
    'training/',
    'data/',
    'models/',
    'frontend/dist/',
    'static/',
    'BactAID_Launcher.bat',
    'run_bactaid.py',
    'INSTALL.txt',
    'README.md',
    'USER_MANUAL.md',
    'eph.ico',
]


def should_exclude(path: Path, rel_path: str) -> bool:
    """Check if a path should be excluded."""
    parts = rel_path.split(os.sep)
    name = path.name

    for pattern in EXCLUDE_PATTERNS:
        # Check directory names
        if pattern in parts:
            return True
        # Check file names
        if name == pattern:
            return True
        # Check wildcards
        if pattern.startswith('*') and name.endswith(pattern[1:]):
            return True

    return False


def create_zip():
    """Create the distribution zip file."""
    output_path = PROJECT_ROOT / f"{OUTPUT_NAME}.zip"

    print(f"Creating distribution: {output_path}")
    print("-" * 60)

    file_count = 0
    total_size = 0

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item in REQUIRED_ITEMS:
            item_path = PROJECT_ROOT / item

            if not item_path.exists():
                print(f"  [SKIP] {item} (not found)")
                continue

            if item_path.is_file():
                rel_path = item
                if not should_exclude(item_path, rel_path):
                    zf.write(item_path, f"BactAID/{rel_path}")
                    file_count += 1
                    total_size += item_path.stat().st_size
                    print(f"  [ADD] {rel_path}")
            else:
                # It's a directory
                for root, dirs, files in os.walk(item_path):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, d)]

                    for file in files:
                        file_path = Path(root) / file
                        rel_path = file_path.relative_to(PROJECT_ROOT)
                        rel_str = str(rel_path)

                        if not should_exclude(file_path, rel_str):
                            zf.write(file_path, f"BactAID/{rel_str}")
                            file_count += 1
                            total_size += file_path.stat().st_size

                print(f"  [ADD] {item} ({file_count} files so far)")

    # Get final zip size
    zip_size = output_path.stat().st_size

    print("-" * 60)
    print(f"Files included: {file_count}")
    print(f"Uncompressed size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Compressed size: {zip_size / 1024 / 1024:.1f} MB")
    print(f"\nDistribution created: {output_path}")
    print("\nTo share BactAID:")
    print(f"  1. Send '{OUTPUT_NAME}.zip' to recipients")
    print("  2. They extract and run BactAID_Launcher.bat")
    print("  3. They need Python 3.10+ and Ollama installed")

    return output_path


if __name__ == "__main__":
    create_zip()
