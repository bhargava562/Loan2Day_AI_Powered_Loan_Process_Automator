#!/usr/bin/env python3
"""
Utility script for cleaning up temporary files and directories.

This script helps maintain a clean development environment by removing
temporary files, logs, and generated content that shouldn't be committed.

Usage:
    python utils/cleanup.py [--dry-run]
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List

def get_cleanup_patterns() -> List[str]:
    """
    Get list of file patterns to clean up.
    
    Returns:
        List of glob patterns for files/directories to remove
    """
    return [
        "*.log",
        "*.tmp",
        "*.temp",
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        "logs/*.log",
        "logs/*.txt",
        "uploads/*",
        "generated_pdfs/*.pdf",
        "frontend/dist",
        "frontend/build",
        "node_modules",
    ]

def cleanup_directory(dry_run: bool = False) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting
    """
    project_root = Path(__file__).parent.parent
    patterns = get_cleanup_patterns()
    
    print(f"🧹 Cleaning up project directory: {project_root}")
    print(f"{'🔍 DRY RUN MODE - No files will be deleted' if dry_run else '🗑️  CLEANUP MODE - Files will be deleted'}")
    print("-" * 60)
    
    for pattern in patterns:
        files_to_clean = list(project_root.glob(pattern))
        
        for file_path in files_to_clean:
            # Skip .gitkeep files
            if file_path.name == '.gitkeep':
                continue
                
            if dry_run:
                print(f"Would delete: {file_path}")
            else:
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"✅ Deleted file: {file_path}")
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        print(f"✅ Deleted directory: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
    
    print("-" * 60)
    print("🎉 Cleanup completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up temporary files and directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    cleanup_directory(dry_run=args.dry_run)