#!/usr/bin/env python
"""
Cleanup script to remove all previous outputs before running the pipeline
"""
import sys
import os
import shutil
from pathlib import Path
import argparse

def cleanup_project(project_root=None, keep_data=False):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    
    print("="*60)
    print("CLEANING UP PROJECT")
    print("="*60)
    
    # Directories to clean
    dirs_to_clean = [
        'artifacts',
        'logs',
        'data/processed',
        'data/interim'
    ]
    
    # Files to clean (outside of directories above)
    files_to_clean = [
        'order_driver_pairs.csv',
        'detailed_predictions.csv',
        'trained_model.pkl',
        'feature_engineer.pkl',
        'model_metrics.json',
        'model_metrics.csv',
        'training_results.png'
    ]
    
    total_removed = 0
    
    # Clean directories
    for dir_path in dirs_to_clean:
        full_path = project_root / dir_path
        if full_path.exists():
            try:
                # Count files before removal
                file_count = sum(len(files) for _, _, files in os.walk(full_path))
                
                if dir_path == 'data/processed' and keep_data:
                    print(f"⚠ Skipping {dir_path} (keep_data=True)")
                    continue
                
                # Remove directory
                shutil.rmtree(full_path)
                print(f"✓ Removed directory: {dir_path} ({file_count} files)")
                total_removed += file_count
            except Exception as e:
                print(f"✗ Error removing {dir_path}: {e}")
        else:
            print(f"○ Directory does not exist: {dir_path}")
    
    # Clean individual files
    for file_path in files_to_clean:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                print(f"✓ Removed file: {file_path}")
                total_removed += 1
            except Exception as e:
                print(f"✗ Error removing {file_path}: {e}")
        else:
            print(f"○ File does not exist: {file_path}")
    
    # Clean Python cache files
    print("\nCleaning Python cache files...")
    cache_dirs_removed = 0
    cache_files_removed = 0
    
    for root, dirs, files in os.walk(project_root):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            cache_dir = Path(root) / '__pycache__'
            try:
                shutil.rmtree(cache_dir)
                cache_dirs_removed += 1
                dirs.remove('__pycache__')  # Don't walk into removed dir
            except:
                pass
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo'):
                try:
                    (Path(root) / file).unlink()
                    cache_files_removed += 1
                except:
                    pass
        
        # Remove .pytest_cache
        if '.pytest_cache' in dirs:
            cache_dir = Path(root) / '.pytest_cache'
            try:
                shutil.rmtree(cache_dir)
                cache_dirs_removed += 1
                dirs.remove('.pytest_cache')
            except:
                pass
    
    if cache_dirs_removed > 0 or cache_files_removed > 0:
        print(f"✓ Removed {cache_dirs_removed} cache directories and {cache_files_removed} cache files")
    
    # Create necessary directories
    print("\nRecreating necessary directories...")
    necessary_dirs = [
        'artifacts/models',
        'artifacts/features',
        'artifacts/metrics',
        'artifacts/predictions',
        'logs',
        'data/processed',
        'data/interim',
        'data/raw'
    ]
    
    for dir_path in necessary_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    print(f"\n" + "="*60)
    print(f"CLEANUP COMPLETED")
    print(f"="*60)
    print(f"Total items removed: {total_removed}")
    print(f"Project ready for fresh run!")
    
    return total_removed

def verify_cleanup(project_root=None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    
    print("\n" + "="*60)
    print("VERIFYING CLEANUP")
    print("="*60)
    
    # Check directories
    check_dirs = [
        'artifacts',
        'logs',
        'data/processed',
        'data/interim'
    ]
    
    all_clean = True
    
    for dir_path in check_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            # Check if directory is empty
            items = list(full_path.iterdir())
            if items:
                print(f"⚠ Directory not empty: {dir_path} ({len(items)} items)")
                all_clean = False
            else:
                print(f"✓ Directory clean: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_clean = False
    
    # Check files
    check_files = [
        'order_driver_pairs.csv',
        'detailed_predictions.csv',
        'trained_model.pkl'
    ]
    
    for file_path in check_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✗ File still exists: {file_path}")
            all_clean = False
        else:
            print(f"✓ File removed: {file_path}")
    
    if all_clean:
        print("\n✅ All checks passed! Project is clean.")
    else:
        print("\n⚠ Some issues found. Project may not be fully clean.")
    
    return all_clean

def main():
    parser = argparse.ArgumentParser(description='Clean up project outputs before running pipeline')
    
    parser.add_argument('--keep-data', action='store_true',
                       help='Keep processed data files')
    
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify cleanup without actually cleaning')
    
    parser.add_argument('--project-root', type=str, default=None,
                       help='Project root directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).resolve().parent.parent
    
    print(f"Project root: {project_root}")
    
    if args.verify_only:
        # Only verify
        verify_cleanup(project_root)
    else:
        # Clean and then verify
        cleanup_project(project_root, args.keep_data)
        print("\n")
        verify_cleanup(project_root)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())