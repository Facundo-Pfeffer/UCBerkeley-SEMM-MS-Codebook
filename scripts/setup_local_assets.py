#!/usr/bin/env python3
"""
Setup script for local development environment.

This script creates symlinks or copies assets and images to the project root
so that highlighted_htmls pages can access them with ../../assets/ paths
when serving locally from the project root.

This matches the deployment structure where webpage/ contents are at dist/ root.
"""

import os
import sys
from pathlib import Path

def setup_local_assets():
    """Set up assets and images in project root for local testing."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    webpage_assets = project_root / 'webpage' / 'assets'
    webpage_images = project_root / 'webpage' / 'images'
    
    root_assets = project_root / 'assets'
    root_images = project_root / 'images'
    
    # Check if webpage assets exist
    if not webpage_assets.exists():
        print(f"Error: {webpage_assets} does not exist!")
        return False
    
    if not webpage_images.exists():
        print(f"Error: {webpage_images} does not exist!")
        return False
    
    # Try to create symlinks (works on Unix/Mac, requires admin on Windows)
    try:
        if root_assets.exists():
            if root_assets.is_symlink():
                print(f"[OK] Assets symlink already exists: {root_assets}")
            else:
                print(f"[WARN] Assets directory exists but is not a symlink: {root_assets}")
        else:
            root_assets.symlink_to(webpage_assets, target_is_directory=True)
            print(f"[OK] Created assets symlink: {root_assets} -> {webpage_assets}")
        
        if root_images.exists():
            if root_images.is_symlink():
                print(f"[OK] Images symlink already exists: {root_images}")
            else:
                print(f"[WARN] Images directory exists but is not a symlink: {root_images}")
        else:
            root_images.symlink_to(webpage_images, target_is_directory=True)
            print(f"[OK] Created images symlink: {root_images} -> {webpage_images}")
        
        return True
    except OSError as e:
        # Symlinks not supported or require admin (Windows)
        print(f"[WARN] Could not create symlinks: {e}")
        print("  Attempting to copy directories instead...")
        
        # Copy directories instead
        import shutil
        
        try:
            if root_assets.exists():
                shutil.rmtree(root_assets)
            shutil.copytree(webpage_assets, root_assets)
            print(f"[OK] Copied assets directory: {root_assets}")
            
            if root_images.exists():
                shutil.rmtree(root_images)
            shutil.copytree(webpage_images, root_images)
            print(f"[OK] Copied images directory: {root_images}")
            
            print("\n[NOTE] Directories were copied, not symlinked.")
            print("  If you update assets/images, re-run this script to sync changes.")
            
            return True
        except Exception as e2:
            print(f"[ERROR] Error copying directories: {e2}")
            return False

if __name__ == '__main__':
    print("Setting up local assets for development...")
    print("=" * 50)
    
    if setup_local_assets():
        print("\n" + "=" * 50)
        print("[SUCCESS] Setup complete!")
        print("\nYou can now serve the site from the project root:")
        print("  python -m http.server 5510")
        print("\nPages in highlighted_htmls/ will be able to access:")
        print("  ../../assets/js/navigation.js")
        print("  ../../images/logo.png")
    else:
        print("\n" + "=" * 50)
        print("[ERROR] Setup failed. Please check the errors above.")
        sys.exit(1)

