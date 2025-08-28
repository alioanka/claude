#!/usr/bin/env python3
"""
Setup script to create necessary directories and __init__.py files.
Run this before starting the trading bot to ensure all packages are properly initialized.
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create necessary directories and __init__.py files."""
    
    # Directories to create
    directories = [
        "config",
        "core", 
        "data",
        "ml",
        "risk",
        "strategies", 
        "utils",
        "monitoring",
        "scripts",
        "tests",
        "logs",
        "storage/historical",
        "storage/models",
        "storage/backups",
        "storage/exports"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    python_packages = [
        "core",
        "data", 
        "ml",
        "risk",
        "strategies",
        "utils",
        "monitoring",
        "scripts",
        "tests"
    ]
    
    for package in python_packages:
        init_file = Path(package) / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization file."""\n')
            print(f"Created __init__.py in {package}")
    
    # Create root __init__.py
    root_init = Path("__init__.py")
    if not root_init.exists():
        root_init.write_text('"""Trading Bot Application Package"""\n')
        print("Created root __init__.py")
    
    print("Directory structure setup complete!")

if __name__ == "__main__":
    create_directory_structure()