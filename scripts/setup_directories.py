#!/usr/bin/env python3
"""
Setup script to create necessary directories for the trading bot.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create all necessary directories for the trading bot."""
    directories = [
        'logs',
        'storage',
        'storage/historical',
        'storage/models',
        'storage/backups',
        'storage/exports',
        'storage/performance',
        'ml',
        'ml/models',
        'data',
        'data/cache',
        'config'
    ]
    
    created_count = 0
    
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
                created_count += 1
            else:
                logger.debug(f"Directory already exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
    
    logger.info(f"Directory setup complete. Created {created_count} new directories.")
    
    # Create empty __init__.py files for Python packages
    package_dirs = [
        'ml',
        'data',
        'config',
        'core',
        'strategies',
        'risk',
        'utils',
        'monitoring'
    ]
    
    for pkg_dir in package_dirs:
        init_file = os.path.join(pkg_dir, '__init__.py')
        if not os.path.exists(init_file):
            try:
                os.makedirs(pkg_dir, exist_ok=True)
                with open(init_file, 'w') as f:
                    f.write('"""Package initialization file."""\n')
                logger.info(f"Created package init file: {init_file}")
            except Exception as e:
                logger.error(f"Failed to create init file {init_file}: {e}")

if __name__ == "__main__":
    create_directories()