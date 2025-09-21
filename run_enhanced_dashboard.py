#!/usr/bin/env python3
"""
Enhanced Dashboard Runner
Script to run the ClaudeBot Enhanced Dashboard
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration
from enhanced_dashboard_config import get_config, get_log_level, get_port, get_host, is_development

def setup_logging():
    """Setup logging configuration"""
    log_level = get_log_level()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enhanced_dashboard.log')
        ]
    )
    
    # Set specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('websockets').setLevel(logging.WARNING)

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'fastapi',
        'uvicorn',
        'websockets',
        'pandas',
        'numpy',
        'psycopg2',
        'redis'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_database_connection():
    """Check if database is accessible"""
    try:
        from data.database import DatabaseManager
        from config.config import config
        
        # Try to connect to database
        db_manager = DatabaseManager(config.database.url)
        # Note: This is a synchronous check, actual async initialization happens in the app
        print("✅ Database connection check passed")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_redis_connection():
    """Check if Redis is accessible"""
    try:
        import redis
        from enhanced_dashboard_config import get_redis_url
        
        redis_url = get_redis_url()
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Redis connection check passed")
        return True
        
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("Redis is optional for caching, continuing without it...")
        return False

def main():
    """Main function to run the enhanced dashboard"""
    print("🚀 Starting ClaudeBot Enhanced Dashboard...")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check database connection
    if not check_database_connection():
        print("❌ Cannot start dashboard without database connection")
        sys.exit(1)
    
    # Check Redis connection (optional)
    check_redis_connection()
    
    # Get configuration
    config = get_config()
    host = get_host()
    port = get_port()
    
    print(f"📊 Dashboard Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Environment: {'Development' if is_development() else 'Production'}")
    print(f"   Debug Mode: {is_development()}")
    print("=" * 50)
    
    try:
        # Import and run the dashboard
        from enhanced_dashboard import app
        
        # Run with uvicorn
        import uvicorn
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=is_development(),
            log_level=get_log_level().lower(),
            access_log=True,
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        logger.info("Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
