#!/usr/bin/env python3
"""
Safe Enhanced Dashboard Runner
This script runs the enhanced dashboard without affecting existing bots
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables to avoid conflicts
os.environ['DASHBOARD_PORT'] = '8001'
os.environ['DASHBOARD_HOST'] = '0.0.0.0'

# Import and run the enhanced dashboard
try:
    from enhanced_dashboard_simple import app
    import uvicorn
    
    print("🚀 Starting Enhanced Dashboard on port 8001...")
    print("📊 Dashboard will be available at: http://localhost:8001")
    print("🔗 API documentation: http://localhost:8001/api/docs")
    print("⚠️  This will NOT affect your existing bots running on other ports")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Error importing enhanced dashboard: {e}")
    print("Please run the installation script first")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting enhanced dashboard: {e}")
    sys.exit(1)
