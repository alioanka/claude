#!/usr/bin/env python3
"""
Enhanced Dashboard using proven DashboardManager
Reuses the same logic as your working dashboard on port 8000
"""

import os
import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_dashboard_safe")

PORT = int(os.environ.get("ENHANCED_DASHBOARD_PORT", "8001"))

def main():
    """Run the enhanced dashboard standalone"""
    logger.info(f"🚀 Starting Enhanced Dashboard on port {PORT}...")
    logger.info(f"📊 Dashboard will be available at: http://localhost:{PORT}")
    logger.info(f"🔗 API documentation: http://localhost:{PORT}/docs")
    logger.info("⚠️ This will NOT affect your existing bots running on other ports")
    logger.info("=" * 60)

    try:
        # Import the enhanced dashboard standalone
        from enhanced_dashboard_standalone import app
        
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    except Exception as e:
        logger.error(f"❌ Error starting enhanced dashboard: {e}")
        return 1

if __name__ == "__main__":
    exit(main())