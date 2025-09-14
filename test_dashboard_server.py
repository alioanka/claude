"""
Test dashboard server with VPS connection.
"""

import asyncio
import sys
import os
sys.path.append('.')

from monitoring.dashboard import DashboardManager
from core.portfolio_manager import PortfolioManager
from local_config import get_database_url

async def start_test_dashboard():
    """Start test dashboard server"""
    
    print("🚀 Starting test dashboard server...")
    
    try:
        # Create database manager with VPS connection
        database_url = get_database_url()
        print(f"📡 Connecting to: {database_url}")
        
        from data.database import DatabaseManager
        db = DatabaseManager(database_url=database_url)
        await db.initialize()
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(db, initial_capital=10000)
        
        # Create dashboard manager
        dashboard = DashboardManager(db, portfolio_manager)
        
        print("✅ Dashboard initialized successfully!")
        print("🌐 Starting dashboard server on http://localhost:8000")
        print("📊 Recent Trades should now show closed positions!")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Start the dashboard server
        await dashboard.start_dashboard(host="127.0.0.1", port=8000)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped by user")
    except Exception as e:
        print(f"❌ Dashboard server failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db' in locals():
            await db.close()

if __name__ == "__main__":
    asyncio.run(start_test_dashboard())
