"""
Database setup and migration script for the crypto trading bot.
"""

import asyncio
import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager, Base, MarketData, Position, Trade, Signal
from sqlalchemy import create_engine, text
import alembic.config
from alembic import command

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and management utilities"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.db_manager = None
        
    async def initialize(self):
        """Initialize database setup"""
        try:
            self.engine = create_engine(self.database_url)
            self.db_manager = DatabaseManager(self.database_url)
            await self.db_manager.initialize()
            logger.info("Database setup initialized")
        except Exception as e:
            logger.error(f"Database setup initialization failed: {e}")
            raise
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            logger.info("Creating database tables...")
            
            # Create all tables defined in models
            Base.metadata.create_all(self.engine)
            
            # Verify tables were created
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'") 
                                    if 'sqlite' in self.database_url 
                                    else text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
                tables = [row[0] for row in result]
                
            logger.info(f"Created tables: {tables}")
            return True
            
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            return False
    
    async def seed_database(self):
        """Seed database with initial data"""
        try:
            logger.info("Seeding database with initial data...")
            
            # Add sample configuration data
            session = self.db_manager.get_session()
            
            # You can add initial trading pairs, strategies config, etc.
            # For now, we'll just verify the database is working
            
            # Test insert
            test_signal = Signal(
                symbol='BTCUSDT',
                strategy='test',
                signal_type='hold',
                confidence=0.5,
                price=50000.0,
                metadata='{"test": true}'
            )
            
            session.add(test_signal)
            session.commit()
            
            # Test query
            signals = session.query(Signal).all()
            logger.info(f"Database seeded successfully. Test signals: {len(signals)}")
            
            # Clean up test data
            session.delete(test_signal)
            session.commit()
            session.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Database seeding failed: {e}")
            return False
    
    async def migrate_database(self):
        """Run database migrations"""
        try:
            logger.info("Running database migrations...")
            
            # Check if alembic is configured
            alembic_cfg_path = Path("alembic.ini")
            if not alembic_cfg_path.exists():
                await self._create_alembic_config()
            
            # Run migrations
            alembic_cfg = alembic.config.Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            
            logger.info("Database migrations completed")
            return True
            
        except Exception as e:
            logger.warning(f"Migration failed (may not be configured): {e}")
            # Fall back to direct table creation
            return await self.create_tables()
    
    async def _create_alembic_config(self):
        """Create basic alembic configuration"""
        logger.info("Creating alembic configuration...")
        
        # Create alembic.ini
        alembic_ini = """
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = {database_url}

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
        """.format(database_url=self.database_url)
        
        with open("alembic.ini", "w") as f:
            f.write(alembic_ini)
        
        # Create alembic directory structure
        os.makedirs("alembic/versions", exist_ok=True)
        
        # Create env.py
        env_py = """
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.database import Base

config = context.config
fileConfig(config.config_file_name)
target_metadata = Base.metadata

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
        """
        
        with open("alembic/env.py", "w") as f:
            f.write(env_py)
    
    async def backup_database(self, backup_path: str = None) -> str:
        """Create database backup"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backups/db_backup_{timestamp}.sql"
            
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            if 'postgresql' in self.database_url:
                # PostgreSQL backup
                result = subprocess.run([
                    "pg_dump", self.database_url, "-f", backup_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"PostgreSQL backup created: {backup_path}")
                else:
                    logger.error(f"PostgreSQL backup failed: {result.stderr}")
                    return None
            else:
                # SQLite backup (simple file copy)
                db_file = self.database_url.replace('sqlite:///', '')
                if os.path.exists(db_file):
                    import shutil
                    shutil.copy2(db_file, backup_path)
                    logger.info(f"SQLite backup created: {backup_path}")
                else:
                    logger.error(f"SQLite database file not found: {db_file}")
                    return None
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None
    
    async def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            logger.info(f"Restoring database from {backup_path}")
            
            if 'postgresql' in self.database_url:
                # PostgreSQL restore
                result = subprocess.run([
                    "psql", self.database_url, "-f", backup_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("PostgreSQL restore completed")
                    return True
                else:
                    logger.error(f"PostgreSQL restore failed: {result.stderr}")
                    return False
            else:
                # SQLite restore (simple file copy)
                db_file = self.database_url.replace('sqlite:///', '')
                import shutil
                shutil.copy2(backup_path, db_file)
                logger.info("SQLite restore completed")
                return True
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    async def reset_database(self):
        """Reset database (drop and recreate all tables)"""
        try:
            logger.warning("Resetting database - all data will be lost!")
            
            # Drop all tables
            Base.metadata.drop_all(self.engine)
            logger.info("All tables dropped")
            
            # Recreate tables
            success = await self.create_tables()
            if success:
                logger.info("Database reset completed")
            return success
            
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            return False
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health and connection"""
        try:
            health_info = {
                'connected': False,
                'tables_exist': False,
                'can_read': False,
                'can_write': False,
                'table_count': 0,
                'error': None
            }
            
            # Test connection
            with self.engine.connect() as conn:
                health_info['connected'] = True
                
                # Check tables exist
                if 'sqlite' in self.database_url:
                    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                else:
                    result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
                
                tables = [row[0] for row in result]
                health_info['tables_exist'] = len(tables) > 0
                health_info['table_count'] = len(tables)
                
                if health_info['tables_exist']:
                    # Test read
                    try:
                        session = self.db_manager.get_session()
                        signals = session.query(Signal).limit(1).all()
                        health_info['can_read'] = True
                        
                        # Test write
                        test_signal = Signal(
                            symbol='TEST',
                            strategy='health_check',
                            signal_type='hold',
                            confidence=0.5,
                            price=1.0
                        )
                        session.add(test_signal)
                        session.commit()
                        session.delete(test_signal)
                        session.commit()
                        session.close()
                        health_info['can_write'] = True
                        
                    except Exception as e:
                        health_info['error'] = str(e)
            
            return health_info
            
        except Exception as e:
            return {
                'connected': False,
                'tables_exist': False,
                'can_read': False,
                'can_write': False,
                'table_count': 0,
                'error': str(e)
            }

async def main():
    """Main database setup function"""
    parser = argparse.ArgumentParser(description='Setup crypto trading bot database')
    parser.add_argument('--database-url', required=True, help='Database URL')
    parser.add_argument('--action', 
                       choices=['create', 'migrate', 'seed', 'backup', 'restore', 'reset', 'health'],
                       required=True, help='Action to perform')
    parser.add_argument('--backup-path', help='Backup file path')
    parser.add_argument('--force', action='store_true', help='Force action without confirmation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        db_setup = DatabaseSetup(args.database_url)
        await db_setup.initialize()
        
        if args.action == 'create':
            success = await db_setup.create_tables()
            print(f"Table creation {'successful' if success else 'failed'}")
            
        elif args.action == 'migrate':
            success = await db_setup.migrate_database()
            print(f"Migration {'successful' if success else 'failed'}")
            
        elif args.action == 'seed':
            success = await db_setup.seed_database()
            print(f"Database seeding {'successful' if success else 'failed'}")
            
        elif args.action == 'backup':
            backup_path = await db_setup.backup_database(args.backup_path)
            if backup_path:
                print(f"Backup created: {backup_path}")
            else:
                print("Backup failed")
                
        elif args.action == 'restore':
            if not args.backup_path:
                print("--backup-path required for restore")
                sys.exit(1)
            
            if not args.force:
                confirm = input("This will overwrite existing data. Continue? (y/N): ")
                if confirm.lower() != 'y':
                    print("Restore cancelled")
                    return
            
            success = await db_setup.restore_database(args.backup_path)
            print(f"Restore {'successful' if success else 'failed'}")
            
        elif args.action == 'reset':
            if not args.force:
                confirm = input("This will delete ALL data. Continue? (y/N): ")
                if confirm.lower() != 'y':
                    print("Reset cancelled")
                    return
            
            success = await db_setup.reset_database()
            print(f"Database reset {'successful' if success else 'failed'}")
            
        elif args.action == 'health':
            health = await db_setup.check_database_health()
            print("Database Health Check:")
            print(f"  Connected: {health['connected']}")
            print(f"  Tables exist: {health['tables_exist']}")
            print(f"  Can read: {health['can_read']}")
            print(f"  Can write: {health['can_write']}")
            print(f"  Table count: {health['table_count']}")
            if health['error']:
                print(f"  Error: {health['error']}")
            
            status = "HEALTHY" if all([
                health['connected'], 
                health['tables_exist'], 
                health['can_read'], 
                health['can_write']
            ]) else "UNHEALTHY"
            print(f"  Status: {status}")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())