-- Database initialization script for PostgreSQL
-- This script sets up the initial database structure

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create indexes for better performance
-- (Tables will be created by the application)

-- Set timezone
SET timezone = 'UTC';

-- Create a basic health check function
CREATE OR REPLACE FUNCTION health_check() 
RETURNS TEXT AS $$
BEGIN
    RETURN 'Database is healthy at ' || NOW();
END;
$$ LANGUAGE plpgsql;

-- Create a function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 90)
RETURNS TEXT AS $$
DECLARE
    cutoff_timestamp BIGINT;
    deleted_count INTEGER;
BEGIN
    -- Calculate cutoff timestamp (90 days ago in milliseconds)
    cutoff_timestamp := EXTRACT(EPOCH FROM (NOW() - INTERVAL '1 day' * days_to_keep)) * 1000;
    
    -- Clean old OHLCV data for short timeframes
    DELETE FROM ohlcv_data 
    WHERE timestamp < cutoff_timestamp 
    AND timeframe IN ('1m', '5m');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN 'Cleaned ' || deleted_count || ' old OHLCV records';
EXCEPTION
    WHEN OTHERS THEN
        RETURN 'Cleanup failed: ' || SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Create a function to get trading statistics
CREATE OR REPLACE FUNCTION get_trading_stats()
RETURNS TABLE(
    total_trades BIGINT,
    winning_trades BIGINT,
    losing_trades BIGINT,
    total_pnl NUMERIC,
    win_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_trades,
        COUNT(CASE WHEN pnl > 0 THEN 1 END)::BIGINT as winning_trades,
        COUNT(CASE WHEN pnl < 0 THEN 1 END)::BIGINT as losing_trades,
        COALESCE(SUM(pnl), 0) as total_pnl,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                ROUND((COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / COUNT(*)), 2)
            ELSE 0 
        END as win_rate
    FROM trades
    WHERE timestamp > EXTRACT(EPOCH FROM (NOW() - INTERVAL '30 days')) * 1000;
EXCEPTION
    WHEN OTHERS THEN
        -- Return zeros if trades table doesn't exist yet
        RETURN QUERY SELECT 0::BIGINT, 0::BIGINT, 0::BIGINT, 0::NUMERIC, 0::NUMERIC;
END;
$$ LANGUAGE plpgsql;

-- Create a simple logging table for database events
CREATE TABLE IF NOT EXISTS db_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initialization event
INSERT INTO db_events (event_type, message) 
VALUES ('INITIALIZATION', 'Database initialized successfully');

-- Create indexes that will be useful once tables are created
-- These will only run if the tables exist

DO $$
BEGIN
    -- Try to create indexes for tables that might exist
    BEGIN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp 
        ON ohlcv_data(symbol, timeframe, timestamp);
    EXCEPTION WHEN OTHERS THEN
        -- Ignore if table doesn't exist
        NULL;
    END;
    
    BEGIN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_timestamp 
        ON trades(symbol, timestamp);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
    
    BEGIN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_strategy_timestamp 
        ON trades(strategy, timestamp);
    EXCEPTION WHEN OTHERS THEN
        NULL;
    END;
END $$;