-- Add index for retry_after column to optimize tournament tick() queries
-- This index is needed for efficient queries on PAUSED games with expired retry_after timers

-- Check if index already exists
SELECT indexname FROM pg_indexes WHERE tablename = 'games' AND indexname LIKE '%retry_after%';

-- Create index if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_games_retry_after ON games(retry_after);

-- Also create a composite index for tournament queries (optional but recommended)
CREATE INDEX IF NOT EXISTS idx_games_tournament_status_retry 
ON games(tournament_id, status, retry_after) 
WHERE status = 'PAUSED';

-- Verify indexes were created
SELECT 
    indexname, 
    indexdef 
FROM pg_indexes 
WHERE tablename = 'games' 
ORDER BY indexname;