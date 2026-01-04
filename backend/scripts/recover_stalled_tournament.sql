-- Recovery Script for Stalled Tournaments
-- Use this to manually fix tournaments that were prematurely marked as COMPLETED

-- 1. First, identify stalled tournaments
SELECT 
    id,
    status,
    created_at,
    total_matches,
    config->>'type' as tournament_type,
    (
        SELECT COUNT(*) 
        FROM games g 
        WHERE g.tournament_id = t.id 
        AND g.status IN ('PENDING', 'IN_PROGRESS', 'PAUSED')
    ) as unfinished_games
FROM tournaments t
WHERE status = 'COMPLETED'
ORDER BY created_at DESC
LIMIT 5;

-- 2. Check the specific tournament's game status breakdown
-- Replace [TOURNAMENT_ID] with the actual tournament ID
SELECT 
    status,
    COUNT(*) as game_count,
    CASE 
        WHEN status = 'PAUSED' THEN 
            STRING_AGG(
                CASE 
                    WHEN retry_after IS NULL THEN 'No retry_after'
                    WHEN retry_after <= NOW() THEN 'Expired: ' || retry_after::text
                    ELSE 'Active until: ' || retry_after::text
                END, 
                ', '
            )
        ELSE ''
    END as details
FROM games
WHERE tournament_id = [TOURNAMENT_ID]  -- Replace with actual ID
GROUP BY status
ORDER BY status;

-- 3. Reactivate the tournament (if you confirm it has unfinished games)
-- Replace [TOURNAMENT_ID] with the actual tournament ID
-- UPDATE tournaments 
-- SET status = 'IN_PROGRESS' 
-- WHERE id = [TOURNAMENT_ID] 
-- AND status = 'COMPLETED';

-- 4. Check PAUSED games with retry_after timers
SELECT 
    id,
    player_1_type,
    player_2_type,
    status,
    retry_after,
    CASE 
        WHEN retry_after IS NULL THEN 'No timer set'
        WHEN retry_after <= NOW() THEN 'READY to resume'
        ELSE 'Will be ready at: ' || retry_after::text
    END as resume_status
FROM games
WHERE tournament_id = [TOURNAMENT_ID]  -- Replace with actual ID
    AND status = 'PAUSED'
ORDER BY retry_after ASC;

-- 5. Optional: Force resume PAUSED games with expired timers
-- UPDATE games
-- SET status = 'PENDING', retry_after = NULL
-- WHERE tournament_id = [TOURNAMENT_ID]  -- Replace with actual ID
--     AND status = 'PAUSED'
--     AND retry_after <= NOW();

-- 6. After recovery, verify the tournament can proceed
-- Run the diagnostic script: python backend/scripts/tournament_diagnostic.py
-- The tournament_watcher should now detect and resume the games