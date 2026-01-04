-- Fallback Audit Query
-- Check which models are triggering fallback moves (random moves due to AI errors)

-- Query 1: Fallback rates per model (as player 1)
SELECT 
    player_1_type as model, 
    COUNT(*) as total_moves,
    SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END) as fallback_count,
    ROUND(
        (SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END)::float / COUNT(*)::float) * 100, 
        2
    ) as fallback_percentage
FROM (
    SELECT player_1_type, jsonb_array_elements(history) as move 
    FROM games 
    WHERE history IS NOT NULL AND jsonb_array_length(history) > 0
) sub
GROUP BY model
ORDER BY fallback_percentage DESC;

-- Query 2: Fallback rates per model (as player 2)
SELECT 
    player_2_type as model, 
    COUNT(*) as total_moves,
    SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END) as fallback_count,
    ROUND(
        (SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END)::float / COUNT(*)::float) * 100, 
        2
    ) as fallback_percentage
FROM (
    SELECT player_2_type, jsonb_array_elements(history) as move 
    FROM games 
    WHERE history IS NOT NULL AND jsonb_array_length(history) > 0
) sub
GROUP BY model
ORDER BY fallback_percentage DESC;

-- Query 3: Combined view (both players)
SELECT 
    model,
    SUM(total_moves) as total_moves,
    SUM(fallback_count) as total_fallback,
    ROUND((SUM(fallback_count)::float / SUM(total_moves)::float) * 100, 2) as overall_fallback_percentage
FROM (
    -- Player 1 moves
    SELECT 
        player_1_type as model, 
        COUNT(*) as total_moves,
        SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END) as fallback_count
    FROM (
        SELECT player_1_type, jsonb_array_elements(history) as move 
        FROM games 
        WHERE history IS NOT NULL AND jsonb_array_length(history) > 0
    ) sub1
    GROUP BY player_1_type
    
    UNION ALL
    
    -- Player 2 moves
    SELECT 
        player_2_type as model, 
        COUNT(*) as total_moves,
        SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END) as fallback_count
    FROM (
        SELECT player_2_type, jsonb_array_elements(history) as move 
        FROM games 
        WHERE history IS NOT NULL AND jsonb_array_length(history) > 0
    ) sub2
    GROUP BY player_2_type
) combined
GROUP BY model
ORDER BY overall_fallback_percentage DESC;

-- Query 4: Recent tournament fallback analysis (last 7 days)
SELECT 
    g.player_1_type as model,
    COUNT(*) as total_moves,
    SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END) as fallback_count,
    ROUND(
        (SUM(CASE WHEN (move->>'is_fallback')::boolean = true THEN 1 ELSE 0 END)::float / COUNT(*)::float) * 100, 
        2
    ) as fallback_percentage
FROM (
    SELECT g.player_1_type, jsonb_array_elements(g.history) as move 
    FROM games g
    JOIN tournaments t ON g.tournament_id = t.id
    WHERE g.history IS NOT NULL 
      AND jsonb_array_length(g.history) > 0
      AND t.created_at >= NOW() - INTERVAL '7 days'
) sub
GROUP BY model
ORDER BY fallback_percentage DESC
LIMIT 10;