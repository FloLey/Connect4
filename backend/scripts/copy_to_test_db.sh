#!/bin/bash
set -e # Exit on error

# Configuration
DB_CONTAINER="db"
DB_USER="user"
SOURCE_DB="connect4_arena"
TARGET_DB="connect4_test"
BACKUP_FILE="backups/connect4_arena_20260104_143930.sql"

echo "📋 Checking backup file..."
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "🗑️  Clearing test database '$TARGET_DB'..."
docker compose exec -T $DB_CONTAINER psql -U $DB_USER -d $TARGET_DB -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

echo "📥 Restoring backup to test database..."
docker compose exec -T $DB_CONTAINER psql -U $DB_USER -d $TARGET_DB < $BACKUP_FILE

echo "✅ Success! Data copied from '$SOURCE_DB' to '$TARGET_DB'"

echo "📊 Verifying data copy..."
docker compose exec -T $DB_CONTAINER psql -U $DB_USER -d $TARGET_DB -c "
SELECT 'games' as table_name, COUNT(*) as row_count FROM games 
UNION ALL 
SELECT 'tournaments' as table_name, COUNT(*) as row_count FROM tournaments 
UNION ALL 
SELECT 'elo_ratings' as table_name, COUNT(*) as row_count FROM elo_ratings 
UNION ALL 
SELECT 'elo_history' as table_name, COUNT(*) as row_count FROM elo_history 
ORDER BY table_name;"