#!/bin/bash
set -e # Exit on error

# Configuration
DB_CONTAINER="db" # Name of service in docker-compose
DB_USER="user"
DB_NAME="connect4_arena"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backups/${DB_NAME}_${TIMESTAMP}.sql"

# Ensure backup dir exists
mkdir -p backups

echo "🔒 Freezing Backend..."
docker compose stop backend

echo "📦 Dumping Database '$DB_NAME'..."
docker compose exec -T $DB_CONTAINER pg_dump -U $DB_USER $DB_NAME > $BACKUP_FILE

if [ -s "$BACKUP_FILE" ]; then
    echo "✅ Success! Backup saved to: $BACKUP_FILE"
    echo "📄 Size: $(du -h $BACKUP_FILE | cut -f1)"
else
    echo "❌ Error: Backup file is empty."
    exit 1
fi

echo "▶️  Restarting Backend..."
docker compose start backend