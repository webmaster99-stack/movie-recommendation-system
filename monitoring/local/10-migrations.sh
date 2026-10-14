#!/bin/sh
# Apply every migration, oldest first, as Supabase does.
set -e
for migration in /migrations/*.sql; do
  echo "applying $migration"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$migration"
done
