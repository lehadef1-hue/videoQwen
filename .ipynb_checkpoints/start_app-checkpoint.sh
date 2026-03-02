#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
# export PERFORMER_DB_PATH="$PROJECT_DIR/performers_db.pkl"
export PERFORMER_DB_PATH="/workspace/my_performers.pkl"
cd "$PROJECT_DIR"
uvicorn video_processor:app --host 0.0.0.0 --port 8000
