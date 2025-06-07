#!/usr/bin/env bash
set -euo pipefail

echo "🗂  Raw-data directory sample (first 30 lines):"
# Ignore the SIGPIPE from ls → head
ls -R /app/data | head -n 30 || true
echo "--------------------------------------------------"

echo "🚀 Converting raw data to Parquet …"
python /app/feast_demo/feature_repo/convert_data.py

echo "🗄️  Running feast apply …"
cd /app/feast_demo/feature_repo
feast apply

echo "🤖 Starting training pipeline …"
cd /app
python main.py
