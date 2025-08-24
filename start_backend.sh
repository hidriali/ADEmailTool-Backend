#!/bin/bash
# Start backend with activated environment

echo "Activating Python environment..."
source /Users/idrisalo/emailnlp-env/bin/activate

echo "Starting backend service..."
cd /Users/idrisalo/emailtool-new
python3 advanced_ai_service.py
