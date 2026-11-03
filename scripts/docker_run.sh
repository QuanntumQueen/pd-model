#!/bin/bash
# Build and run the PD-model API in Docker (Linux / Mac)
set -e
docker build -t pd-model .
docker rm -f pd-model 2>/dev/null || true
docker run -d --name pd-model -p 8000:8000 pd-model
echo "API is running: http://localhost:8000/docs"
