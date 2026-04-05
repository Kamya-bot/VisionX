#!/bin/bash

# VisionX Health Check Script
# Verifies all services are running correctly

echo "🏥 VisionX Health Check"
echo "======================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check function
check_service() {
    local name=$1
    local url=$2
    
    if curl -f -s "$url" > /dev/null; then
        echo -e "${GREEN}✅ $name is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ $name is down${NC}"
        return 1
    fi
}

# Backend health
check_service "Backend API" "http://localhost:8000/health"

# Frontend
check_service "Frontend" "http://localhost/"

# Database (via backend)
if curl -f -s "http://localhost:8000/api/v1/health" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Database connection is healthy${NC}"
else
    echo -e "${RED}❌ Database connection failed${NC}"
fi

# ML Models
if curl -f -s "http://localhost:8000/api/v1/ml/metrics" > /dev/null; then
    echo -e "${GREEN}✅ ML models loaded${NC}"
else
    echo -e "${YELLOW}⚠️  ML models not loaded${NC}"
fi

# Drift Detection
if curl -f -s "http://localhost:8000/api/v1/drift/summary" > /dev/null; then
    echo -e "${GREEN}✅ Drift detection active${NC}"
else
    echo -e "${YELLOW}⚠️  Drift detection unavailable (need more predictions)${NC}"
fi

# Model Versioning
if curl -f -s "http://localhost:8000/api/v1/models/versions" > /dev/null; then
    echo -e "${GREEN}✅ Model versioning active${NC}"
else
    echo -e "${YELLOW}⚠️  Model versioning unavailable${NC}"
fi

echo ""
echo "Health check complete!"
