#!/bin/bash

# VisionX Deployment Script
# Automates the deployment process

set -e  # Exit on error

echo "🚀 VisionX Deployment Script"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Check for .env file
if [ ! -f .env ]; then
    print_info ".env file not found. Copying from .env.example..."
    cp .env.example .env
    print_info "Please edit .env file with your configuration before continuing."
    read -p "Press Enter after you've updated .env file..."
fi

print_success ".env file found"

# Stop existing containers
print_info "Stopping existing containers..."
docker-compose down 2>/dev/null || true
print_success "Existing containers stopped"

# Build images
print_info "Building Docker images..."
docker-compose build --no-cache
print_success "Docker images built successfully"

# Start services
print_info "Starting services..."
docker-compose up -d
print_success "Services started"

# Wait for backend to be healthy
print_info "Waiting for backend to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "Backend is ready!"
        break
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    print_error "Backend failed to start within expected time"
    print_info "Check logs with: docker-compose logs backend"
    exit 1
fi

# Initialize database
print_info "Initializing database..."
docker-compose exec -T backend python init_database.py || true
print_success "Database initialized"

# Train ML models (if not already trained)
print_info "Checking ML models..."
if ! docker-compose exec -T backend test -f /app/models/prediction_model.joblib; then
    print_info "ML models not found. Training..."
    docker-compose exec -T backend python training/train_models.py
    print_success "ML models trained"
else
    print_info "ML models already exist"
fi

# Show status
echo ""
echo "=============================="
print_success "VisionX Deployment Complete!"
echo "=============================="
echo ""
echo "📊 Service URLs:"
echo "  Frontend:  http://localhost"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Health:    http://localhost:8000/health"
echo ""
echo "🔧 Useful Commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart services: docker-compose restart"
echo "  View status:      docker-compose ps"
echo ""
print_info "Check logs with: docker-compose logs -f"
