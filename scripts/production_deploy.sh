#!/bin/bash

# VisionX Production Deployment Script
# For deploying to cloud providers (AWS/GCP/Azure)

set -e

echo "🚀 VisionX Production Deployment"
echo "================================="
echo ""

# Load environment
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
else
    echo "❌ .env.production not found!"
    exit 1
fi

# Deployment type
read -p "Choose deployment platform (1=AWS, 2=GCP, 3=Azure, 4=Custom): " PLATFORM

case $PLATFORM in
    1)
        echo "📦 Deploying to AWS..."
        ./scripts/deploy_aws.sh
        ;;
    2)
        echo "📦 Deploying to GCP..."
        ./scripts/deploy_gcp.sh
        ;;
    3)
        echo "📦 Deploying to Azure..."
        ./scripts/deploy_azure.sh
        ;;
    4)
        echo "📦 Custom deployment..."
        echo "Please run your custom deployment script"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "✅ Production deployment complete!"
echo "🔗 Your app should be available at: https://$DOMAIN"
