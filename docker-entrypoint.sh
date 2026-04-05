#!/bin/sh
# Inject the backend API URL into a runtime JS config file
# so the frontend knows where to call without rebuilding the image

API_URL="${VISIONX_API_URL:-http://localhost:8000}"

cat > /usr/share/nginx/html/js/runtime-config.js << JSEOF
// Runtime configuration — injected by Docker entrypoint
window.VISIONX_API_URL = '${API_URL}';
JSEOF

echo "VisionX frontend: API URL set to ${API_URL}"
exec "$@"
