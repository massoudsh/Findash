#!/bin/bash

# Octopus Trading Platform - Development Startup Script
# Run from repo root: ./scripts/start-dev.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "🐙 Starting Octopus Trading Platform Development Environment..."

if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp config/env.example .env
    echo "✅ .env file created. Please update with your API keys if needed."
fi

cleanup() {
    echo "🛑 Shutting down servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM EXIT

echo "🚀 Starting Backend API Server (Python)..."
python3 start.py --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

echo "🌐 Starting Frontend Development Server (Next.js)..."
cd frontend-nextjs
npm run dev &
FRONTEND_PID=$!

sleep 3

echo ""
echo "✅ Development servers are running:"
echo "   🔗 Frontend: http://localhost:3000"
echo "   🔗 Backend API: http://localhost:8000"
echo "   📚 API Docs: http://localhost:8000/docs"
echo ""
echo "   Dashboard: http://localhost:3000/dashboard"
echo ""
echo "💡 Press Ctrl+C to stop all servers"

wait
