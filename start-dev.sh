#!/bin/bash

# Octopus Trading Platform - Development Startup Script
echo "🐙 Starting Octopus Trading Platform Development Environment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created. Please update with your API keys if needed."
fi

# Function to kill background processes on exit
cleanup() {
    echo "🛑 Shutting down servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set up cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start backend server
echo "🚀 Starting Backend API Server (Python)..."
python3 start.py --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "🌐 Starting Frontend Development Server (Next.js)..."
cd frontend-nextjs
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
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

# Wait for background processes
wait 