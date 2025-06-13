#!/bin/bash

# URL Analyzer startup script

set -e

echo "🔍 URL Analyzer Startup Script"
echo "==============================="

# Check if we're in the correct directory
if [ ! -f "backend/main.py" ] || [ ! -f "frontend/app.py" ]; then
    echo "❌ Error: Please run this script from the url_analyzer directory"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to start backend
start_backend() {
    echo "🚀 Starting backend on port 8002..."
    if ! check_port 8002; then
        echo "❌ Backend port 8002 is already in use. Please stop the existing service first."
        return 1
    fi
    
    cd backend
    python main.py &
    BACKEND_PID=$!
    cd ..
    
    echo "✅ Backend started with PID $BACKEND_PID"
    return 0
}

# Function to start frontend
start_frontend() {
    echo "🚀 Starting frontend on port 8501..."
    if ! check_port 8501; then
        echo "⚠️  Frontend port 8501 is already in use. Attempting to stop existing service..."
        pkill -f "streamlit run" || true
        sleep 2
    fi
    
    cd frontend
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
    FRONTEND_PID=$!
    cd ..
    
    echo "✅ Frontend started with PID $FRONTEND_PID"
    return 0
}

# Function to check dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo "❌ Python is not installed"
        return 1
    fi
    
    # Check pip packages
    python -c "import fastapi, streamlit, requests" 2>/dev/null || {
        echo "⚠️  Some required packages are missing. Installing..."
        pip install -r requirements.txt
    }
    
    echo "✅ Dependencies check completed"
    return 0
}

# Function to wait for backend to be ready
wait_for_backend() {
    echo "⏳ Waiting for backend to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8002/health >/dev/null 2>&1; then
            echo "✅ Backend is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts: Backend not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ Backend failed to start within 60 seconds"
    return 1
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "   Backend stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "   Frontend stopped"
    fi
    
    # Also kill any remaining streamlit processes
    pkill -f "streamlit run" 2>/dev/null || true
    
    echo "✅ Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    local mode=${1:-"both"}
    
    case $mode in
        "backend")
            echo "Starting backend only..."
            check_dependencies
            start_backend
            echo ""
            echo "🎉 Backend is running!"
            echo "   API: http://localhost:8002"
            echo "   Docs: http://localhost:8002/docs"
            echo ""
            echo "Press Ctrl+C to stop"
            wait
            ;;
            
        "frontend")
            echo "Starting frontend only..."
            check_dependencies
            start_frontend
            echo ""
            echo "🎉 Frontend is running!"
            echo "   UI: http://localhost:8501"
            echo ""
            echo "Press Ctrl+C to stop"
            wait
            ;;
            
        "both"|*)
            echo "Starting both backend and frontend..."
            check_dependencies
            
            if start_backend; then
                wait_for_backend
                start_frontend
                
                echo ""
                echo "🎉 URL Analyzer is running!"
                echo "================================"
                echo "   Frontend: http://localhost:8501"
                echo "   Backend:  http://localhost:8002"
                echo "   API Docs: http://localhost:8002/docs"
                echo ""
                echo "Press Ctrl+C to stop both services"
                
                wait
            else
                echo "❌ Failed to start backend"
                exit 1
            fi
            ;;
    esac
}

# Show usage if help is requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [mode]"
    echo ""
    echo "Modes:"
    echo "  both      Start both backend and frontend (default)"
    echo "  backend   Start backend only"
    echo "  frontend  Start frontend only"
    echo ""
    echo "Examples:"
    echo "  $0              # Start both services"
    echo "  $0 both         # Start both services"
    echo "  $0 backend      # Start backend only"
    echo "  $0 frontend     # Start frontend only"
    exit 0
fi

# Run main function
main "$@"