#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start the backend server
echo "Starting FastAPI backend..."
cd "$DIR/backend"
PYTHONPATH="$DIR" python -m uvicorn main:app --reload &
BACKEND_PID=$!

# Start the frontend development server
echo "Starting React frontend..."
cd "$DIR/frontend"
npm run start &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

# Set up trap to catch termination signal
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait 