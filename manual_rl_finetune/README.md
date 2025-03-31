# Manual RL Finetune

A tool for manually classifying network traffic data to improve a baseline model's predictions through interactive visualization and human feedback.

## Overview

This project provides an interactive web interface for network traffic classification, allowing users to:

1. Visualize network traffic data in a 2D PCA projection
2. View model predictions for each data point
3. Manually classify points as benign or attack traffic
4. Track performance improvements over the baseline model
5. Analyze feature importance for individual points

The interface uses human feedback to fine-tune the model's predictions, similar to a Reinforcement Learning (RL) approach but with manual supervision.

## Features

- **Mass Classification**: Select and classify multiple data points at once
- **Single Point Classification**: Analyze and classify one point at a time with detailed feature information
- **Feature Analysis**: View the most important features for each point to aid classification decisions
- **Performance Metrics**: Track accuracy, precision, recall, and F1 score improvements over the baseline model
- **Classification History**: Review previous classification decisions and their correctness
- **Interactive Visualization**: PCA-based 2D visualization with color coding for attack probability

## System Architecture

The project consists of:

- **Backend**: FastAPI server that manages the environment and provides data endpoints
- **Frontend**: React-based web interface with Plotly for interactive visualization
- **Environment**: RL-inspired environment that tracks state and rewards for classifications

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd manual_rl_finetune/backend
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd manual_rl_finetune/frontend
   ```

2. Install Node.js dependencies:
   ```
   npm install
   ```

## Running the Application

1. Use the provided run script to start both backend and frontend:
   ```
   cd manual_rl_finetune
   ./run.sh
   ```

2. This will start:
   - FastAPI backend on http://localhost:8000
   - React frontend on http://localhost:3000

3. Access the web interface in your browser at http://localhost:3000

## Usage Guide

### Mass Classification View

1. The main visualization shows data points colored by their attack probability (red = attack, green = benign)
2. Select points by clicking or dragging a box around them
3. Click "Classify as Benign" or "Classify as Attack" to classify the selected points
4. Hover over points to see detailed feature information
5. Pin a point to keep its feature details displayed

### Single Point View

1. The application will select a random unclassified point
2. View the model's prediction and detailed feature analysis
3. Make a classification decision based on the evidence
4. Points will be presented sequentially for classification

### Metrics Tab

1. Track improvements over the baseline model
2. View confusion matrix and detailed performance statistics
3. Monitor progress through the dataset

### History Tab

View a record of all classifications made, including:
- Point ID
- True label
- Predicted label
- Correctness of prediction
- Reward received

## How It Works

1. The system uses a pre-trained baseline model to generate initial predictions
2. As you classify points, the environment tracks the accuracy of your decisions
3. The interface provides real-time feedback on performance improvements
4. Feature importance helps guide classification decisions
5. The training history serves as a dataset for potential automated fine-tuning

## Project Structure

- `backend/` - FastAPI server files
  - `main.py` - Primary API endpoints
  - `data_handler.py` - Data loading and feature importance calculation
  
- `frontend/` - React application
  - `src/App.jsx` - Main application logic and UI components
  - `src/App.css` - Styling for the application
  
- `manual_env.py` - RL environment implementation
- `run.sh` - Script to start both frontend and backend services

## License

This project is for educational and research purposes. 