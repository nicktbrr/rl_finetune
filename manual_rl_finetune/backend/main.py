from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import os
import logging
from data_handler import DataHandler
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manual_env import ManualRLEnvironment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data handler and environment
data_handler = None
env = None

@app.on_event("startup")
async def startup_event():
    global data_handler, env
    try:
        # Initialize data handler
        data_handler = DataHandler()
        
        # Get the absolute path to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "..", "data")
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        data, labels, predictions = data_handler.load_data(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Initialize environment with hidden representations
        env = ManualRLEnvironment(data, labels, predictions, data_handler.hidden_reps)
        logger.info("Environment initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.get("/current_state")
async def get_current_state():
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
            
        state = env.get_state()
        info = env.get_info()
        
        # Get point details from data handler
        point_details = data_handler.get_point_details(
            state["current_point"],
            state["current_idx"],
            n_important_features=10
        )
        
        return {
            "state": state,
            "info": info,
            "point_details": point_details
        }
        
    except Exception as e:
        logger.error(f"Error getting current state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/unclassified_points")
async def get_unclassified_points():
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
            
        # Return the indices directly, not the points
        return {"points": env.unclassified_indices}
        
    except Exception as e:
        logger.error(f"Error getting unclassified points: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class ClassificationRequest(BaseModel):
    points: List[int]
    label: int

@app.post("/classify_points")
async def classify_points(request: ClassificationRequest):
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
            
        # Classify points
        state, reward, done, info = env.classify_points(request.points, request.label)
        
        # Get point details
        point_details = data_handler.get_point_details(
            state["current_point"],
            state["current_idx"],
            n_important_features=10
        )
        
        # Get updated training history
        history = env.get_training_history()
        
        return {
            "state": state,
            "reward": reward,
            "done": done,
            "info": info,
            "point_details": point_details,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error classifying points: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training_history")
async def get_training_history():
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
            
        history = env.get_training_history()
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
        
        # Get current classified points and their labels
        history = env.get_training_history()
        if not history:
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "false_positives": 0,
                "false_negatives": 0,
                "total_classified": 0,
                "remaining_unclassified": len(env.get_unclassified_points())
            }
            
        # Extract true and predicted labels from history
        true_labels = [entry["true_label"] for entry in history]
        predicted_labels = [entry["predicted_label"] for entry in history]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
        
        # Get detailed counts
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_classified": len(history),
            "remaining_unclassified": len(env.get_unclassified_points())
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset():
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
            
        state = env.reset()
        info = env.get_info()
        
        # Get point details
        point_details = data_handler.get_point_details(
            state["current_point"],
            state["current_idx"],
            n_important_features=10
        )
        
        return {
            "state": state,
            "info": info,
            "point_details": point_details
        }
        
    except Exception as e:
        logger.error(f"Error resetting environment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/point_details/{point_index}")
async def get_point_details(point_index: int):
    """Get detailed information about a specific point."""
    try:
        if env is None:
            raise HTTPException(status_code=500, detail="Environment not initialized")
        
        if point_index < 0 or point_index >= len(env.data):
            raise HTTPException(status_code=404, detail=f"Point index {point_index} out of range")
        
        # Get the point data
        point_data = env.data[point_index].tolist()
        
        # Get detailed feature information
        point_details = data_handler.get_point_details(
            point_data,
            point_index,
            n_important_features=15  # Return more features for detailed view
        )
        
        # Include original model prediction
        prediction = float(env.predictions[point_index])
        
        # Include label if the point has been classified
        true_label = int(env.labels[point_index]) if point_index < len(env.labels) else None
        
        return {
            "point_index": point_index,
            "point_data": point_data,
            "feature_details": point_details,
            "prediction": prediction,
            "true_label": true_label,
            "hidden_rep": env.hidden_reps[point_index].tolist() if env.hidden_reps is not None else None
        }
        
    except Exception as e:
        logger.error(f"Error getting point details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 