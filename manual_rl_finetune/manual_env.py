import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class ManualRLEnvironment:
    def __init__(self, data: np.ndarray, labels: np.ndarray, predictions: np.ndarray, hidden_reps: Optional[np.ndarray] = None):
        """
        Initialize the environment for manual RL training.
        Args:
            data: Feature vectors for all points
            labels: True labels (0 for benign, 1 for attack)
            predictions: Model predictions for each point
            hidden_reps: Optional hidden representations for visualization
        """
        self.data = data
        self.labels = labels
        self.predictions = predictions
        
        # Generate hidden representations if not provided
        if hidden_reps is None:
            logger.info("Generating hidden representations using PCA")
            pca = PCA(n_components=min(50, data.shape[1]))
            self.hidden_reps = pca.fit_transform(data)
        else:
            self.hidden_reps = hidden_reps
        
        # Initialize state variables
        self.classified_indices = []
        self.unclassified_indices = list(range(len(data)))
        self.current_idx = 0
        self.current_point = self.data[self.current_idx]
        
        # Initialize metrics
        self.total_reward = 0.0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Track training history
        self.training_history = []
        
        logger.info(f"Initialized environment with {len(data)} points")
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        self.classified_indices = []
        self.unclassified_indices = list(range(len(self.data)))
        self.current_idx = 0
        self.current_point = self.data[self.current_idx]
        
        # Reset metrics
        self.total_reward = 0.0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Clear training history
        self.training_history = []
        
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment."""
        # Get the current point
        current_point = self.data[self.current_idx]
        
        # Get the remaining unclassified points
        remaining_points = self.data[self.unclassified_indices]
        
        # Get corresponding hidden representations if available
        hidden_reps = None
        if self.hidden_reps is not None:
            hidden_reps = self.hidden_reps[self.unclassified_indices]
        
        # Get all predictions
        all_predictions = self.predictions
        
        # Get features for analysis
        features = self.data
        
        state = {
            "current_point": current_point.tolist(),
            "current_idx": self.current_idx,
            "remaining_points": remaining_points.tolist(),
            "hidden_reps": hidden_reps.tolist() if hidden_reps is not None else None,
            "features": features.tolist(),
            "original_predictions": all_predictions.tolist(),
            "predictions": all_predictions.tolist(),
            "classified_indices": self.classified_indices,
            "unclassified_indices": self.unclassified_indices,
        }
        
        return state
    
    def get_info(self) -> Dict[str, Any]:
        """Get current metrics and information about the environment."""
        accuracy = self.correct_predictions / max(1, self.total_predictions) if self.total_predictions > 0 else 0
        
        return {
            "total_reward": float(self.total_reward),
            "accuracy": float(accuracy),
            "total_predictions": int(self.total_predictions),
            "correct_predictions": int(self.correct_predictions),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "remaining_points": len(self.unclassified_indices),
            "classified_points": len(self.classified_indices),
            "unclassified_points": len(self.unclassified_indices),
            "current_step": len(self.classified_indices),
        }
    
    def get_unclassified_points(self) -> np.ndarray:
        """Get the unclassified point indices."""
        return np.array(self.unclassified_indices)
    
    def classify_points(self, point_indices: List[int], label: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Classify multiple points with the given label.
        Args:
            points: List of point indices to classify
            label: Label to assign (0 for benign, 1 for attack)
        Returns:
            (state, reward, done, info)
        """
        if not point_indices:
            logger.warning("No points provided for classification")
            return self.get_state(), 0.0, False, self.get_info()
        
        reward = 0.0
        for idx in point_indices:
            # Ensure the point is in unclassified_indices
            if idx not in self.unclassified_indices:
                logger.warning(f"Point {idx} already classified, skipping")
                continue
            
            # Get true label and predicted label
            true_label = int(self.labels[idx])
            predicted_label = label
            
            # Calculate reward based on correct/incorrect classification
            point_reward = 1.0 if true_label == predicted_label else -1.0
            reward += point_reward
            
            # Update metrics
            self.total_reward += point_reward
            self.total_predictions += 1
            if true_label == predicted_label:
                self.correct_predictions += 1
                self.classified_indices.append(idx)
            elif predicted_label == 1 and true_label == 0:
                self.false_positives += 1
                self.classified_indices.append(idx)
            elif predicted_label == 0 and true_label == 1:
                self.false_negatives += 1
                self.classified_indices.append(idx)
            
            # Add to training history
            self.training_history.append({
                "index": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
                "reward": point_reward
            })
            
            # Remove from unclassified_indices and add to classified_indices
            self.unclassified_indices.remove(idx)
        
        # Update current_idx to next unclassified point if available
        if self.unclassified_indices:
            self.current_idx = self.unclassified_indices[0]
            self.current_point = self.data[self.current_idx]
        
        # Check if done (all points classified)
        done = len(self.unclassified_indices) == 0
        
        # Get updated state and info
        state = self.get_state()
        info = self.get_info()
        
        return state, reward, done, info
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the history of all classifications made so far."""
        return self.training_history 