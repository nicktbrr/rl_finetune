import numpy as np
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn

class ManualRLEnvironment:
    def __init__(self, data: np.ndarray, labels: np.ndarray, original_predictions: np.ndarray, hidden_reps: np.ndarray):
        self.data = data
        self.labels = labels
        self.original_predictions = original_predictions
        self.hidden_reps = hidden_reps
        self.current_idx = 0
        self.classified_points = set()  # Track which points have been classified
        self.total_reward = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.remaining_indices = list(range(len(data)))  # Track remaining unclassified points
    
    def reset(self) -> Tuple[Dict, bool]:
        """Reset the environment."""
        self.current_idx = 0
        self.total_reward = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.classified_points = set()
        self.remaining_indices = list(range(len(self.data)))
        return self._get_state(), False
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Take a step in the environment."""
        # Get current point's true label
        true_label = self.labels[self.current_idx]
        
        # Calculate reward
        reward = 1.0 if action == true_label else -1.0
        self.total_reward += reward
        
        # Update metrics
        self.total_predictions += 1
        if action == true_label:
            self.correct_predictions += 1
        elif action == 1 and true_label == 0:
            self.false_positives += 1
        elif action == 0 and true_label == 1:
            self.false_negatives += 1
        
        # Mark current point as classified
        self.classified_points.add(self.current_idx)
        if self.current_idx in self.remaining_indices:
            self.remaining_indices.remove(self.current_idx)
        
        # Move to next unclassified point
        if self.remaining_indices:
            self.current_idx = self.remaining_indices[0]
            done = False
        else:
            done = True
        
        return self._get_state(), reward, done, self._get_info()
    
    def classify_points(self, indices: List[int], action: int) -> Tuple[Dict, float, bool, Dict]:
        """Classify multiple points at once."""
        total_reward = 0
        done = False
        
        for idx in indices:
            if idx not in self.classified_points:
                # Store current state
                current_idx = self.current_idx
                self.current_idx = idx
                
                # Classify the point
                state, reward, done, info = self.step(action)
                total_reward += reward
                
                if done:
                    break
        
        return self._get_state(), total_reward, done, self._get_info()
    
    def _get_state(self) -> Dict:
        """Get the current state."""
        return {
            'current_point': self.data[self.current_idx],
            'current_idx': self.current_idx,
            'true_label': self.labels[self.current_idx],
            'original_prediction': self.original_predictions[self.current_idx],
            'hidden_rep': self.hidden_reps[self.current_idx]
        }
    
    def _get_info(self) -> Dict:
        """Get information about the current state."""
        return {
            'total_reward': self.total_reward,
            'accuracy': self.correct_predictions / max(1, self.total_predictions),
            'false_positive_rate': self.false_positives / max(1, self.total_predictions),
            'false_negative_rate': self.false_negatives / max(1, self.total_predictions)
        }
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self._get_info()
    
    def get_unclassified_indices(self) -> List[int]:
        """Get indices of points that haven't been classified yet."""
        return self.remaining_indices 