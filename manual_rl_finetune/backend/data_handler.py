import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import joblib
from pathlib import Path
import warnings
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import logging
import os

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.feature_names = None
        self.hidden_reps = None
        self.sample_size = 1000  # Limit sample size for memory management
        self.baseline_model = None
        self.data = None  # Store the data
        self.original_predictions = None  # Store original predictions
        self.scaler = StandardScaler()  # For feature normalization
        
    def load_baseline_model(self, model_path: str) -> None:
        """Load the baseline model."""
        try:
            self.baseline_model = torch.load(model_path)
            self.baseline_model.eval()  # Set to evaluation mode
        except Exception as e:
            warnings.warn(f"Could not load baseline model: {e}")
            self.baseline_model = None
    
    def get_baseline_predictions(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and hidden representations from the baseline model."""
        if self.baseline_model is None:
            warnings.warn("No baseline model loaded. Using random predictions.")
            return np.random.rand(len(data)), np.random.randn(len(data), 64)
        
        # Convert data to tensor
        data_tensor = torch.FloatTensor(data)
        
        # Get predictions and hidden representations
        with torch.no_grad():
            predictions, hidden_reps = self.baseline_model(data_tensor)
            predictions = predictions.numpy()
            hidden_reps = hidden_reps.numpy()
        
        return predictions, hidden_reps
    
    def load_data(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from the specified path or use sample data.
        Returns: (data, labels, original_predictions)
        """
        if data_path:
            try:
                logger.info(f"Loading data from {data_path}")
                
                # First set up feature names
                self.feature_names = [
                    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'spkts', 'dpkts',
                    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload',
                    'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
                    'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime',
                    'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
                    'ct_state_ttl', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
                    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                    'ct_dst_src_ltm', 'attack_cat', 'label'
                ]
                logger.info(f"Using UNSW-NB15 feature names: {self.feature_names}")
                
                # Load the data and labels
                try:
                    self.data = np.load(os.path.join(data_path, "X_train.npy"))
                    labels = np.load(os.path.join(data_path, "y_train.npy"))
                    logger.info(f"Loaded data with shape: {self.data.shape}")
                except Exception as e:
                    logger.error(f"Error loading data files: {str(e)}")
                    raise
                
                # Load or generate predictions
                try:
                    self.original_predictions = np.load(os.path.join(data_path, "predictions.npy"))
                    logger.info("Loaded original predictions")
                except FileNotFoundError:
                    logger.warning("No predictions file found, using random predictions")
                    self.original_predictions = np.random.random(len(self.data))
                
                # Load or generate hidden representations
                try:
                    self.hidden_reps = np.load(os.path.join(data_path, "hidden_reps.npy"))
                    logger.info("Loaded hidden representations")
                except FileNotFoundError:
                    logger.warning("No hidden representations found, using PCA of original data")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(50, self.data.shape[1]))
                    self.hidden_reps = pca.fit_transform(self.data)
                
                return self.data, labels, self.original_predictions
                
            except Exception as e:
                logger.error(f"Error in load_data: {str(e)}", exc_info=True)
                raise
        else:
            raise ValueError("No data path provided")
    
    def calculate_feature_importance(self, point: np.ndarray, current_idx: int, n_features: int = 10) -> List[Tuple[str, float, float]]:
        """
        Calculate feature importance for a given point.
        Args:
            point: The feature vector
            current_idx: The index of the current point
            n_features: Number of important features to return
        Returns: List of (feature_name, importance_score, feature_value) tuples
        """
        if self.feature_names is None or self.data is None:
            return []
            
        try:
            # Convert point to numpy array if it's a list
            if isinstance(point, list):
                point = np.array(point)
                
            # Normalize the point
            point_normalized = self.scaler.fit_transform(point.reshape(1, -1)).flatten()
            
            # Calculate importance scores based on:
            # 1. Absolute value of the feature (normalized)
            # 2. Distance from mean of that feature
            # 3. Original prediction confidence
            importance_scores = []
            for i, (name, value) in enumerate(zip(self.feature_names, point)):
                # Get feature statistics
                feature_values = self.data[:, i]
                feature_mean = np.mean(feature_values)
                feature_std = np.std(feature_values)
                
                # Calculate z-score
                z_score = abs((value - feature_mean) / (feature_std + 1e-10))
                
                # Calculate importance score
                importance = z_score * abs(point_normalized[i])
                
                # If we have original predictions, incorporate them
                if self.original_predictions is not None:
                    pred_confidence = abs(self.original_predictions[current_idx] - 0.5) * 2
                    importance *= (1 + pred_confidence)
                
                importance_scores.append((name, importance, value))
            
            # Sort by importance and return top n_features
            importance_scores.sort(key=lambda x: x[1], reverse=True)
            return importance_scores[:n_features]
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}", exc_info=True)
            return []
    
    def get_point_details(self, point: np.ndarray, current_idx: int, n_important_features: int = 10) -> dict:
        """
        Get feature names and values for a point, focusing on important features.
        Args:
            point: The feature vector
            current_idx: The index of the current point
            n_important_features: Number of important features to return
        """
        try:
            if self.feature_names is None:
                return {f"Feature_{i}": {"value": val, "importance": 1.0} for i, val in enumerate(point)}
                
            # Get important features
            important_features = self.calculate_feature_importance(point, current_idx, n_important_features)
            
            # Create dictionary with feature values and importance scores
            details = {}
            for name, importance, value in important_features:
                details[name] = {
                    'value': float(value),  # Convert to float for JSON serialization
                    'importance': float(importance)  # Convert to float for JSON serialization
                }
                
            return details
            
        except Exception as e:
            logger.error(f"Error getting point details: {str(e)}", exc_info=True)
            return {}
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        if self.feature_names is None and self.data is not None:
            return [f"Feature_{i}" for i in range(self.data.shape[1])]
        return self.feature_names
    
    def save_data(self, data: np.ndarray, labels: np.ndarray, 
                 original_predictions: np.ndarray, filename: str) -> None:
        """Save the data to a file."""
        save_path = self.data_dir / filename
        np.save(save_path, {
            'data': data,
            'labels': labels,
            'original_predictions': original_predictions
        })
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        return joblib.load(model_path)
    
    def save_model(self, model, model_path: str) -> None:
        """Save a trained model."""
        joblib.dump(model, model_path)
    
    def get_data_info(self, data: np.ndarray) -> dict:
        """Get information about the dataset."""
        return {
            'n_samples': len(data),
            'n_features': data.shape[1],
            'feature_ranges': {
                'min': data.min(axis=0),
                'max': data.max(axis=0),
                'mean': data.mean(axis=0),
                'std': data.std(axis=0)
            }
        }
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess the data (normalization, etc.)."""
        # Normalize the data
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (data - mean) / std 