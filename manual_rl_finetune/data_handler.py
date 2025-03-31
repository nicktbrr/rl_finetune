import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import joblib
from pathlib import Path
import warnings
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


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

    def load_data(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from the specified path or use sample data.
        Returns: (data, labels, original_predictions)
        """
        if data_path:
            try:
                # Load data from CSV
                df = pd.read_csv(f"{data_path}/perturbed_train_data.csv")

                # Get feature names from DataFrame columns
                self.feature_names = df.columns.tolist()

                # Separate features and labels
                # Assuming the last column is the label
                self.data = df.iloc[:, :-1].values
                labels = df.iloc[:, -1].values

                # Sample data if too large
                if len(self.data) > self.sample_size:
                    indices = np.random.choice(
                        len(self.data), self.sample_size, replace=False)
                    self.data = self.data[indices]
                    labels = labels[indices]
                    warnings.warn(
                        f"Sampled {self.sample_size} points from {len(self.data)} total points")

                return self.data, labels

            except Exception as e:
                raise Exception(f"Error loading data: {str(e)}")
        else:
            # Use sample data if no path provided
            self.data = np.random.randn(100, 10)
            labels = np.random.randint(0, 2, 100)
            original_predictions = np.random.rand(100)
            return self.data, labels, original_predictions

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

        # Normalize the point
        point_normalized = self.scaler.fit_transform(
            point.reshape(1, -1)).flatten()

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
            z_score = abs((value - feature_mean) / feature_std)

            # Calculate importance score
            importance = z_score * abs(point_normalized[i])

            # If we have original predictions, incorporate them
            if self.original_predictions is not None:
                pred_confidence = abs(
                    self.original_predictions[current_idx] - 0.5) * 2
                importance *= (1 + pred_confidence)

            importance_scores.append((name, importance, value))

        # Sort by importance and return top n_features
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return importance_scores[:n_features]

    def get_point_details(self, point: np.ndarray, current_idx: int, n_important_features: int = 10) -> Dict[str, float]:
        """
        Get feature names and values for a point, focusing on important features.
        Args:
            point: The feature vector
            current_idx: The index of the current point
            n_important_features: Number of important features to return
        """
        if self.feature_names is None:
            return {f"Feature_{i}": val for i, val in enumerate(point)}

        # Get important features
        important_features = self.calculate_feature_importance(
            point, current_idx, n_important_features)

        # Create dictionary with feature values and importance scores
        details = {}
        for name, importance, value in important_features:
            details[name] = {
                'value': value,
                'importance': importance
            }

        return details

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
