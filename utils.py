import os
import numpy as np
import joblib
import mlflow
from stable_baselines3 import TD3


def save_data(baseline_probs, hidden_reps, true_labels, output_dir="./data"):
    """
    Save prediction data to disk for later use

    Args:
        baseline_probs: Original prediction probabilities from base model
        hidden_reps: Hidden representations from base model
        true_labels: Ground truth labels
        output_dir: Directory to save data to
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(baseline_probs, f"{output_dir}/baseline_probs.joblib")
    joblib.dump(hidden_reps, f"{output_dir}/hidden_reps.joblib")
    joblib.dump(true_labels, f"{output_dir}/true_labels.joblib")

    print(f"Data saved to {output_dir}")


def load_data(input_dir):
    """
    Load prediction data from disk

    Args:
        input_dir: Directory to load data from

    Returns:
        Tuple of (baseline_probs, hidden_reps, true_labels)
    """
    baseline_probs = np.load(f"{input_dir}/baseline_probs.npy")
    hidden_reps = np.load(f"{input_dir}/hidden_reps.npy")
    true_labels = np.load(f"{input_dir}/true_labels.npy")

    return baseline_probs, hidden_reps, true_labels


def get_best_model(models_dir="./best_model", priority="fp"):
    """
    Load the best model based on specified priority

    Args:
        models_dir: Directory containing saved models
        priority: Model selection priority ('accuracy', 'fp', or 'weighted')

    Returns:
        Loaded TD3 model
    """
    if priority == "accuracy":
        model_path = f"{models_dir}/best_accuracy_model.zip"
    elif priority == "fp":
        model_path = f"{models_dir}/best_fp_model.zip"
    elif priority == "weighted":
        model_path = f"{models_dir}/best_weighted_model.zip"
    else:
        raise ValueError(
            "Priority must be one of 'accuracy', 'fp', or 'weighted'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    return TD3.load(model_path)


def apply_corrections(model, baseline_probs, hidden_reps, max_adjustment=0.6):
    """
    Apply RL model corrections to a set of probabilities

    Args:
        model: Trained TD3 model
        baseline_probs: Original prediction probabilities
        hidden_reps: Hidden representations
        max_adjustment: Maximum probability adjustment allowed

    Returns:
        Corrected probabilities
    """
    n_samples = len(baseline_probs)
    corrected_probs = np.copy(baseline_probs)

    # Calculate class distribution (approximation)
    positive_ratio = np.mean(baseline_probs >= 0.5)
    negative_ratio = 1 - positive_ratio

    # Apply corrections without true labels (inference mode)
    for i in range(n_samples):
        prob = baseline_probs[i]
        feats = hidden_reps[i]

        # Use prediction instead of true label during inference
        pred_label = 1 if prob >= 0.5 else 0

        # Calculate distance from threshold
        dist_from_threshold = abs(prob - 0.5)

        # Create observation
        obs = np.concatenate(([prob, dist_from_threshold, float(pred_label),
                             positive_ratio, negative_ratio], feats)).astype(np.float32)

        # Get model's action
        action, _ = model.predict(obs, deterministic=True)
        action_value = action[0]

        # Apply non-linear scaling
        confidence = dist_from_threshold

        # Potential false positive handling
        if pred_label == 1 and confidence < 0.3:  # Low confidence positive prediction
            if action_value < 0:  # Model wants to reduce probability
                scaled_action = action_value * 1.2  # Boost downward adjustments
                offset = np.clip(scaled_action, -max_adjustment * 1.3, 0)
            else:
                # Be conservative with upward adjustments for positives
                offset = np.clip(action_value, 0, max_adjustment * 0.3)
        else:
            # Standard scaling based on confidence
            if confidence > 0.3:  # High confidence
                scaled_action = action_value * (1 + confidence)
                offset = np.clip(
                    scaled_action, -max_adjustment, max_adjustment)
            else:
                # More conservative for low confidence
                offset = np.clip(action_value, -0.3, 0.3)

        # Apply correction
        corrected_probs[i] = np.clip(prob + offset, 0, 1)

    return corrected_probs


def log_model_to_registry(model_path, model_name, run_id=None):
    """
    Log a model to MLflow model registry

    Args:
        model_path: Path to the saved model
        model_name: Name for the registered model
        run_id: Optional MLflow run ID to associate with the model

    Returns:
        MLflow model version
    """
    if run_id:
        with mlflow.start_run(run_id=run_id):
            model_uri = mlflow.log_artifact(model_path)
    else:
        with mlflow.start_run():
            model_uri = mlflow.log_artifact(model_path)

    result = mlflow.register_model(model_uri, model_name)
    print(f"Model registered: {model_name} (version {result.version})")
    return result.version
