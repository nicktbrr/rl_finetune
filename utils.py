import os
import numpy as np
import joblib
import mlflow
from stable_baselines3 import TD3

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


def preprocess_data(train_df, test_df):
    """
    Preprocess the training and testing data.
    """
    # Remove unneeded columns
    list_drop = ['id', 'attack_cat'] if 'id' in train_df.columns else []
    train_df = train_df.drop(list_drop, axis=1)
    test_df = test_df.drop(list_drop, axis=1)

    # Apply data preprocessing steps
    train_df, test_df = clip_outliers(train_df, test_df)
    train_df, test_df = log_transform(train_df, test_df)
    train_df, test_df = encode_categorical(train_df, test_df)

    # Split features and target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Identify categorical columns for one-hot encoding
    cat_cols = [i for i, col in enumerate(X_train.columns)
                if X_train[col].dtype == 'object']

    # Apply one-hot encoding and scaling
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(
            handle_unknown='ignore'), cat_cols)],
        remainder='passthrough')

    X_train = np.array(ct.fit_transform(X_train))
    X_test = np.array(ct.transform(X_test))

    # Scale numerical features
    start_col = X_train.shape[1] - (X_train.shape[1] - len(cat_cols))
    sc = StandardScaler()
    X_train[:, start_col:] = sc.fit_transform(X_train[:, start_col:])
    X_test[:, start_col:] = sc.transform(X_test[:, start_col:])

    return X_train, y_train, X_test, y_test


def clip_outliers(train_df, test_df):
    """
    Clip extreme values to prevent outliers from affecting model training.
    """
    train = train_df.copy()
    test = test_df.copy()

    df_numeric = train.select_dtypes(include=[np.number])

    clip_values = {}
    for feature in df_numeric.columns:
        max_val = df_numeric[feature].max()
        median_val = df_numeric[feature].median()
        clip_threshold = df_numeric[feature].quantile(0.95)

        if max_val > 10 * median_val and max_val > 10:
            clip_values[feature] = clip_threshold
            train[feature] = np.where(
                train[feature] < clip_threshold, train[feature], clip_threshold)

    for feature, threshold in clip_values.items():
        test[feature] = np.where(
            test[feature] < threshold, test[feature], threshold)

    return train, test


def log_transform(train_df, test_df):
    """
    Apply log transformation to numerical features with many unique values.
    """
    train = train_df.copy()
    test = test_df.copy()

    df_numeric = train.select_dtypes(include=[np.number])

    transform_features = []
    for feature in df_numeric.columns:
        if df_numeric[feature].nunique() > 50:
            transform_features.append(feature)
            train[feature] = np.log(
                train[feature] + 1) if df_numeric[feature].min() == 0 else np.log(train[feature])

    for feature in transform_features:
        test[feature] = np.log(
            test[feature] + 1) if df_numeric[feature].min() == 0 else np.log(test[feature])

    return train, test


def encode_categorical(train_df, test_df):
    """
    Encode categorical features with many categories by keeping top categories.
    """
    train = train_df.copy()
    test = test_df.copy()

    df_cat = train.select_dtypes(exclude=[np.number])

    encode_features = {}
    for feature in df_cat.columns:
        if df_cat[feature].nunique() > 6:
            top_categories = train[feature].value_counts(
            ).head().index.tolist()
            encode_features[feature] = top_categories
            train[feature] = np.where(train[feature].isin(
                top_categories), train[feature], '-')

    for feature, top_categories in encode_features.items():
        test[feature] = np.where(test[feature].isin(
            top_categories), test[feature], '-')

    return train, test


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


def save_rl_model_with_mlflow(model, model_name, artifacts=None, params=None, metrics=None):
    """
    Save a reinforcement learning model with MLflow

    Args:
        model: The trained RL model (e.g., TD3, PPO)
        model_name: Name for the model in the registry
        artifacts: Dictionary of additional artifacts to log
        params: Dictionary of parameters to log
        metrics: Dictionary of metrics to log
    """
    # Start MLflow run if not already in one
    with mlflow.start_run(run_name=f"{model_name}_save") as run:
        run_id = run.info.run_id

        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Save model to a temporary file
        temp_model_path = f"temp_{model_name}.zip"
        model.save(temp_model_path)

        # Log the model file as an artifact
        mlflow.log_artifact(temp_model_path, "model")

        # Log additional artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, name)

        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        # Register the model in the MLflow registry
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)

        print(f"Model saved with run_id: {run_id}")
        print(
            f"Model registered as: {model_name}, version: {registered_model.version}")

        return run_id, registered_model
