"""
Integration script to run both the baseline model and RL fine-tuning.
"""
import argparse
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from train import train_baseline_model
from test import evaluate_model
from model import BaselineModel
from utils import preprocess_data


def test_baseline(run_id, test_path, model_dir=None, experiment_name="Network_Intrusion_Detection", batch_size=256, device=None):
    """
    Test an already trained baseline model against a test dataset.

    Args:
        run_id: MLflow run ID of the trained baseline model to evaluate
        test_path: Path to testing data CSV
        model_dir: Directory to save model files (optional)
        experiment_name: MLflow experiment name
        batch_size: Batch size for testing
        device: Computing device ('cuda' or 'cpu'). If None, will use CUDA if available.

    Returns:
        Dictionary of evaluation metrics
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up MLflow tracking and experiment
    mlflow.set_experiment(experiment_name)

    print("\n===== Testing Baseline Model =====")
    print(f"Loading test data from {test_path}...")

    # Load and preprocess test data

    # Log parameters
    with mlflow.start_run(run_name="baseline_model_test"):
        test_data = pd.read_csv(test_path)

        # Assuming preprocess_data is defined elsewhere
        X_test, y_test = preprocess_data(train_df=test_data, run_id=run_id)
        mlflow.log_param("test_dataset_path", test_path)
        mlflow.log_param("batch_size", batch_size)

        # Call evaluate_model with the appropriate parameters
        results = evaluate_model(
            run_id=run_id,
            X_test=X_test,
            y_test=y_test,
            device=device,
            batch_size=batch_size
        )

        print("Testing completed")
        return results


def train_baseline(train_path, test_path, experiment_name="Network_Intrusion_Detection", baseline_epochs=5):
    """
    Run the full pipeline: baseline model training and RL fine-tuning.

    Args:
        train_path: Path to training data CSV
        test_path: Path to testing data CSV
        experiment_name: MLflow experiment name
        baseline_epochs: Number of epochs for baseline model
        rl_timesteps: Number of timesteps for RL training
        fp_weight: Weight for false positives in RL scoring
        baseline_batch_size: Batch size for baseline training
        model_dir: Directory to save model files
    """
    train = pd.read_csv(test_path)
    test = pd.read_csv(train_path)

    print("Preprocessing data...")

    # Set up MLflow tracking URI and experiment
    mlflow.set_experiment(experiment_name)

    print("\n===== Step 1: Training Baseline Model =====")
    with mlflow.start_run(run_name="baseline_model_train") as baseline_run:
        X_train, y_train, X_val, y_val = preprocess_data(
            train, test)
        model = BaselineModel(input_dim=X_train.shape[1])

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        mlflow.log_param("train_dataset_path", train_path)
        mlflow.log_param("test_dataset_path", test_path)
        model = train_baseline_model(
            X_train,
            y_train,
            X_val,
            y_val,
            model,
            device,
            optimizer,
            criterion,
            baseline_epochs,
            batch_size=256
        )
        input_schema = Schema(
            [TensorSpec(np.dtype(np.float32), (-1, X_train.shape[1]))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Save the model with its signature
        mlflow.pytorch.log_model(
            model,
            "baseline_model",
            signature=signature,
            code_paths=["baseline/model.py"],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Machine Learning Pipeline for Network Intrusion Detection")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Choose the command to run")

    # Common arguments for all commands
    parser.add_argument("--experiment", default="Network_Intrusion_Detection",
                        help="MLflow experiment name")

    # Train command and its specific arguments
    train_parser = subparsers.add_parser(
        "train", help="Train a new baseline model")
    train_parser.add_argument("--train", default='./data/UNSW_NB15_training-set.csv',
                              help="Path to training data CSV")
    train_parser.add_argument("--test", default='./data/UNSW_NB15_testing-set.csv',
                              help="Path to validation data CSV")
    train_parser.add_argument("--baseline_epochs", type=int,
                              default=5, help="Epochs for baseline model")

    # Test command and its specific arguments
    test_parser = subparsers.add_parser(
        "test", help="Test an existing baseline model")
    test_parser.add_argument("--run_id", required=True,
                             help="MLflow run ID of the model to evaluate")
    test_parser.add_argument("--test", default='./data/UNSW_NB15_training-set.csv',
                             help="Path to testing data CSV")
    test_parser.add_argument("--batch_size", type=int, default=256,
                             help="Batch size for testing")
    test_parser.add_argument("--device", choices=['cuda', 'cpu'], default=None,
                             help="Computing device ('cuda' or 'cpu'). Default: use CUDA if available")

    # Parse arguments
    args = parser.parse_args()

    mlflow.set_tracking_uri('sqlite:///./mlflow.db')

    # Execute the selected command
    if args.command == "train":
        print(f"Training new baseline model using data from {args.train}")
        train_baseline(
            args.train,
            args.test,
            experiment_name=args.experiment,
            baseline_epochs=args.baseline_epochs,
        )
    elif args.command == "test":
        print(
            f"Testing baseline model (run_id: {args.run_id}) using data from {args.test}")
        test_results = test_baseline(
            run_id=args.run_id,
            test_path=args.test,
            experiment_name=args.experiment,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        parser.print_help()
        print("\nPlease specify a command: train or test")
