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
from model import BaselineModel
from utils import preprocess_data


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
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Preprocessing data...")
    X_train, y_train, X_val, y_val = preprocess_data(
        train, test)
    model = BaselineModel(input_dim=X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    # Set up MLflow tracking URI and experiment
    mlflow.set_tracking_uri('sqlite:///./mlflow.db')
    mlflow.set_experiment(experiment_name)

    # Step 1: Train baseline model (main approach - only trains and tests)
    print("\n===== Step 1: Training Baseline Model =====")
    with mlflow.start_run(run_name="baseline_model") as baseline_run:
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
    parser = argparse.ArgumentParser(
        description="Run full pipeline: baseline model and RL fine-tuning")
    parser.add_argument("--train", default='./data/UNSW_NB15_training-set.csv',
                        help="Path to training data CSV")
    parser.add_argument("--test", default='./data/UNSW_NB15_testing-set.csv',
                        help="Path to testing data CSV")
    parser.add_argument(
        "--experiment", default="Network_Intrusion_Detection", help="MLflow experiment name")
    parser.add_argument("--baseline_epochs", type=int,
                        default=5, help="Epochs for baseline model")

    args = parser.parse_args()

    train_baseline(
        args.train,
        args.test,
        experiment_name=args.experiment,
        baseline_epochs=args.baseline_epochs,
    )
