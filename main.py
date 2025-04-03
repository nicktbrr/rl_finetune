import os
import argparse
import numpy as np
import mlflow
import torch
from baseline.utils import preprocess_data
import pandas as pd
from generate_purturbed_data import create_perturbed_data
from sklearn.model_selection import train_test_split
from train import train_with_accuracy_checkpoints
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from evaluate import td3_inference


def main():

    parser = argparse.ArgumentParser(
        description='RL Fine-tuning for Classification')

    # Main operation modes
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'infer'],
                        help='Operation mode: train, evaluate, or infer')
    parser.add_argument('--baseline_run_id', type=str, required=True,
                        help='Run id of baseline model')

    # Training arguments
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Number of training timesteps')
    parser.add_argument('--max_adjustment', type=float, default=0.6,
                        help='Maximum probability adjustment allowed')
    parser.add_argument('--fp_weight', type=float, default=1.2,
                        help='Weight for false positives in score calculation')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for TD3 agent')

    # New argument for using original input features
    parser.add_argument('--use_input_features', action='store_true',
                        help='Include original input features in the RL model training')
    parser.add_argument('--normalize_features', action='store_true',
                        help='Normalize input features before passing to RL model')

    # MLflow arguments
    parser.add_argument('--experiment_name', type=str, default='RL_Correction_input_features',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name')
    parser.add_argument('--register_model', action='store_true',
                        help='Register best model in MLflow model registry')
    parser.add_argument('--model_name', type=str, default='rl_correction_model',
                        help='Name for registered model')

    args = parser.parse_args()

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment(args.experiment_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Different operational modes
    if args.mode == 'train':
        print("Loading base model")
        # In a real application, you would get these from your base model predictions
        # This is just a placeholder - replace with actual data loading logic
        logged_model = f'runs:/{args.baseline_run_id}/baseline_model'
        try:
            model = mlflow.pytorch.load_model(logged_model)
            model = model.to(device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        try:
            print("Loading test data")
            perturbed_data = pd.read_csv('data/perturbed_train_data.csv')
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        try:
            print("Getting model Predictions")
            X = perturbed_data.to_numpy()
            y = X[:, -1]
            X = X[:, :-1]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y)
            X_train_tensor = torch.tensor(
                X_train, dtype=torch.float32).to(device)
            baseline_probs, hidden_reps = model(X_train_tensor)

            y_pred = (baseline_probs >= 0.5).float().cpu().numpy().squeeze()

            # Compute confusion matrix
            cm = confusion_matrix(y_train, y_pred)

            # Display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix Before RL Training")
        except Exception as e:
            print(f"Error running model: {e}")
            return None

        with mlflow.start_run(run_name=args.run_name):
            # Save confusion matrix figure
            cm_fig = disp.figure_
            cm_path = "before_rl_confusion_matrix.png"
            cm_fig.savefig(cm_path)
            mlflow.log_artifact(cm_path, artifact_path="before_rl")

            # Log some basic pre-training metrics
            cm_values = cm.ravel()
            if len(cm_values) == 4:
                tn, fp, fn, tp = cm_values
                mlflow.log_metric("before_rl_true_negatives", tn)
                mlflow.log_metric("before_rl_false_positives", fp)
                mlflow.log_metric("before_rl_false_negatives", fn)
                mlflow.log_metric("before_rl_true_positives", tp)
                mlflow.log_metric("before_rl_accuracy", (tp + tn) / cm.sum())
                mlflow.log_metric("before_rl_precision", tp / (tp + fp + 1e-8))
                mlflow.log_metric("before_rl_recall", tp / (tp + fn + 1e-8))

            # Prepare input features if requested
            input_features = None
            if args.use_input_features:
                print("Including original input features in RL training...")
                input_features = X_train.copy()

                # Log feature info
                mlflow.log_param("input_features_dim", input_features.shape[1])

            # Train the RL model
            print(f"Starting training with {args.timesteps} timesteps...")

            trained_model = train_with_accuracy_checkpoints(
                baseline_probs=baseline_probs.detach().cpu().numpy(),
                hidden_reps=hidden_reps.detach().cpu().numpy(),
                true_labels=y_train,
                input_features=input_features,  # This is the new parameter
                max_adjustment=args.max_adjustment,
                timesteps=args.timesteps,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                fp_weight=args.fp_weight,
                learning_rate=args.learning_rate
            )
        with mlflow.start_run(run_name=args.run_name + '_evaluate'):
            print("Loading test data...")
            input_features = None
            if args.use_input_features:
                print("Including original input features in RL training...")
                input_features = X_test.copy()

                # Log feature info
                mlflow.log_param("input_features_dim", input_features.shape[1])

            try:
                mlflow.log_param("test_samples", X_test.shape[0])
                mlflow.log_param("positive_ratio", np.mean(y_test))
            except Exception as e:
                print(f"Error loading test data: {e}")
                return None

            X_test_tensor = torch.tensor(
                X_test, dtype=torch.float32).to(device)

            baseline_probs, hidden_reps = model(X_test_tensor)

            y_pred = (baseline_probs >= 0.5).float().cpu().numpy().squeeze()

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix Before RL Training Evaluation")

            cm_fig = disp.figure_
            cm_path = "before_rl_confusion_matrix_evaluation.png"
            cm_fig.savefig(cm_path)
            mlflow.log_artifact(cm_path, artifact_path="before_rl_evaluation")

            # Log some basic pre-training metrics
            cm_values = cm.ravel()
            if len(cm_values) == 4:
                tn, fp, fn, tp = cm_values
                mlflow.log_metric("before_rl_evaluation_true_negatives", tn)
                mlflow.log_metric("before_rl_evaluation_false_positives", fp)
                mlflow.log_metric("before_rl_evaluation_false_negatives", fn)
                mlflow.log_metric("before_rl_evaluation_true_positives", tp)
                mlflow.log_metric(
                    "before_rl_evaluation_accuracy", (tp + tn) / cm.sum())
                mlflow.log_metric(
                    "before_rl_evaluation_precision", tp / (tp + fp + 1e-8))
                mlflow.log_metric(
                    "before_rl_evaluation_recall", tp / (tp + fn + 1e-8))

            print("Evaluating models...")
            corrected_probs, adjustments = td3_inference(
                trained_model=trained_model,
                baseline_probs=baseline_probs,
                hidden_reps=hidden_reps,
                input_features=input_features,  # Optional
                max_adjustment=0.6    # Same as during training
            )

            # 4. Use the corrected probabilities
            corrected_predictions = (corrected_probs >= 0.5).astype(int)

            # Compute confusion matrix
            cm = confusion_matrix(y_test, corrected_predictions)

            # Display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix After RL Training Evaluation")

            cm_fig = disp.figure_
            cm_path = "after_rl_confusion_matrix_evaluation.png"
            cm_fig.savefig(cm_path)
            mlflow.log_artifact(cm_path, artifact_path="after_rl_evaluation")

            # Log the same metrics as training for the corrected predictions
            cm_values = cm.ravel()
            if len(cm_values) == 4:
                tn, fp, fn, tp = cm_values
                mlflow.log_metric("after_rl_evaluation_true_negatives", tn)
                mlflow.log_metric("after_rl_evaluation_false_positives", fp)
                mlflow.log_metric("after_rl_evaluation_false_negatives", fn)
                mlflow.log_metric("after_rl_evaluation_true_positives", tp)
                mlflow.log_metric(
                    "after_rl_evaluation_accuracy", (tp + tn) / cm.sum())
                mlflow.log_metric(
                    "after_rl_evaluation_precision", tp / (tp + fp + 1e-8))
                mlflow.log_metric("after_rl_evaluation_recall",
                                  tp / (tp + fn + 1e-8))

            # 7. Log the link to the run
            run_id = mlflow.active_run().info.run_id
            print(f"\nEvaluation complete! MLflow run ID: {run_id}")

    elif args.mode == 'infer':
        # Inference logic would be updated to handle input features
        print("Inference mode - implement based on your specific inference needs")
        # Example of what might be added for inference with input features


if __name__ == "__main__":
    main()
