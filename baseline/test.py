import os
import torch
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(run_id, X_test, y_test=None, device='cuda', batch_size=256):
    """
    Evaluate a trained model on test data or run inference.

    Args:
        run_id: MLflow run ID of the model to evaluate
        X_test: Test feature data (numpy array or torch tensor)
        y_test: Test labels (numpy array or torch tensor), optional for inference mode
        device: Computing device ('cuda' or 'cpu')
        batch_size: Batch size for evaluation
        threshold: Classification threshold for binary prediction

    Returns:
        Dictionary of evaluation metrics if y_test is provided, 
        or predictions if y_test is None (inference mode)
    """
    # Determine if we're in evaluation or inference mode
    evaluation_mode = y_test is not None
    # Load the model
    print(f"Loading model {run_id}")
    logged_model = f'runs:/{run_id}/baseline_model'
    try:
        model = mlflow.pytorch.load_model(logged_model)
        model = model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Convert inputs to torch tensors if they're not already
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    X_test = X_test.to(device)

    # Create proper data loader
    if evaluation_mode:
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.FloatTensor(y_test)
        y_test = y_test.to(device)

        # Log parameters
        mlflow.log_params({
            "model_id": run_id,
            "test_samples": X_test.shape[0],
            "num_pos_class": int(y_test.sum().item()),
            "num_neg_class": int(y_test.shape[0] - y_test.sum().item()),
            "positive_test_ratio": float(torch.mean(y_test).item()),
        })

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # For inference, we don't have labels
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_true_labels = [] if evaluation_mode else None

    with torch.no_grad():
        if evaluation_mode:
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)

                # Handle different model output formats
                if isinstance(outputs, tuple):
                    # Assuming the first element is the main output
                    outputs = outputs[0]

                probabilities = outputs.squeeze()
                predictions = (probabilities >= 0.5).float()

                all_probabilities.extend(probabilities.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(batch_y.cpu().numpy())
        else:
            for (batch_X,) in test_loader:
                outputs = model(batch_X)

                # Handle different model output formats
                if isinstance(outputs, tuple):
                    # Assuming the first element is the main output
                    outputs = outputs[0]

                probabilities = outputs.squeeze()
                predictions = (probabilities >= 0.5).float()

                all_probabilities.extend(probabilities.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # If in evaluation mode, calculate and log metrics
    if evaluation_mode:
        all_true_labels = np.array(all_true_labels)

        # Calculate metrics
        accuracy = accuracy_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions)

        # Log metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)

        # Generate confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        cm_filename = 'confusion_matrix.png'
        plt.savefig(cm_filename)
        plt.close()
        mlflow.log_artifact(cm_filename)
        os.remove(cm_filename)

        # Generate ROC curve
        fpr, tpr, _ = roc_curve(all_true_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_filename = 'roc_curve.png'
        plt.savefig(roc_filename)
        plt.close()
        mlflow.log_artifact(roc_filename)
        os.remove(roc_filename)

        # Generate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            all_true_labels, all_probabilities)
        pr_auc = auc(recall_curve, precision_curve)
        plt.figure(figsize=(6, 5))
        plt.plot(recall_curve, precision_curve, color='green',
                 lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        pr_filename = 'precision_recall_curve.png'
        plt.savefig(pr_filename)
        plt.close()
        mlflow.log_artifact(pr_filename)
        os.remove(pr_filename)

        print(
            f"Evaluation completed with accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
        return metrics
    else:
        # In inference mode, return predictions and probabilities
        print(f"Inference completed on {X_test.shape[0]} samples")
        return {
            "predictions": all_predictions,
            "probabilities": all_probabilities
        }
