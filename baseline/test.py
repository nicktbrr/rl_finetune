"""
Test and evaluate the trained baseline model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

from model import BaselineModel
from train import preprocess_data


def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve and save to file."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_path = "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    
    return roc_path, roc_auc


def plot_precision_recall_curve(y_true, y_prob):
    """Plot precision-recall curve and save to file."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    
    pr_path = "precision_recall_curve.png"
    plt.savefig(pr_path)
    plt.close()
    
    return pr_path


def plot_prob_distribution(y_true, y_prob):
    """Plot probability distribution for both classes."""
    plt.figure(figsize=(10, 6))
    
    # Plot distributions
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Normal (Class 0)')
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Attack (Class 1)')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    dist_path = "probability_distribution.png"
    plt.savefig(dist_path)
    plt.close()
    
    return dist_path


def evaluate_model(model_path, test_path, threshold=0.5):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the saved model
        test_path: Path to test data CSV
        threshold: Classification threshold
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Set up MLflow
    mlflow.set_experiment("Network_Intrusion_Detection")
    
    with mlflow.start_run(run_name="baseline_evaluation"):
        # Load the model
        print(f"Loading model from {model_path}")
        model = BaselineModel(input_dim=1, model_path=model_path)  # Input dim will be overridden by loaded model
        try:
            model.load()
        except FileNotFoundError:
            print(f"Model not found at {model_path}. Please train the model first.")
            return None
        
        # Load and preprocess test data
        print(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        
        # Preprocess test data (this requires train data for consistent preprocessing)
        if os.path.exists(test_path.replace('test', 'train')):
            train_path = test_path.replace('test', 'train')
            train_df = pd.read_csv(train_path)
            _, _, X_test, y_test = preprocess_data(train_df, test_df)
        else:
            print("Warning: Training data not found for consistent preprocessing.")
            print("Using simplified preprocessing which may affect results.")
            X_test = test_df.iloc[:, :-1].values  # Assuming last column is the target
            y_test = test_df.iloc[:, -1].values
        
        # Log parameters
        mlflow.log_params({
            "model_path": model_path,
            "test_data": test_path,
            "threshold": threshold,
            "test_samples": X_test.shape[0],
            "positive_test_ratio": float(np.mean(y_test))
        })
        
        # Make predictions
        print("Making predictions...")
        probs, preds = model.predict(X_test, threshold=threshold)
        
        # Generate evaluation metrics
        print("Computing metrics...")
        accuracy = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds, average='weighted')
        precision = precision_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # Compute intrusion detection specific metrics
        fpr = fp / (fp + tn)  # False positive rate
        fnr = fn / (fn + tp)  # False negative rate (miss rate)
        detection_rate = tp / (tp + fn)  # True positive rate (sensitivity)
        
        # Plot and save visualizations
        print("Generating visualizations...")
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        cm_disp.plot(ax=ax)
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # Generate ROC curve
        roc_path, roc_auc = plot_roc_curve(y_test, probs)
        
        # Generate precision-recall curve
        pr_path = plot_precision_recall_curve(y_test, probs)
        
        # Generate probability distribution plot
        dist_path = plot_prob_distribution(y_test, probs)
        
        # Log all metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "detection_rate": detection_rate
        }
        
        mlflow.log_metrics(metrics)
        
        # Log plots
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(dist_path)
        
        # Generate baseline predictions for RL fine-tuning
        print("Generating data for RL fine-tuning...")
        output_dir = os.path.dirname(model_path)
        
        # Extract hidden representations
        hidden_reps = model.get_hidden_reps(X_test)
        
        # Save data for RL
        np.save(os.path.join(output_dir, "baseline_probs.npy"), probs)
        np.save(os.path.join(output_dir, "hidden_reps.npy"), hidden_reps)
        np.save(os.path.join(output_dir, "true_labels.npy"), y_test)
        
        # Log data artifacts
        mlflow.log_artifact(os.path.join(output_dir, "baseline_probs.npy"), "rl_data")
        mlflow.log_artifact(os.path.join(output_dir, "hidden_reps.npy"), "rl_data")
        mlflow.log_artifact(os.path.join(output_dir, "true_labels.npy"), "rl_data")
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"Detection Rate: {detection_rate:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline network intrusion detection model")
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--test", required=True, help="Path to test data CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test, args.threshold)


if __name__ == "__main__":
    main()
