import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
from torchinfo import summary
import os
import mlflow
from tqdm import tqdm


class BaselineModel(nn.Module):
    def __init__(self, input_dim):
        super(BaselineModel, self).__init__()

        # Feature extractor (all layers up to the 64-dimensional layer)
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Final classifier (from 64 -> 1 -> sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract intermediate features
        hidden_reps = self.features(x)
        # Then pass those features to the classifier to get the probability
        out = self.classifier(hidden_reps)
        return out, hidden_reps


# class BaselineModel:
#     def __init__(self, input_dim, model_path='models/baseline', log_to_mlflow=True):
#         """
#         Initialize the baseline model for network intrusion detection.

#         Args:
#             input_dim (int): Number of input features
#             model_path (str): Path to save/load the model
#             log_to_mlflow (bool): Whether to log model to MLflow
#         """
#         self.input_dim = input_dim
#         self.model_path = model_path
#         self.log_to_mlflow = log_to_mlflow
#         self.device = torch.device(
#             'cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = BaselineNet(input_dim).to(self.device)
#         self.criterion = nn.BCELoss()
#         self.optimizer = optim.Adam(self.model.parameters())

#         if self.log_to_mlflow:
#             mlflow.log_param("model_type", "BaselineNet")
#             mlflow.log_param("input_dimension", input_dim)
#             mlflow.log_param("device", str(self.device))

#     def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
#         """
#         Train the model with early stopping.

#         Args:
#             X_train (np.ndarray): Training features
#             y_train (np.ndarray): Training labels
#             X_val (np.ndarray): Validation features
#             y_val (np.ndarray): Validation labels
#             epochs (int): Number of training epochs
#             batch_size (int): Batch size for training
#             class_weights (dict): Class weights for imbalanced data
#         """
#         if self.log_to_mlflow:
#             mlflow.log_params({
#                 "epochs": epochs,
#                 "batch_size": batch_size,
#                 "train_samples": len(X_train),
#                 "val_samples": len(X_val),
#                 "positive_train_ratio": np.mean(y_train),
#                 "positive_val_ratio": np.mean(y_val)
#             })

#         # Convert data to PyTorch tensors
#         X_train = torch.FloatTensor(X_train).to(self.device)
#         y_train = torch.FloatTensor(y_train).to(self.device)
#         X_val = torch.FloatTensor(X_val).to(self.device)
#         y_val = torch.FloatTensor(y_val).to(self.device)

#         # Create data loaders
#         train_dataset = TensorDataset(X_train, y_train)
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True)

#         # Initialize early stopping variables
#         best_val_loss = float('inf')
#         patience = 5
#         patience_counter = 0
#         best_model_state = None

#         # Training history
#         history = {
#             'train_loss': [],
#             'val_loss': [],
#             'train_acc': [],
#             'val_acc': []
#         }

#         # Training loop
#         for epoch in range(epochs):
#             # Training phase
#             self.model.train()
#             train_loss = 0
#             correct_train = 0
#             total_train = 0

#             for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
#                 self.optimizer.zero_grad()
#                 outputs, _ = self.model(batch_X)
#                 outputs = outputs.squeeze()

#                 loss = self.criterion(outputs, batch_y)

#                 loss.backward()
#                 self.optimizer.step()

#                 train_loss += loss.item()
#                 predicted = (outputs >= 0.5).float()
#                 total_train += batch_y.size(0)
#                 correct_train += (predicted == batch_y).sum().item()

#             # Validation phase
#             self.model.eval()
#             val_loss = 0
#             correct_val = 0
#             total_val = 0

#             with torch.no_grad():
#                 val_outputs, _ = self.model(X_val)
#                 val_outputs = val_outputs.squeeze()
#                 val_loss = self.criterion(val_outputs, y_val).item()
#                 predicted = (val_outputs >= 0.5).float()
#                 correct_val = (predicted == y_val).sum().item()
#                 total_val = y_val.size(0)

#             # Calculate metrics
#             train_loss = train_loss / len(train_loader)
#             train_acc = correct_train / total_train
#             val_acc = correct_val / total_val

#             # Update history
#             history['train_loss'].append(train_loss)
#             history['val_loss'].append(val_loss)
#             history['train_acc'].append(train_acc)
#             history['val_acc'].append(val_acc)

#             print(f'Epoch {epoch + 1}/{epochs}:')
#             print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
#             print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

#             # Log metrics to MLflow
#             if self.log_to_mlflow:
#                 mlflow.log_metrics({
#                     "train_loss": train_loss,
#                     "val_loss": val_loss,
#                     "train_accuracy": train_acc,
#                     "val_accuracy": val_acc
#                 }, step=epoch)

#             # Early stopping
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_model_state = self.model.state_dict().copy()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print(f'Early stopping triggered after {epoch + 1} epochs')
#                     if self.log_to_mlflow:
#                         mlflow.log_param("early_stopping",
#                                          f"Triggered at epoch {epoch+1}")
#                     break

#         # Restore best model
#         if best_model_state is not None:
#             self.model.load_state_dict(best_model_state)

#         # Save training history
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#         history_path = os.path.join(os.path.dirname(
#             self.model_path), 'training_history.json')
#         with open(history_path, 'w') as f:
#             json.dump(history, f)

#         if self.log_to_mlflow:
#             mlflow.log_artifact(history_path)

#         return history

#     def get_hidden_reps(self, X):
#         """
#         Return the hidden representations from the penultimate layer.
#         """
#         self.model.eval()  # set to eval mode

#         X_tensor = torch.FloatTensor(X).to(self.device)
#         with torch.no_grad():
#             # Notice we now get two outputs from forward: (prob, hidden_reps)
#             _, hidden_reps = self.model(X_tensor)
#         return hidden_reps.cpu().numpy()

#     def predict(self, X, threshold=0.5):
#         """
#         Make predictions on new data. Returns (probabilities, binary_predictions).
#         """
#         self.model.eval()
#         X_tensor = torch.FloatTensor(X).to(self.device)

#         with torch.no_grad():
#             # We only need the first part for predictions
#             probs, _ = self.model(X_tensor)
#             probs = probs.squeeze().cpu().numpy()

#         preds = (probs >= threshold).astype(int)
#         return probs, preds

#     def evaluate(self, X_test, y_test):
#         """
#         Evaluate the model performance.

#         Args:
#             X_test (np.ndarray): Test features
#             y_test (np.ndarray): Test labels

#         Returns:
#             dict: Evaluation metrics
#         """
#         probs, preds = self.predict(X_test)

#         # Calculate metrics
#         accuracy = (preds == y_test).mean()
#         auc = roc_auc_score(y_test, probs)
#         report = classification_report(y_test, preds, output_dict=True)
#         conf_matrix = confusion_matrix(y_test, preds)

#         # Calculate more specific metrics
#         tn, fp, fn, tp = conf_matrix.ravel()
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1 = 2 * precision * recall / \
#             (precision + recall) if (precision + recall) > 0 else 0

#         # Calculate specific metrics for intrusion detection
#         fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#         fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

#         results = {
#             'metrics': {
#                 'accuracy': float(accuracy),
#                 'auc': float(auc),
#                 'precision': float(precision),
#                 'recall': float(recall),
#                 'f1_score': float(f1),
#                 'false_positive_rate': float(fpr),
#                 'false_negative_rate': float(fnr),
#                 'true_positives': int(tp),
#                 'true_negatives': int(tn),
#                 'false_positives': int(fp),
#                 'false_negatives': int(fn)
#             },
#             'classification_report': report,
#             'confusion_matrix': conf_matrix.tolist()
#         }

#         # Save evaluation results
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#         results_path = os.path.join(os.path.dirname(
#             self.model_path), 'evaluation_results.json')
#         with open(results_path, 'w') as f:
#             json.dump(results, f)

#         # Log to MLflow
#         if self.log_to_mlflow:
#             mlflow.log_metrics(results['metrics'])
#             mlflow.log_artifact(results_path)

#         return results

#     def save(self):
#         """Save the model to disk."""
#         os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
#         model_file = f"{self.model_path}.pt"
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'input_dim': self.input_dim
#         }, model_file)

#         if self.log_to_mlflow:
#             with open("model_summary.txt", "w") as f:
#                 f.write(str(summary(self.model)))
#             mlflow.log_artifact("model_summary.txt")
#             mlflow.log_artifact(model_file)
#             # Also log model to MLflow model registry for easier loading
#             mlflow.pytorch.log_model(self.model, "pytorch-model")

#     def load(self):
#         """Load the model from disk."""
#         model_file = f"{self.model_path}.pt"
#         if os.path.exists(model_file):
#             checkpoint = torch.load(model_file, map_location=self.device)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             self.model.eval()
#         else:
#             raise FileNotFoundError(f"No model found at {model_file}")
