import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from utils import preprocess_data


def train_baseline_model(
        X_train,
        y_train,
        X_val,
        y_val,
        model,
        device,
        optimizer,
        criterion,
        epochs=10,
        batch_size=250
):

    # Log dataset characteristics
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],  # Fixed variable name X_test → X_val
        "feature_dimension": X_train.shape[1],
        "positive_train_ratio": float(np.mean(y_train)),
        "positive_test_ratio": float(np.mean(y_val))
    })

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    # Create proper tensor datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create proper data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            outputs = outputs.squeeze()

            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            # Use validation loader instead of the entire validation set at once
            for batch_X, batch_y in val_loader:
                val_outputs, _ = model(batch_X)
                val_outputs = val_outputs.squeeze()

                # Accumulate validation loss
                batch_loss = criterion(val_outputs, batch_y).item()
                val_loss += batch_loss

                predicted = (val_outputs >= 0.5).float()
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)  # Normalize by number of batches
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        }, step=epoch)
    return model
