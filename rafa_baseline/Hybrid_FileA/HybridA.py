import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import pandas as pd
from model import HybridModel
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

# Constants
BATCH_SIZE = 1024
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
RANDOM_STATE = 42
SEQUENCE_LENGTH = 32  # For transformer part
EARLY_STOPPING_PATIENCE = 15

def create_directory_structure():
    """Create the required directory structure for saving models and metrics."""
    base_dir = Path('Hybrid_FileA/model_data')
    model_dir = base_dir / 'model'
    metrics_dir = base_dir / 'metrics'
    
    for dir_path in [model_dir, metrics_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return model_dir, metrics_dir

def save_metrics(metrics_dir, metrics, epoch=None, is_final=False):
    """Save metrics to JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_final:
        filename = metrics_dir / f'final_metrics_{timestamp}.json'
    else:
        filename = metrics_dir / f'metrics_epoch_{epoch}_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

class NetworkDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        # Original features for CNN
        self.X_cnn = torch.FloatTensor(X)
        
        # Reshape features for transformer
        feature_size = X.shape[1]
        # Pad the features if needed to make it divisible by sequence_length
        if feature_size % sequence_length != 0:
            pad_size = sequence_length - (feature_size % sequence_length)
            X_padded = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
            feature_size = X_padded.shape[1]
            self.X_transformer = torch.FloatTensor(X_padded).reshape(-1, sequence_length, feature_size // sequence_length)
        else:
            self.X_transformer = torch.FloatTensor(X).reshape(-1, sequence_length, feature_size // sequence_length)
        
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_cnn[idx], self.X_transformer[idx], self.y[idx]

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1': float(f1_score(y_true, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Calculate specificity for each class
    spec_scores = []
    for class_idx in range(len(np.unique(y_true))):
        y_true_binary = (y_true == class_idx)
        y_pred_binary = (y_pred == class_idx)
        tn = np.sum((~y_true_binary) & (~y_pred_binary))
        fp = np.sum((~y_true_binary) & y_pred_binary)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_scores.append(float(specificity))
    
    metrics['specificity'] = spec_scores
    return metrics

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X_cnn, X_transformer, y in loader:
            X_cnn = X_cnn.to(device)
            X_transformer = X_transformer.to(device)
            y = y.to(device)
            
            outputs = model(X_cnn, X_transformer)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
    metrics['loss'] = avg_loss
    
    return metrics

def main():
    # Set up logging
    # logging.basicConfig(
    #     filename=f'Hybrid_FileA/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    
    # Set device
    device = torch.device("cpu")  # Use GPU 2
    # torch.cuda.set_device(device)
    # logging.info(f"Using device: {device}")
    # logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(device)}")
    
    # Create directory structure
    model_dir, metrics_dir = create_directory_structure()
    
    try:
        # Load data
        logging.info("\nLoading data...")
        df = pd.read_parquet('FileA/train_cleaned.parquet')
        
        # Prepare features and labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['full_multiclass'])
        X = df.drop(['binary_label', 'unified_label', 'full_multiclass', 
                    'aggregated_label', 'ts', 'uid'], axis=1)
        
        # Convert categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        feature_encoders = {}
        for col in categorical_cols:
            feature_encoders[col] = LabelEncoder()
            X[col] = feature_encoders[col].fit_transform(X[col])
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Calculate feature sizes for transformer
        n_features = X.shape[1]
        if n_features % SEQUENCE_LENGTH != 0:
            pad_size = SEQUENCE_LENGTH - (n_features % SEQUENCE_LENGTH)
            n_features_padded = n_features + pad_size
            transformer_feature_size = n_features_padded // SEQUENCE_LENGTH
        else:
            transformer_feature_size = n_features // SEQUENCE_LENGTH
        
        logging.info(f"Number of features: {n_features}")
        logging.info(f"Transformer feature size: {transformer_feature_size}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
        
        # Create datasets
        train_dataset = NetworkDataset(X_train, y_train, SEQUENCE_LENGTH)
        val_dataset = NetworkDataset(X_val, y_val, SEQUENCE_LENGTH)
        test_dataset = NetworkDataset(X_test, y_test, SEQUENCE_LENGTH)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize model
        n_classes = len(np.unique(y))
        
        model = HybridModel(
            cnn_input_size=n_features,
            transformer_feature_size=transformer_feature_size,
            num_classes=n_classes,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training loop
        logging.info("\nStarting training...")
        best_val_f1 = 0
        patience_counter = 0
        training_history = []
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0
            batch_count = 0
            
            logging.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            for batch_idx, (X_cnn, X_transformer, y_batch) in enumerate(train_loader):
                X_cnn = X_cnn.to(device)
                X_transformer = X_transformer.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_cnn, X_transformer)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                               f"Loss: {loss.item():.4f}, "
                               f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            scheduler.step()
            avg_train_loss = train_loss / batch_count
            
            # Evaluate
            train_metrics = evaluate_model(model, train_loader, criterion, device)
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            
            # Save metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'validation': val_metrics,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_metrics)
            save_metrics(metrics_dir, epoch_metrics, epoch=epoch+1)
            
            # Model checkpoint
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                logging.info(f"\nNew best model! F1: {best_val_f1:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'scaler': scaler,
                    'label_encoder': label_encoder,
                    'feature_encoders': feature_encoders
                }, model_dir / 'best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            logging.info(f"\nEpoch {epoch+1} Summary:")
            logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            logging.info(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            logging.info(f"Best Val F1 so far: {best_val_f1:.4f}")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info("\nEarly stopping triggered!")
                break
        
        # Final evaluation
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        final_metrics = {
            'test': test_metrics,
            'best_val_f1': best_val_f1,
            'training_history': training_history
        }
        save_metrics(metrics_dir, final_metrics, is_final=True)
        
        logging.info("\nTraining completed!")
        logging.info(f"Best validation F1: {best_val_f1:.4f}")
        logging.info(f"Final test metrics:")
        logging.info(f"  F1: {test_metrics['f1']:.4f}")
        logging.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {test_metrics['precision']:.4f}")
        logging.info(f"  Recall: {test_metrics['recall']:.4f}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 