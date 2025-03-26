# Network Intrusion Detection Baseline Model

This module contains a baseline neural network model for network intrusion detection, designed to work with the RL fine-tuning module.

## Overview

The baseline model is a simple neural network that:
- Processes network traffic data to detect intrusions
- Extracts hidden representations which are then used for RL-based fine-tuning
- Integrates with MLflow for experiment tracking

## Files

- `model.py`: Contains the PyTorch model architecture and training/evaluation functions
- `train.py`: Script for training the baseline model and preprocessing data
- `test.py`: Script for testing and evaluating the baseline model
- `integrate.py`: Script that combines baseline training and RL fine-tuning in one pipeline

## Usage

### Training

Train the baseline model:

```bash
python train.py --train /path/to/UNSW_NB15_training-set.csv --test /path/to/UNSW_NB15_testing-set.csv --epochs 10 --batch_size 250
```

### Testing

Evaluate a trained model:

```bash
python test.py --model models/baseline --test /path/to/UNSW_NB15_testing-set.csv
```

### Full Pipeline

Run the complete pipeline (baseline model + RL fine-tuning):

```bash
python integrate.py --train /path/to/UNSW_NB15_training-set.csv --test /path/to/UNSW_NB15_testing-set.csv --baseline_epochs 10 --rl_timesteps 200000
```

## MLflow Integration

This module tracks all experiments with MLflow:

- Model parameters
- Training metrics
- Evaluation results
- Visualizations (confusion matrices, ROC curves, etc.)
- Artifacts (model files, preprocessed data, etc.)

To view the MLflow UI:

```bash
mlflow ui
```

Then navigate to http://localhost:5000 in your browser.

## Preprocessing Steps

The training script includes several preprocessing steps:

1. **Outlier Clipping**: Extreme values are clipped to prevent outliers from affecting model training
2. **Log Transformation**: Log transformation is applied to numerical features with many unique values
3. **Categorical Encoding**: Categorical features with many categories are encoded to keep only the top categories
4. **One-Hot Encoding**: Categorical features are one-hot encoded
5. **Standardization**: Numerical features are standardized

## Integration with RL Fine-tuning

After training the baseline model, it generates three key files:

1. `baseline_probs.npy`: Prediction probabilities from the baseline model
2. `hidden_reps.npy`: Hidden layer representations (features)
3. `true_labels.npy`: True labels for the test data

These files are then used by the RL fine-tuning process to learn how to adjust the baseline model's predictions to reduce false positives.
