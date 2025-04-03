# RL-Finetune for Classification

This package uses reinforcement learning to fine-tune classification probabilities, with a special focus on reducing false positives while maintaining overall accuracy.

## Overview

The system works by training a reinforcement learning agent (TD3) to adjust the output probabilities of an existing classification model. The agent learns to:

- Correct false positives and false negatives
- Apply larger corrections for high-confidence mistakes
- Prioritize false positive reduction when configured
- Make minimal adjustments to already correct predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-finetune.git
cd rl-finetune

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- gymnasium
- stable-baselines3
- scikit-learn
- numpy
- matplotlib
- mlflow
- joblib

## Usage

### Training

Train a new RL correction model:

```bash
python main.py --mode train --timesteps 200000 --experiment_name "rl_correction" --run_name "initial_training" --fp_weight 1.2 --save_data
```

### Evaluation

Evaluate a trained model:

```bash
python main.py \
    --mode train \
    --baseline_run_id a64b253029b54b8d9405875c85d95870 \
    --timesteps 5000 \
    --max_adjustment 0.7 \
    --fp_weight 1.3 \
    --use_input_features \
    --experiment_name "RL_Correction_Experiment_input" \
    --run_name "with_input_features" \
    --register_model \
    --model_name "rl_correction_with_features"
```

### Inference

Apply a trained model to new data:

```bash
python main.py --mode infer --model_priority fp --data_dir ./data
```

## MLflow Integration

This package includes full integration with MLflow for experiment tracking:

- Metrics, parameters, and artifacts are logged automatically
- Models can be registered in the MLflow Model Registry
- Visualizations are saved and logged

To view the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

If the port is in use:

```bash
lsof -ti :5000 | xargs kill -9
```

Then navigate to http://localhost:5000 in your browser.

## Key Parameters

- `--max_adjustment`: Maximum probability adjustment allowed (default 0.6)
- `--fp_weight`: Weight for false positives in score calculation (default 1.2)
- `--model_priority`: Which model to use during evaluation/inference:
  - `accuracy`: Best overall accuracy
  - `fp`: Best false positive reduction
  - `weighted`: Best weighted score balancing accuracy and FP reduction

## Project Structure

- `env.py`: The reinforcement learning environment
- `callbacks.py`: Callbacks for model training and evaluation
- `train.py`: Training functionality
- `evaluate.py`: Evaluation and visualization functions
- `utils.py`: Utility functions
- `main.py`: Main executable script

## Example Workflow

1. Train a model:

   ```bash
   python main.py --mode train --timesteps 200000 --experiment_name "rl_correction" --run_name "run1" --save_data
   ```

2. Evaluate the results:

   ```bash
   python main.py --mode evaluate --model_priority fp
   ```

3. Deploy for inference:

   ```bash
   python main.py --mode infer --model_priority fp --data_dir ./production_data
   ```

4. Compare multiple models in MLflow UI:
   ```bash
   mlflow ui
   ```
