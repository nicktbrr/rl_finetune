import os
import argparse
import numpy as np
import mlflow

from train import train_with_accuracy_checkpoints
from evaluate import evaluate_rl_corrections
from utils import save_data, load_data, get_best_model, apply_corrections, log_model_to_registry


def main():

    parser = argparse.ArgumentParser(
        description='RL Fine-tuning for Classification')

    # Main operation modes
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'infer'],
                        help='Operation mode: train, evaluate, or infer')
    parser.add_argument('--baseline_data', type=str, required=True,
                        help='Directory for saving/loading baseline model predictions and hidden representations files')

    # Training arguments
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Number of training timesteps')
    parser.add_argument('--max_adjustment', type=float, default=0.6,
                        help='Maximum probability adjustment allowed')
    parser.add_argument('--fp_weight', type=float, default=1.2,
                        help='Weight for false positives in score calculation')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for TD3 agent')

    # MLflow arguments
    parser.add_argument('--experiment_name', type=str, default='RL_Correction',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name')
    parser.add_argument('--register_model', action='store_true',
                        help='Register best model in MLflow model registry')
    parser.add_argument('--model_name', type=str, default='rl_correction_model',
                        help='Name for registered model')

    # Evaluation/Inference arguments
    parser.add_argument('--model_priority', type=str, default='fp',
                        choices=['accuracy', 'fp', 'weighted'],
                        help='Model selection priority: accuracy, fp, or weighted')

    args = parser.parse_args()

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment(args.experiment_name)

    # Different operational modes
    if args.mode == 'train':
        print("Loading base model predictions and representations...")
        # In a real application, you would get these from your base model predictions
        # This is just a placeholder - replace with actual data loading logic
        try:
            # Try to load from data_dir if available
            baseline_probs, hidden_reps, true_labels = load_data(args.data_dir)
            print(f"Loaded data from {args.data_dir}")
        except FileNotFoundError:
            # If files don't exist, this would typically come from your model
            print("Data files not found. In a real application, you would use:")
            print("baseline_probs, preds = model.predict(X_val)")
            print("hidden_reps = model.get_hidden_reps(X_val)")
            print("true_labels = y_val")
            print("For demonstration, using randomly generated data instead...")

            # Generate some sample data for demonstration
            n_samples = 1000
            feature_dim = 64

            baseline_probs = np.random.uniform(0, 1, size=n_samples)
            hidden_reps = np.random.normal(0, 1, size=(n_samples, feature_dim))
            true_labels = (np.random.random(n_samples) >
                           0.7).astype(int)  # 30% positive rate

        # Save data if requested
        if args.save_data:
            save_data(baseline_probs, hidden_reps, true_labels, args.data_dir)

        # Train the RL model
        print(f"Starting training with {args.timesteps} timesteps...")
        trained_model = train_with_accuracy_checkpoints(
            baseline_probs=baseline_probs,
            hidden_reps=hidden_reps,
            true_labels=true_labels,
            max_adjustment=args.max_adjustment,
            timesteps=args.timesteps,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            fp_weight=args.fp_weight,
            learning_rate=args.learning_rate
        )

        # Register model if requested
        if args.register_model:
            print(f"Registering models with MLflow...")
            # Register all three model variants
            log_model_to_registry(
                "./best_model/best_accuracy_model.zip",
                f"{args.model_name}_accuracy"
            )
            log_model_to_registry(
                "./best_model/best_fp_model.zip",
                f"{args.model_name}_fp"
            )
            log_model_to_registry(
                "./best_model/best_weighted_model.zip",
                f"{args.model_name}_weighted"
            )

    elif args.mode == 'evaluate':
        print("Loading data for evaluation...")
        try:
            baseline_probs, hidden_reps, true_labels = load_data(args.data_dir)
        except FileNotFoundError:
            print(f"Error: Data files not found in {args.data_dir}")
            return

        print(f"Loading best model with priority: {args.model_priority}")
        try:
            model = get_best_model(priority=args.model_priority)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading model: {e}")
            return

        # Run evaluation
        with mlflow.start_run(run_name=f"evaluation_{args.model_priority}"):
            mlflow.log_param("model_priority", args.model_priority)
            mlflow.log_param("max_adjustment", args.max_adjustment)
            mlflow.log_param("data_size", len(baseline_probs))

            print("Evaluating RL model corrections...")
            results = evaluate_rl_corrections(
                model,
                baseline_probs,
                hidden_reps,
                true_labels,
                max_adjustment=args.max_adjustment,
                log_to_mlflow=True
            )

    elif args.mode == 'infer':
        print("Loading data for inference...")
        try:
            baseline_probs, hidden_reps, _ = load_data(args.data_dir)
            # In inference mode, we don't need true labels
        except FileNotFoundError:
            print(f"Error: Data files not found in {args.data_dir}")
            return

        print(f"Loading best model with priority: {args.model_priority}")
        try:
            model = get_best_model(priority=args.model_priority)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading model: {e}")
            return

        # Apply corrections
        print("Applying RL model corrections...")
        corrected_probs = apply_corrections(
            model,
            baseline_probs,
            hidden_reps,
            max_adjustment=args.max_adjustment
        )

        # Calculate changes
        original_preds = (baseline_probs >= 0.5).astype(int)
        corrected_preds = (corrected_probs >= 0.5).astype(int)
        changed = np.sum(original_preds != corrected_preds)

        print(f"Applied corrections to {len(baseline_probs)} predictions")
        print(
            f"Changed {changed} predictions ({changed/len(baseline_probs)*100:.2f}%)")

        # Save the corrected probabilities
        output_file = os.path.join(args.data_dir, "corrected_probs.npy")
        np.save(output_file, corrected_probs)
        print(f"Saved corrected probabilities to {output_file}")


if __name__ == "__main__":
    main()
