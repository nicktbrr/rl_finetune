import os
import mlflow
import numpy as np
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from env import EnhancedFineTuneEnv
from callbacks import AccuracyCheckpointCallback


def train_with_accuracy_checkpoints(baseline_probs, hidden_reps, true_labels, 
                                   max_adjustment=0.6, timesteps=200000, 
                                   experiment_name="RL_Correction", 
                                   run_name=None,
                                   fp_weight=1.2,
                                   learning_rate=3e-4):
    """
    Train a TD3 agent to correct model predictions with focus on reducing false positives.

    Args:
        baseline_probs: Original prediction probabilities from base model
        hidden_reps: Hidden representations from base model
        true_labels: Ground truth labels
        max_adjustment: Maximum probability adjustment allowed
        timesteps: Number of training timesteps
        experiment_name: MLflow experiment name
        run_name: MLflow run name (optional)
        fp_weight: Weight for false positives in score calculation
        learning_rate: Learning rate for TD3 agent

    Returns:
        Trained TD3 model
    """
    # Create directories
    os.makedirs("./best_model/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Start MLflow run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log hyperparameters to MLflow
        mlflow.log_params({
            "max_adjustment": max_adjustment,
            "timesteps": timesteps,
            "fp_weight": fp_weight,
            "learning_rate": learning_rate,
            "input_dim": hidden_reps.shape[1] + 5,  # 5 for the additional features
            "positive_ratio": np.mean(true_labels),
            "data_size": len(true_labels)
        })

        # Create environment factory function
        def make_env(max_adj=max_adjustment):
            def _init():
                return EnhancedFineTuneEnv(baseline_probs, hidden_reps, true_labels, max_adj)
            return _init

        # Create vectorized training environment
        train_env = DummyVecEnv([make_env(max_adjustment=max_adjustment)])

        # Create evaluation environment (same as training but separate instance)
        eval_env = DummyVecEnv([make_env(max_adjustment=max_adjustment)])

        # TD3 configuration optimized for this task
        td3_config = {
            'learning_rate': learning_rate,
            'buffer_size': 200000,
            'learning_starts': 5000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'policy_delay': 2,
            'target_policy_noise': 0.3,
            'target_noise_clip': 0.5,
            'policy_kwargs': dict(
                net_arch=[256, 256, 128],
                activation_fn=nn.ReLU
            )
        }

        # Log TD3 config to MLflow
        mlflow.log_params(td3_config)

        # Create the accuracy-based checkpoint callback
        accuracy_callback = AccuracyCheckpointCallback(
            eval_env=eval_env,
            baseline_probs=baseline_probs,
            true_labels=true_labels,
            check_freq=5000,
            save_path='./best_model/',
            fp_weight=fp_weight,  # Higher weight means more focus on reducing false positives
            log_to_mlflow=True,
            verbose=1
        )

        # Create and train the model
        print("Starting TD3 training with accuracy-based checkpoints...")
        model_rl = TD3("MlpPolicy", train_env, verbose=1, **td3_config)

        # Train with callbacks
        model_rl.learn(
            total_timesteps=timesteps,
            callback=accuracy_callback
        )

        # Final evaluation
        print("\n--- Final Model Evaluation ---")
        final_metrics = accuracy_callback._evaluate_model()

        print(
            f"Final Accuracy: {final_metrics['accuracy']:.4f} (Baseline: {accuracy_callback.baseline_accuracy:.4f})")
        print(
            f"Final False Positives: {final_metrics['false_positives']} (Baseline: {accuracy_callback.baseline_fp})")
        print(
            f"Final False Negatives: {final_metrics['false_negatives']} (Baseline: {accuracy_callback.baseline_fn})")
        print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")

        # Log final metrics
        mlflow.log_metrics({
            "final_accuracy": final_metrics['accuracy'],
            "final_false_positives": final_metrics['false_positives'],
            "final_false_negatives": final_metrics['false_negatives'],
            "final_precision": final_metrics['precision'],
            "final_recall": final_metrics['recall'],
            "final_f1": final_metrics['f1'],
            "accuracy_improvement": final_metrics['accuracy'] - accuracy_callback.baseline_accuracy,
            "fp_reduction": accuracy_callback.baseline_fp - final_metrics['false_positives'],
            "fn_reduction": accuracy_callback.baseline_fn - final_metrics['false_negatives']
        })

        # Save final model
        final_model_path = "./best_model/final_model.zip"
        model_rl.save(final_model_path)
        mlflow.log_artifact(final_model_path, "models")
        print(f"Training complete. Final model saved to {final_model_path}")

        return model_rl