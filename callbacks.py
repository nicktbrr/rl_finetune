import numpy as np
import mlflow
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os


class AccuracyCheckpointCallback(BaseCallback):
    """
    Callback for saving the model when classification accuracy improves,
    with special focus on reducing false positives.
    """

    def __init__(self,
                 eval_env,
                 baseline_probs,
                 true_labels,
                 check_freq=5000,
                 save_path='./best_model/',
                 fp_weight=1.2,  # Higher weight for false positives in the score
                 log_to_mlflow=True,
                 verbose=1):
        super(AccuracyCheckpointCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.baseline_probs = np.array(baseline_probs.squeeze())
        self.true_labels = np.array(true_labels)
        self.check_freq = check_freq
        self.save_path = save_path
        self.fp_weight = fp_weight  # Weight for false positives in score calculation
        self.log_to_mlflow = log_to_mlflow

        # Initialize best metrics
        self.best_accuracy = 0.0
        self.best_weighted_score = 0.0
        self.best_fp = float('inf')
        self.best_fn = float('inf')

        # Calculate baseline metrics for comparison
        self.baseline_preds = (self.baseline_probs >= 0.5).astype(int)
        self.baseline_accuracy = np.mean(
            self.baseline_preds == self.true_labels)
        self.baseline_cm = confusion_matrix(
            self.true_labels, self.baseline_preds)
        self.baseline_fp = self.baseline_cm[0, 1]
        self.baseline_fn = self.baseline_cm[1, 0]

        # Ensure the save path exists
        os.makedirs(self.save_path, exist_ok=True)

        # Log baseline metrics
        if self.verbose > 0:
            print(f"\n--- Baseline Model Metrics ---")
            print(f"Accuracy: {self.baseline_accuracy:.4f}")
            print(f"False Positives: {self.baseline_fp}")
            print(f"False Negatives: {self.baseline_fn}")
            print(f"Confusion Matrix: \n{self.baseline_cm}")
            print("----------------------------\n")

        # Log baseline metrics to MLflow
        if self.log_to_mlflow:
            mlflow.log_metrics({
                "baseline_accuracy": self.baseline_accuracy,
                "baseline_false_positives": self.baseline_fp,
                "baseline_false_negatives": self.baseline_fn
            })

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current metrics
            current_metrics = self._evaluate_model()
            accuracy = current_metrics['accuracy']
            fp = current_metrics['false_positives']
            fn = current_metrics['false_negatives']
            precision = current_metrics['precision']
            recall = current_metrics['recall']
            f1 = current_metrics['f1']

            # Calculate weighted score that prioritizes false positive reduction
            # This score will be higher for better models, especially those that reduce FP
            weighted_score = accuracy - \
                (self.fp_weight * fp / len(self.true_labels))

            # Log the current metrics
            print(f"\nStep {self.n_calls} - Evaluation Metrics:")
            print(
                f"Accuracy: {accuracy:.4f} (Best: {self.best_accuracy:.4f}, Baseline: {self.baseline_accuracy:.4f})")
            print(
                f"FP: {fp} (Best: {self.best_fp}, Baseline: {self.baseline_fp})")
            print(
                f"FN: {fn} (Best: {self.best_fn}, Baseline: {self.baseline_fn})")
            print(
                f"Weighted Score: {weighted_score:.4f} (Best: {self.best_weighted_score:.4f})")

            # Record metrics in tensorboard if available
            self.logger.record("eval/accuracy", accuracy)
            self.logger.record("eval/false_positives", fp)
            self.logger.record("eval/false_negatives", fn)
            self.logger.record("eval/weighted_score", weighted_score)
            self.logger.record("eval/precision", precision)
            self.logger.record("eval/recall", recall)
            self.logger.record("eval/f1", f1)

            # Log to MLflow
            if self.log_to_mlflow:
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "weighted_score": weighted_score,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }, step=self.n_calls)

            # Check if metrics have improved
            accuracy_improved = accuracy > self.best_accuracy
            fp_improved = fp < self.best_fp
            score_improved = weighted_score > self.best_weighted_score

            # Save model if accuracy has improved
            if accuracy_improved:
                improvement = accuracy - self.best_accuracy
                self.best_accuracy = accuracy
                model_path = f"{self.save_path}/best_accuracy_model.zip"
                self.model.save(model_path)

                if self.log_to_mlflow:
                    mlflow.log_artifact(model_path, "models")

                if self.verbose > 0:
                    print(
                        f"✅ Saved model with best accuracy: {accuracy:.4f} (improved by {improvement:.4f})")

            # Save model if weighted score has improved (prioritizes FP reduction)
            if score_improved:
                self.best_weighted_score = weighted_score
                model_path = f"{self.save_path}/best_weighted_model.zip"
                self.model.save(model_path)

                if self.log_to_mlflow:
                    mlflow.log_artifact(model_path, "models")

                if self.verbose > 0:
                    print(
                        f"✅ Saved model with best weighted score: {weighted_score:.4f}")

            # Save model if false positives have decreased
            if fp_improved:
                improvement = self.best_fp - fp
                self.best_fp = fp
                model_path = f"{self.save_path}/best_fp_model.zip"
                self.model.save(model_path)

                if self.log_to_mlflow:
                    mlflow.log_artifact(model_path, "models")

                if self.verbose > 0:
                    print(
                        f"Saved model with best FP: {fp} (improved by {improvement})")

            # Update best false negatives if improved
            if fn < self.best_fn:
                self.best_fn = fn

            # Save checkpoint at specific accuracy thresholds if crossed
            if accuracy_improved:
                for threshold in [0.87, 0.88, 0.89, 0.90, 0.91, 0.92]:
                    if self.best_accuracy >= threshold and (self.best_accuracy - improvement) < threshold:
                        threshold_path = f"{self.save_path}/model_acc{int(threshold*100)}.zip"
                        self.model.save(threshold_path)

                        if self.log_to_mlflow:
                            mlflow.log_artifact(
                                threshold_path, "threshold_models")

                        if self.verbose > 0:
                            print(
                                f"Saved threshold model at {threshold_path} (accuracy ≥ {threshold:.2f})")

        return True

    def _evaluate_model(self):
        """
        Evaluate the current model on all validation data

        Returns:
            dict: Dictionary with various metrics
        """
        # Predict using current model
        corrected_probs = self._predict_batch(self.baseline_probs)
        corrected_preds = (corrected_probs >= 0.5).astype(int)

        # Calculate basic metrics
        accuracy = np.mean(corrected_preds == self.true_labels)
        cm = confusion_matrix(self.true_labels, corrected_preds)

        # Extract values from confusion matrix
        try:
            tn, fp, fn, tp = cm.ravel()
        except ValueError:
            # Handle case when confusion matrix doesn't have expected shape
            cm_shape = cm.shape
            if cm_shape == (1, 1):
                # Only one class present in predictions
                if self.true_labels[0] == 0:  # If that class is negative
                    tn = cm[0, 0]
                    fp, fn, tp = 0, 0, 0
                else:  # If that class is positive
                    tp = cm[0, 0]
                    tn, fp, fn = 0, 0, 0
            else:
                # For other unexpected shapes, set defaults
                tn, fp, fn, tp = 0, 0, 0, 0

        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _predict_batch(self, baseline_probs):
        """
        Predict on a batch of data using the current model

        Args:
            baseline_probs: Original prediction probabilities

        Returns:
            np.array: Corrected probabilities
        """
        corrected_probs = np.copy(baseline_probs)

        # Get environment attributes
        features = self.eval_env.get_attr('hidden_reps')[0]
        true_labels = self.eval_env.get_attr('true_labels')[0]
        positive_ratio = self.eval_env.get_attr('positive_ratio')[0]
        negative_ratio = self.eval_env.get_attr('negative_ratio')[0]

        # Process each example
        for i in range(len(baseline_probs)):
            prob = baseline_probs[i].squeeze()
            feature = features[i]
            true_label = true_labels[i]

            # Create observation
            dist_from_threshold = abs(prob - 0.5)

            obs = np.concatenate(([prob, dist_from_threshold, float(true_label),
                                   positive_ratio, negative_ratio], feature)).astype(np.float32)

            # Get model's action
            action, _ = self.model.predict(obs, deterministic=True)

            # Apply the same non-linear scaling as in the environment
            confidence = abs(prob - 0.5)

            # Potential false positive case
            if true_label == 0 and prob > 0.5:
                if action[0] < 0:  # Downward adjustment
                    max_downward = self.eval_env.get_attr(
                        'max_adjustment')[0] * 1.5
                    if confidence > 0.3:
                        scaled_action = action[0] * (1 + 3 * confidence**2)
                        offset = np.clip(scaled_action, -max_downward,
                                         self.eval_env.get_attr('max_adjustment')[0] * 0.5)
                    else:
                        scaled_action = action[0] * 1.2
                        offset = np.clip(scaled_action, -max_downward,
                                         self.eval_env.get_attr('max_adjustment')[0] * 0.7)
                else:  # Upward adjustment (should be rare for FP case)
                    offset = np.clip(action[0], 0, self.eval_env.get_attr(
                        'max_adjustment')[0] * 0.3)
            else:
                # Standard scaling for other cases
                if confidence > 0.3:
                    scaled_action = action[0] * (1 + 2 * confidence**2)
                    offset = np.clip(scaled_action, -self.eval_env.get_attr('max_adjustment')[0],
                                     self.eval_env.get_attr('max_adjustment')[0])
                else:
                    offset = np.clip(action[0], -0.2, 0.2)

            # Apply correction
            corrected_probs[i] = np.clip(prob + offset, 0, 1)

        return corrected_probs
