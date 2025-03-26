import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def evaluate_rl_corrections(trained_rl_model, baseline_probs, hidden_reps, true_labels, max_adjustment=0.6, log_to_mlflow=True):
    """
    Evaluate how the RL model corrects baseline predictions.
    Supports the enhanced model with false positive focus.

    Parameters:
    -----------
    trained_rl_model : stable_baselines3 model
        The trained RL model (TD3)
    baseline_probs : numpy array
        Original prediction probabilities from baseline model
    hidden_reps : numpy array
        Hidden representations from baseline model
    true_labels : numpy array
        Ground truth labels
    max_adjustment : float
        Maximum allowed adjustment (should match training)
    log_to_mlflow : bool
        Whether to log results to MLflow

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and predictions
    """
    # Initialize arrays to store results
    n_samples = len(baseline_probs)
    corrected_probs = np.zeros(n_samples)
    actions_taken = np.zeros(n_samples)

    # Calculate class distribution for the enhanced observation space
    positive_ratio = np.mean(true_labels)
    negative_ratio = 1 - positive_ratio

    # Process each sample to get corrected probabilities
    for i in range(n_samples):
        # Get sample data
        prob = baseline_probs[i]
        feats = hidden_reps[i]
        true_label = true_labels[i]

        # Calculate confidence (distance from threshold)
        dist_from_threshold = abs(prob - 0.5)

        # Create observation - including class distribution info
        # Format: [prob, dist_from_threshold, true_label, positive_ratio, negative_ratio, *feats]
        obs = np.concatenate(([prob, dist_from_threshold, float(true_label),
                              positive_ratio, negative_ratio], feats)).astype(np.float32)

        # Get RL model's action
        try:
            action, _ = trained_rl_model.predict(obs, deterministic=True)
            action_value = action[0]  # Extract scalar action

            # Apply same non-linear scaling as in training environment
            confidence = abs(prob - 0.5)

            # Potential false positive case
            if true_label == 0 and prob > 0.5:
                if action_value < 0:  # Downward adjustment
                    max_downward = max_adjustment * 1.5  # Max adjustment * 1.5
                    if confidence > 0.3:
                        scaled_action = action_value * (1 + 3 * confidence**2)
                        offset = np.clip(
                            scaled_action, -max_downward, max_adjustment * 0.5)
                    else:
                        scaled_action = action_value * 1.2
                        offset = np.clip(
                            scaled_action, -max_downward, max_adjustment * 0.7)
                else:  # Upward adjustment
                    offset = np.clip(action_value, 0, max_adjustment * 0.3)
            else:
                # Standard scaling for other cases
                if confidence > 0.3:
                    scaled_action = action_value * (1 + 2 * confidence**2)
                    offset = np.clip(scaled_action, -max_adjustment, max_adjustment)
                else:
                    offset = np.clip(action_value, -0.2, 0.2)

            # Apply correction
            corrected_prob = np.clip(prob + offset, 0, 1)

            # Store results
            corrected_probs[i] = float(corrected_prob)
            actions_taken[i] = float(offset)

        except Exception as e:
            print(f"Warning: Error predicting sample {i}: {e}")
            # Use original probability as fallback
            corrected_probs[i] = prob
            actions_taken[i] = 0

    # Convert to binary predictions
    baseline_preds = (baseline_probs >= 0.5).astype(int)
    corrected_preds = (corrected_probs >= 0.5).astype(int)

    # Calculate metrics
    baseline_cm = confusion_matrix(true_labels, baseline_preds)
    corrected_cm = confusion_matrix(true_labels, corrected_preds)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=baseline_cm).plot(
        ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title('Baseline Model Confusion Matrix')

    # Corrected confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=corrected_cm).plot(
        ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('RL-Corrected Confusion Matrix')

    plt.tight_layout()
    confusion_matrices_path = 'confusion_matrices.png'
    plt.savefig(confusion_matrices_path)
    
    if log_to_mlflow:
        mlflow.log_artifact(confusion_matrices_path, "plots")
    
    plt.close()

    # Extract metrics from confusion matrices
    if baseline_cm.shape == (2, 2):
        baseline_tn, baseline_fp, baseline_fn, baseline_tp = baseline_cm.ravel()
        corrected_tn, corrected_fp, corrected_fn, corrected_tp = corrected_cm.ravel()

        # Calculate changes
        fp_change = baseline_fp - corrected_fp
        fn_change = baseline_fn - corrected_fn

        # Calculate metrics
        baseline_accuracy = (baseline_tp + baseline_tn) / n_samples
        corrected_accuracy = (corrected_tp + corrected_tn) / n_samples

        baseline_precision = baseline_tp / \
            (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0
        corrected_precision = corrected_tp / \
            (corrected_tp + corrected_fp) if (corrected_tp + corrected_fp) > 0 else 0

        baseline_recall = baseline_tp / \
            (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0
        corrected_recall = corrected_tp / \
            (corrected_tp + corrected_fn) if (corrected_tp + corrected_fn) > 0 else 0

        baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision +
                                                                    baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0
        corrected_f1 = 2 * (corrected_precision * corrected_recall) / (corrected_precision +
                                                                       corrected_recall) if (corrected_precision + corrected_recall) > 0 else 0

        # Print detailed comparison
        print("\n--- DETAILED COMPARISON ---")
        print("Baseline Model:")
        print(f"  True Positives: {baseline_tp}")
        print(f"  True Negatives: {baseline_tn}")
        print(f"  False Positives: {baseline_fp}")
        print(f"  False Negatives: {baseline_fn}")
        print(f"  Accuracy: {baseline_accuracy:.4f}")
        print(f"  Precision: {baseline_precision:.4f}")
        print(f"  Recall: {baseline_recall:.4f}")
        print(f"  F1 Score: {baseline_f1:.4f}")

        print("\nRL-Corrected Model:")
        print(f"  True Positives: {corrected_tp}")
        print(f"  True Negatives: {corrected_tn}")
        print(f"  False Positives: {corrected_fp}")
        print(f"  False Negatives: {corrected_fn}")
        print(f"  Accuracy: {corrected_accuracy:.4f}")
        print(f"  Precision: {corrected_precision:.4f}")
        print(f"  Recall: {corrected_recall:.4f}")
        print(f"  F1 Score: {corrected_f1:.4f}")

        print("\nImprovements:")
        print(
            f"  Accuracy: +{(corrected_accuracy - baseline_accuracy) * 100:.2f}%")
        print(
            f"  False Positive Reduction: {fp_change} ({fp_change/baseline_fp*100:.1f}% reduction)")
        print(
            f"  False Negative Reduction: {fn_change} ({fn_change/baseline_fn*100:.1f}% reduction)")
        print(f"  Overall Error Reduction: {fp_change + fn_change} cases")
        
        # Log metrics to MLflow
        if log_to_mlflow:
            mlflow.log_metrics({
                "eval_baseline_accuracy": baseline_accuracy,
                "eval_baseline_precision": baseline_precision,
                "eval_baseline_recall": baseline_recall,
                "eval_baseline_f1": baseline_f1,
                "eval_baseline_fp": baseline_fp,
                "eval_baseline_fn": baseline_fn,
                
                "eval_corrected_accuracy": corrected_accuracy,
                "eval_corrected_precision": corrected_precision,
                "eval_corrected_recall": corrected_recall,
                "eval_corrected_f1": corrected_f1,
                "eval_corrected_fp": corrected_fp,
                "eval_corrected_fn": corrected_fn,
                
                "eval_accuracy_improvement": corrected_accuracy - baseline_accuracy,
                "eval_fp_reduction": fp_change,
                "eval_fp_reduction_pct": fp_change/baseline_fp*100 if baseline_fp > 0 else 0,
                "eval_fn_reduction": fn_change,
                "eval_fn_reduction_pct": fn_change/baseline_fn*100 if baseline_fn > 0 else 0,
                "eval_total_error_reduction": fp_change + fn_change
            })

    # Plot analysis of corrections
    fig = plt.figure(figsize=(12, 10))

    # Histogram of actions
    plt.subplot(2, 2, 1)
    plt.hist(actions_taken, bins=20, edgecolor='black')
    plt.title('Distribution of RL Actions')
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')

    # Scatter plot of original vs corrected probabilities
    plt.subplot(2, 2, 2)
    plt.scatter(baseline_probs, corrected_probs, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
    plt.title('Original vs Corrected Probabilities')
    plt.xlabel('Original Probability')
    plt.ylabel('Corrected Probability')

    # Highlight changed predictions
    changed_indices = np.where(baseline_preds != corrected_preds)[0]

    if len(changed_indices) > 0:
        changed_orig = baseline_probs[changed_indices]
        changed_corr = corrected_probs[changed_indices]
        changed_labels = true_labels[changed_indices]

        # Scatter plot focusing on changes
        plt.subplot(2, 2, 3)
        # Color by true label (red for class 0, blue for class 1)
        colors = ['red' if label == 0 else 'blue' for label in changed_labels]
        plt.scatter(changed_orig, changed_corr, c=colors, alpha=0.7)
        plt.axhline(0.5, color='green', linestyle='--')
        plt.axvline(0.5, color='green', linestyle='--')
        plt.title(f'Changed Predictions ({len(changed_indices)} samples)')
        plt.xlabel('Original Probability')
        plt.ylabel('Corrected Probability')
        plt.xlim(0.3, 0.7)  # Focus on boundary region
        plt.ylim(0.3, 0.7)

        # Success rate of changes
        orig_correct = (baseline_preds[changed_indices]
                        == true_labels[changed_indices]).sum()
        new_correct = (corrected_preds[changed_indices]
                       == true_labels[changed_indices]).sum()
        improvement = new_correct - orig_correct

        plt.subplot(2, 2, 4)
        plt.bar(['Originally\nCorrect', 'Corrected\nCorrect'],
                [orig_correct, new_correct])
        plt.title(f'Success of Changes (+{improvement} correct)')
        plt.ylabel('Number of Correct Predictions')
    else:
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, "No predictions changed", ha='center', va='center')

        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, "No predictions changed", ha='center', va='center')

    plt.tight_layout()
    correction_analysis_path = 'correction_analysis.png'
    plt.savefig(correction_analysis_path)
    
    if log_to_mlflow:
        mlflow.log_artifact(correction_analysis_path, "plots")
    
    plt.close()

    # Analyze false positive corrections specifically
    fp_indices = np.where((baseline_preds == 1) & (true_labels == 0))[0]
    fp_fixed = np.where((baseline_preds == 1) & (
        corrected_preds == 0) & (true_labels == 0))[0]

    if len(fp_indices) > 0:
        print("\n--- FALSE POSITIVE ANALYSIS ---")
        print(f"Total False Positives in Baseline: {len(fp_indices)}")
        print(
            f"False Positives Fixed: {len(fp_fixed)} ({len(fp_fixed)/len(fp_indices)*100:.1f}%)")

        # Distribution of original probabilities for false positives
        plt.figure(figsize=(10, 6))
        plt.hist(baseline_probs[fp_indices],
                 bins=20, alpha=0.5, label='All FPs')

        if len(fp_fixed) > 0:
            plt.hist(baseline_probs[fp_fixed], bins=20,
                     alpha=0.5, label='Fixed FPs')

        plt.axvline(0.5, color='red', linestyle='--')
        plt.title('Distribution of False Positive Probabilities')
        plt.xlabel('Original Probability')
        plt.ylabel('Count')
        plt.legend()
        fp_analysis_path = 'fp_analysis.png'
        plt.savefig(fp_analysis_path)
        
        if log_to_mlflow:
            mlflow.log_artifact(fp_analysis_path, "plots")
            mlflow.log_metrics({
                "fp_total": len(fp_indices),
                "fp_fixed": len(fp_fixed),
                "fp_fixed_pct": len(fp_fixed)/len(fp_indices)*100 if len(fp_indices) > 0 else 0
            })
        
        plt.close()

    # Return results dictionary
    return {
        'baseline_preds': baseline_preds,
        'corrected_preds': corrected_preds,
        'baseline_probs': baseline_probs,
        'corrected_probs': corrected_probs,
        'actions': actions_taken,
        'baseline_cm': baseline_cm,
        'corrected_cm': corrected_cm
    }