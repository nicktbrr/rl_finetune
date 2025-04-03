import numpy as np
import torch


def td3_inference(
    trained_model,
    baseline_probs,
    hidden_reps,
    input_features=None,
    max_adjustment=0.6
):
    """
    Use a trained TD3 model to adjust baseline predictions without requiring labels

    Args:
        trained_model: The loaded TD3 model
        baseline_probs: Existing probability predictions from baseline model
        hidden_reps: Existing hidden representations from baseline model
        input_features: Original input features (optional, for RL model)
        max_adjustment: Maximum probability adjustment allowed

    Returns:
        corrected_probs: Probabilities after TD3 adjustment
        adjustments: The adjustment values applied to each prediction
    """

    # Convert to numpy arrays if they are tensors
    if isinstance(baseline_probs, torch.Tensor):
        baseline_probs_np = baseline_probs.detach().cpu().numpy()
    else:
        baseline_probs_np = baseline_probs

    if isinstance(hidden_reps, torch.Tensor):
        hidden_reps_np = hidden_reps.detach().cpu().numpy()
    else:
        hidden_reps_np = hidden_reps

    # Create observation vectors directly without using the environment
    num_samples = len(baseline_probs_np)
    corrected_probs = np.zeros_like(baseline_probs_np)
    adjustments = np.zeros_like(baseline_probs_np)

    # For standalone inference, we need to manually create observations
    for i in range(num_samples):
        # Get the baseline probability and features
        prob = baseline_probs_np[i][0]
        features = hidden_reps_np[i]

        # Calculate distance from threshold
        dist_from_threshold = abs(prob - 0.5)

        # Dummy values for label and class distribution
        # (the model was trained with these in the observation)
        dummy_label = 0
        dummy_pos_ratio = 0.5
        dummy_neg_ratio = 0.5

        # Create observation vector (must match the format used during training)
        obs_base = np.concatenate(([prob, dist_from_threshold, dummy_label,
                                   dummy_pos_ratio, dummy_neg_ratio], features))

        # Add input features if provided
        if input_features is not None:
            obs = np.concatenate((obs_base, input_features[i]))
        else:
            obs = obs_base

        # Get action from TD3 model
        action, _ = trained_model.predict(
            obs.astype(np.float32), deterministic=True)

        # Handle scalar or array actions
        if isinstance(action, np.ndarray):
            if action.size == 1:
                raw_action = action.item()
            else:
                raw_action = action[0]
        else:
            raw_action = action

        # Apply the same adjustment logic as in the environment
        confidence = abs(prob - 0.5)

        if confidence > 0.3:
            scaled_action = raw_action * (1 + 2 * confidence**2)
            offset = np.clip(scaled_action, -max_adjustment, max_adjustment)
        else:
            offset = np.clip(raw_action, -0.2, 0.2)

        # Store the adjustment and corrected probability
        adjustments[i] = offset
        corrected_probs[i] = np.clip(prob + offset, 0, 1)

    return corrected_probs, adjustments
