import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnhancedFineTuneEnv(gym.Env):
    def __init__(self, baseline_probs, hidden_reps, true_labels, max_adjustment=0.6):
        super().__init__()
        self.baseline_probs = np.array(baseline_probs)
        self.hidden_reps = np.array(hidden_reps)
        self.true_labels = np.array(true_labels)
        self.max_adjustment = max_adjustment

        self.n_samples = len(self.true_labels)
        self.current_index = 0

        # Calculate class distribution to add to observation space
        self.positive_ratio = np.mean(self.true_labels)
        self.negative_ratio = 1 - self.positive_ratio

        # Observations: [prob] + [dist_from_threshold] + [true_label] + [class_ratios] + hidden_reps
        # Added 2 for class distribution info
        obs_dim = 5 + self.hidden_reps.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Expanded action space for larger adjustments
        self.action_space = spaces.Box(
            low=-max_adjustment, high=max_adjustment, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_index >= self.n_samples:
            # Return zeros if we're beyond data bounds
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        prob = self.baseline_probs[self.current_index][0]
        feats = self.hidden_reps[self.current_index]
        true_label = self.true_labels[self.current_index]

        # print(prob, feats.shape, true_label)
        # Calculate distance from decision boundary
        dist_from_threshold = abs(prob - 0.5)

        # print(self.positive_ratio.shape, self.negative_ratio.shape)

        # Include distance, true label, and class distribution as observation features
        temp = np.concatenate(([prob, dist_from_threshold, float(true_label),
                              self.positive_ratio, self.negative_ratio], feats)).astype(np.float32)
        return temp

    def step(self, action):
        prob = self.baseline_probs[self.current_index]
        true_label = self.true_labels[self.current_index]

        # Calculate confidence for non-linear action scaling
        confidence = abs(prob - 0.5)

        # Apply non-linear transformation for larger adjustments when needed
        raw_action = action[0]

        # Potential false positive case - allow larger downward adjustments
        if true_label == 0 and prob > 0.5:  # Potential false positive
            # Allow larger downward adjustments for false positives
            if raw_action < 0:  # Only for downward adjustments
                max_downward = self.max_adjustment * 1.5  # 50% larger for reducing FPs
                if confidence > 0.3:  # High confidence false positive
                    scaled_action = raw_action * \
                        (1 + 3 * confidence**2)  # Stronger scaling
                    offset = np.clip(scaled_action, -max_downward,
                                     self.max_adjustment * 0.5)
                else:
                    scaled_action = raw_action * 1.2  # Modest boost for borderline cases
                    offset = np.clip(scaled_action, -max_downward,
                                     self.max_adjustment * 0.7)
            else:
                # For upward adjustments, be more conservative
                offset = np.clip(raw_action, 0, self.max_adjustment * 0.3)
        else:
            # Standard scaling for other cases
            if confidence > 0.3:  # For highly confident predictions
                # Apply cubic scaling to allow larger adjustments
                scaled_action = raw_action * (1 + 2 * confidence**2)
                # Still clip to prevent extreme values
                offset = np.clip(
                    scaled_action, -self.max_adjustment, self.max_adjustment)
            else:
                # For predictions closer to boundary, use smaller adjustments
                offset = np.clip(raw_action, -0.2, 0.2)

        # Apply correction
        corrected_prob = np.clip(prob + offset, 0, 1)

        # Determine predicted classes
        original_class = 1 if prob >= 0.5 else 0
        predicted_class = 1 if corrected_prob >= 0.5 else 0

        # Modified reward function with class-specific adjustments
        confidence_weight = confidence**2  # Square to emphasize high confidence

        if predicted_class == true_label:
            # Correct classification
            if original_class != true_label:
                # Fixed a mistake
                if true_label == 0:  # Fixed false positive
                    # Higher reward for fixing false positives
                    reward = 10.0 * confidence_weight
                else:  # Fixed false negative
                    reward = 7.0 * confidence_weight
            else:
                # Already correct, reward small improvements
                if true_label == 1:
                    # For positive examples, reward moving closer to 1
                    reward = 1.0 + (corrected_prob - prob) * 3
                else:
                    # For negative examples, reward moving closer to 0
                    reward = 1.0 + (prob - corrected_prob) * 3
        else:
            # Incorrect classification
            if original_class == true_label:
                # Made a correct prediction wrong
                if true_label == 0:  # Created false positive
                    # Stronger penalty for creating false positives
                    reward = -12.0 * confidence_weight
                else:  # Created false negative
                    reward = -8.0 * confidence_weight
            else:
                # Still wrong, but reward movement in right direction
                if true_label == 1 and corrected_prob > prob:
                    # For positive examples, moving toward 1 is good
                    reward = -1.0 + (corrected_prob - prob) * \
                        4 * confidence_weight
                elif true_label == 0 and corrected_prob < prob:
                    # For negative examples, moving toward 0 is good
                    # Higher coefficient for FP reduction
                    reward = -1.0 + (prob - corrected_prob) * \
                        6 * confidence_weight
                else:
                    # Moving in wrong direction
                    if true_label == 0:  # Worse false positive
                        reward = -5.0 * confidence_weight  # Stronger penalty
                    else:  # Worse false negative
                        reward = -3.0 * confidence_weight

        # Move to next example
        self.current_index += 1
        terminated = (self.current_index >= self.n_samples)
        truncated = False

        # Get next observation
        obs = self._get_obs()

        # Return additional info for logging/debugging
        info = {
            'original_prob': prob,
            'corrected_prob': corrected_prob,
            'action': offset,
            'true_label': true_label,
            'original_pred': original_class,
            'new_pred': predicted_class,
            'mistake_confidence': confidence if original_class != true_label else 0,
            'priority': confidence_weight if original_class != true_label else 0.1
        }

        return obs, reward, terminated, truncated, info
