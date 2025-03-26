import numpy as np
import pandas as pd
from utils import preprocess_data


def create_perturbed_data(test_df, seed=42, perturb_ratio=0.5, noise_loc=0.3, noise_scale=0.5):
    """
    Create perturbed test data by adding noise to half of the normal class samples.

    Args:
        test_df (pd.DataFrame): The test DataFrame containing features and target.
        model: Trained model with a predict method.
        seed (int): Random seed for reproducibility.
        perturb_ratio (float): Proportion of normal samples to perturb.
        noise_loc (float): Mean of the normal distribution for noise.
        noise_scale (float): Standard deviation of the normal distribution for noise.

    Returns:
        X_test_perturbed (np.ndarray): Perturbed feature array.
        y_test_filtered (np.ndarray): Corresponding labels (normal class only).
        y_pred (np.ndarray): Model predictions on perturbed data.
    """
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    normal_mask = (y_test == 0)
    X_test_filtered = X_test[normal_mask]
    y_test_filtered = y_test[normal_mask]

    normal_indices = np.arange(len(y_test_filtered))

    np.random.seed(seed)
    perturb_indices = np.random.choice(
        normal_indices, size=int(len(normal_indices) * perturb_ratio), replace=False)

    X_test_perturbed = X_test_filtered.copy()
    noise = np.random.normal(
        loc=noise_loc, scale=noise_scale, size=X_test_perturbed[perturb_indices].shape)
    X_test_perturbed[perturb_indices] += noise

    return X_test_perturbed, y_test_filtered


def main():
    # Load datasets
    train_df = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    test_df = pd.read_csv('data/UNSW_NB15_training-set.csv')

    # Preprocess data
    _, _, X_test, y_test = preprocess_data(train_df, test_df)

    # Create DataFrame from processed test data
    processed_test_df = pd.DataFrame(np.column_stack((X_test, y_test)))

    # Perturb the training data
    X_train_perturbed, y_train_filtered = create_perturbed_data(
        processed_test_df)

    merged_X = np.vstack([X_test, X_train_perturbed])
    merged_y = np.concatenate([y_test, y_train_filtered])
    merged_df = pd.DataFrame(np.column_stack((merged_X, merged_y)))

    merged_df.to_csv('data/perturbed_train_data.csv', index=False)

    print("Perturbed training data created successfully")


if __name__ == "__main__":
    main()
