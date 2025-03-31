import numpy as np
import os
from pathlib import Path

def generate_sample_data(n_samples=1000, n_features=49):
    """Generate sample data for testing."""
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    predictions = np.random.random(n_samples)
    
    # Generate hidden representations using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, n_features))
    hidden_reps = pca.fit_transform(X)
    
    # Save the data
    np.save(data_dir / "X_train.npy", X)
    np.save(data_dir / "y_train.npy", y)
    np.save(data_dir / "predictions.npy", predictions)
    np.save(data_dir / "hidden_reps.npy", hidden_reps)
    
    print(f"Generated sample data in {data_dir}")
    print(f"X_train shape: {X.shape}")
    print(f"y_train shape: {y.shape}")
    print(f"predictions shape: {predictions.shape}")
    print(f"hidden_reps shape: {hidden_reps.shape}")

if __name__ == "__main__":
    generate_sample_data() 