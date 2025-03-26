import mlflow
import numpy as np
import sys
import os

# Add the parent directory to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Try both import paths
try:
    from model import BaselineModel
except ImportError:
    try:
        from baseline.model import BaselineModel
    except ImportError:
        print("Could not import BaselineModel using either path")

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('Network_Intrusion_Detection')

logged_model = 'runs:/d0cf2b0317ed4e71846044fe5fbac3c7/baseline_model'

# Try to load with PyTorch first
try:
    loaded_model = mlflow.pytorch.load_model(logged_model)
    print('Loaded model using pytorch flavor')
except Exception as e:
    print(f"PyTorch load failed with error: {e}")
    # Fall back to pyfunc
    try:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print('Loaded model using pyfunc flavor')
    except Exception as e:
        print(f"pyfunc load failed with error: {e}")
