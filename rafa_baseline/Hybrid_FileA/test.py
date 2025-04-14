# from model import HybridModel
# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Define your model
# model = HybridModel(
#     cnn_input_size=15,
#     transformer_feature_size=1,
#     num_classes=19,
# )

# # Load the checkpoint
# checkpoint = torch.load('/Users/nicholasbarsi-rhyne/Projects/rl_finetune/rafa_baseline/Hybrid_FileA/model_data/model/best_model.pt',
#                         map_location=torch.device('cpu'),
#                         weights_only=False)

# # Extract just the model state dict from the checkpoint
# if "model_state_dict" in checkpoint:
#     # If the model weights are stored under "model_state_dict" key
#     model.load_state_dict(checkpoint["model_state_dict"])
# else:
#     # Try loading directly if the structure is different
#     try:
#         model.load_state_dict(checkpoint)
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Please check the model structure or the checkpoint format.")


# print("Model loaded successfully!")

# # Read and process the CSV data

# # Read the CSV file
# df = pd.read_csv('../../data_monitoring/50_recent.csv')

# # Convert DataFrame to tensor
# features_tensor = torch.FloatTensor(df.values)

# # Get predictions
# model.eval()  # Set to evaluation mode
# with torch.no_grad():
#     predictions = model(features_tensor)
#     probabilities = torch.softmax(predictions, dim=1)
#     predicted_classes = torch.argmax(probabilities, dim=1)

# # Convert predictions to numpy for plotting
# predicted_classes = predicted_classes.numpy()
# probabilities = probabilities.numpy()

# # Create visualization
# plt.figure(figsize=(15, 5))

# # Plot 1: Distribution of predicted classes
# plt.subplot(1, 3, 1)
# plt.hist(predicted_classes, bins=range(20), alpha=0.7)
# plt.title('Distribution of Predicted Classes')
# plt.xlabel('Class')
# plt.ylabel('Count')

# # Plot 2: Average probability per class
# plt.subplot(1, 3, 2)
# avg_probs = probabilities.mean(axis=0)
# plt.bar(range(len(avg_probs)), avg_probs, alpha=0.7)
# plt.title('Average Probability per Class')
# plt.xlabel('Class')
# plt.ylabel('Average Probability')

# # Plot 3: Confidence distribution
# plt.subplot(1, 3, 3)
# max_probs = np.max(probabilities, axis=1)
# plt.hist(max_probs, bins=20, alpha=0.7)
# plt.title('Distribution of Prediction Confidence')
# plt.xlabel('Confidence')
# plt.ylabel('Count')

# plt.tight_layout()
# plt.savefig('prediction_analysis.png')
# plt.close()

# # Print summary statistics
# print("\nPrediction Analysis:")
# print(f"Total samples processed: {len(df)}")
# print(f"Most common predicted class: {np.bincount(predicted_classes).argmax()}")
# print(f"Number of unique classes predicted: {len(np.unique(predicted_classes))}")
# print(f"Average confidence: {np.mean(max_probs):.3f}")
# print(f"Minimum confidence: {np.min(max_probs):.3f}")
# print(f"Maximum confidence: {np.max(max_probs):.3f}")

from model import HybridModel
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define your model
model = HybridModel(
    cnn_input_size=15,
    transformer_feature_size=1,
    num_classes=19,
)

# Load the checkpoint
checkpoint = torch.load('/Users/nicholasbarsi-rhyne/Projects/rl_finetune/rafa_baseline/Hybrid_FileA/model_data/model/best_model.pt',
                        map_location=torch.device('cpu'),
                        weights_only=False)

# Extract just the model state dict from the checkpoint
if "model_state_dict" in checkpoint:
    # If the model weights are stored under "model_state_dict" key
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # Try loading directly if the structure is different
    try:
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the model structure or the checkpoint format.")

print("Model loaded successfully!")

# Read and process the CSV data
df = pd.read_csv('../../data_monitoring/150_sparse_features.csv')

# Convert DataFrame to tensor for CNN input
features_tensor = torch.FloatTensor(df.values)

# Create transformer input - reshape the features for transformer
# Similar to how it's done in NetworkDataset class from the original code
SEQUENCE_LENGTH = 32  # Using the same sequence length as in the original code
feature_size = df.shape[1]

# Pad the features if needed to make it divisible by sequence_length
if feature_size % SEQUENCE_LENGTH != 0:
    pad_size = SEQUENCE_LENGTH - (feature_size % SEQUENCE_LENGTH)
    X_padded = np.pad(df.values, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
    feature_size_padded = X_padded.shape[1]
    transformer_features = torch.FloatTensor(X_padded).reshape(-1, SEQUENCE_LENGTH, feature_size_padded // SEQUENCE_LENGTH)
else:
    transformer_features = torch.FloatTensor(df.values).reshape(-1, SEQUENCE_LENGTH, feature_size // SEQUENCE_LENGTH)

# Get predictions
model.eval()  # Set to evaluation mode
with torch.no_grad():
    predictions = model(features_tensor, transformer_features)
    probabilities = torch.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

# Convert predictions to numpy for plotting
predicted_classes = predicted_classes.numpy()
probabilities = probabilities.numpy()

print(probabilities[0], len(probabilities[0]))

# Create visualization
plt.figure(figsize=(15, 5))

# Plot 1: Distribution of predicted classes
plt.subplot(1, 3, 1)
plt.hist(predicted_classes, bins=range(19), alpha=0.7)
plt.title('Distribution of Predicted Classes')
plt.xlabel('Class')
plt.ylabel('Count')

# Plot 2: Average probability per class
plt.subplot(1, 3, 2)
avg_probs = probabilities.mean(axis=0)
plt.bar(range(len(avg_probs)), avg_probs, alpha=0.7)
plt.title('Average Probability per Class')
plt.xlabel('Class')
plt.ylabel('Average Probability')

# Plot 3: Confidence distribution
plt.subplot(1, 3, 3)
max_probs = np.max(probabilities, axis=1)
plt.hist(max_probs, bins=19, alpha=0.7)
plt.title('Distribution of Prediction Confidence')
plt.xlabel('Confidence')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('prediction_analysis.png')
plt.close()

# Print summary statistics
print("\nPrediction Analysis:")
print(f"Total samples processed: {len(df)}")
print(f"Most common predicted class: {np.bincount(predicted_classes).argmax()}")
print(f"Number of unique classes predicted: {len(np.unique(predicted_classes))}")
print(f"Average confidence: {np.mean(max_probs):.3f}")
print(f"Minimum confidence: {np.min(max_probs):.3f}")
print(f"Maximum confidence: {np.max(max_probs):.3f}")