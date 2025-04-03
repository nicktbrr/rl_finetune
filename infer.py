from model import HybridModel as HybridB
from model import HybridModel as HybridA
from Transformer_FileB.Transformer_FileB_train import TransformerClassifier as TransformerB
from Transformer_FileA.Transformer_FileA_train import TransformerClassifier as TransformerA
from nids_basic_classify_B import ModifiedNet as BCNetB
from nids_basic_classify_A import ModifiedNet as BCNet
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
from pathlib import Path
import json
import requests
import numpy as np
import pandas as pd
import torch
import os
import sys
import warnings

# Suppress PyTorch warnings about weights_only
warnings.filterwarnings('ignore', category=FutureWarning,
                        module='torch.serialization')


# Add the parent directory to Python path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Add model directories to path
for model_dir in ['BC_FileA', 'BC_FileB', 'Transformer_FileA', 'Transformer_FileB', 'Hybrid_FileA', 'Hybrid_FileB']:
    model_path = current_dir / model_dir
    if str(model_path) not in sys.path:
        sys.path.append(str(model_path))


# Model imports

# Constants
ATTACK_CLASSES = [
    'Backdoor', 'Benign', 'Bot', 'Brute Force -Web', 'Brute Force -XSS',
    'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP',
    'DOS', 'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest',
    'DoS attacks-Slowloris', 'FTP-BruteForce', 'Infilteration', 'Injection',
    'SQL Injection', 'SSH-Bruteforce', 'Scanning'
]
NUM_CLASSES = len(ATTACK_CLASSES)

# Model hyperparameters (matching training)
TRANSFORMER_CONFIG = {
    'N_LAYERS': 3,
    'ATTENTION_HEADS': 8,
    'WINDOW_SIZE': 10,
    'DROPOUT': 0.2,
    'FF_NEURONS': 512
}


class ModelInference:
    def __init__(self):
        self.device = torch.device(
            'cuda:2' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.models = {
            'FileA': self._load_models('FileA'),
            'FileB': self._load_models('FileB')
        }

    def _load_models(self, file_type):
        models = {}

        # Load Basic Classifier
        bc_path = Path(f'BC_{file_type}/model_data/model/best_model.pt')
        print(f"Loading BC model from: {bc_path}")
        bc_checkpoint = torch.load(bc_path, map_location=self.device)
        print(f"BC checkpoint keys: {bc_checkpoint.keys()}")

        # Get first layer's weight shape for input size
        first_layer_weight = None
        for key, value in bc_checkpoint['model_state_dict'].items():
            if 'weight' in key and len(value.shape) == 2:
                first_layer_weight = value
                break

        input_size = first_layer_weight.shape[1]
        print(f"Detected input size: {input_size}")

        models['BC'] = {
            'model': self._load_bc_model(bc_checkpoint, file_type, input_size, NUM_CLASSES),
            'scaler': bc_checkpoint.get('scaler', None),
            'label_encoder': bc_checkpoint.get('label_encoder', None),
            'feature_encoders': bc_checkpoint.get('feature_encoders', {})
        }

        # Load Transformer model
        transformer_path = Path(
            f'Transformer_{file_type}/model_data/model/best_model.pt')
        print(f"Loading Transformer model from: {transformer_path}")
        transformer_checkpoint = torch.load(
            transformer_path, map_location=self.device)
        print(f"Transformer checkpoint keys: {transformer_checkpoint.keys()}")

        models['Transformer'] = {
            'model': self._load_transformer_model(transformer_checkpoint, file_type, input_size, NUM_CLASSES),
            'scaler': transformer_checkpoint.get('scaler', None),
            'label_encoder': transformer_checkpoint.get('label_encoder', None),
            'feature_encoders': transformer_checkpoint.get('feature_encoders', {})
        }

        # Load Hybrid model
        hybrid_path = Path(
            f'Hybrid_{file_type}/model_data/model/best_model.pt')
        print(f"Loading Hybrid model from: {hybrid_path}")
        hybrid_checkpoint = torch.load(hybrid_path, map_location=self.device)
        print(f"Hybrid checkpoint keys: {hybrid_checkpoint.keys()}")

        models['Hybrid'] = {
            'model': self._load_hybrid_model(hybrid_checkpoint, file_type, input_size, NUM_CLASSES),
            'scaler': hybrid_checkpoint.get('scaler', None),
            'label_encoder': hybrid_checkpoint.get('label_encoder', None),
            'feature_encoders': hybrid_checkpoint.get('feature_encoders', {})
        }

        return models

    def _load_bc_model(self, checkpoint, file_type, input_size, output_size):
        model_class = BCNet if file_type == 'FileA' else BCNetB
        model = model_class(
            input_size=input_size,
            hidden_size_arr=[2048, 1024, 512, 256],
            output_size=output_size,
            dropout_rate=0.4
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _load_transformer_model(self, checkpoint, file_type, input_size, output_size):
        if file_type == 'FileA':
            from Transformer_FileA.utils.attention import MultiHeadAttention
            from Transformer_FileA.utils.layers import EncoderLayer, Norm
        else:
            from Transformer_FileB.utils.attention import MultiHeadAttention
            from Transformer_FileB.utils.layers import EncoderLayer, Norm

        class TransformerClassifierFixed(nn.Module):
            def __init__(self, d_model, N_layers, attention, window, device, dropout, d_ff, n_classes):
                super().__init__()
                self.device = device
                self.window = window
                self.d_model = d_model

                # Encoder layers
                self.encoder_norm = Norm(d_model, device)
                encoder_layer = EncoderLayer(
                    d_model, attention, device, dropout, d_ff)
                self.encoder_layers = nn.ModuleList(
                    [encoder_layer for _ in range(N_layers)])

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(d_model * window, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, n_classes)
                )

            def forward(self, x):
                # Pass through encoder layers
                for encoder in self.encoder_layers:
                    x = encoder(x)

                # Apply final normalization
                x = self.encoder_norm(x)

                # Flatten and classify
                x = x.reshape(x.shape[0], -1)
                x = self.classifier(x)
                return x

        # Initialize model with exact same parameters as checkpoint
        model = TransformerClassifierFixed(
            d_model=input_size,
            N_layers=TRANSFORMER_CONFIG['N_LAYERS'],
            attention=TRANSFORMER_CONFIG['ATTENTION_HEADS'],
            window=TRANSFORMER_CONFIG['WINDOW_SIZE'],
            device=self.device,
            dropout=TRANSFORMER_CONFIG['DROPOUT'],
            d_ff=TRANSFORMER_CONFIG['FF_NEURONS'],
            n_classes=output_size
        ).to(self.device)

        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        return model

    def _load_hybrid_model(self, checkpoint, file_type, input_size, output_size):
        model_class = HybridA if file_type == 'FileA' else HybridB

        # Calculate transformer feature size like in training
        sequence_length = 32  # Same as SEQUENCE_LENGTH in training
        if input_size % sequence_length != 0:
            pad_size = sequence_length - (input_size % sequence_length)
            n_features_padded = input_size + pad_size
            transformer_feature_size = n_features_padded // sequence_length
        else:
            transformer_feature_size = input_size // sequence_length

        model = model_class(
            cnn_input_size=input_size,
            transformer_feature_size=transformer_feature_size,
            num_classes=output_size,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def preprocess_data(self, data, model_info):
        # Extract features from cnrt field
        features = pd.json_normalize(data['_source']['cnrt'])

        # Core features for FileA (15 features that were used in training)
        core_features = [
            'resp_pkts_conn', 'orig_bytes_conn', 'local_resp_conn', 'duration_conn',
            'orig_ip_bytes_conn', 'orig_asn_geo', 'orig_pkts_conn',
            'resp_bytes_conn', 'local_orig_conn', 'resp_p_id', 'orig_long_geo',
            'missed_bytes_conn', 'orig_lat_geo', 'orig_p_id', 'resp_ip_bytes_conn'
        ]

        # Create a DataFrame with all required features initialized to 0
        numeric_features = pd.DataFrame(
            0, index=np.arange(len(features)), columns=core_features)

        # Fill in available values
        for col in core_features:
            if col in features.columns:
                try:
                    numeric_features[col] = pd.to_numeric(
                        features[col], errors='coerce').fillna(0)
                except Exception as e:
                    # Keep the default 0 if conversion fails
                    continue

        # Convert boolean columns to int
        bool_cols = ['local_resp_conn', 'local_orig_conn']
        for col in bool_cols:
            if col in features.columns:
                numeric_features[col] = features[col].astype(int)

        # Convert to float32 array
        features_array = numeric_features.astype(np.float32).values

        # Scale features if scaler exists
        if model_info['scaler'] is not None:
            features_array = model_info['scaler'].transform(features_array)

        return features_array

    def predict(self, data, file_type):
        results = {}

        # Get features once for all models
        features = self.preprocess_data(data, self.models[file_type]['BC'])
        features_tensor = torch.FloatTensor(features).to(self.device)

        # Basic Classifier prediction
        with torch.no_grad():
            bc_model_info = self.models[file_type]['BC']
            bc_outputs = bc_model_info['model'](features_tensor)
            _, predicted = torch.max(bc_outputs, 1)

            if bc_model_info['label_encoder'] is not None:
                label = bc_model_info['label_encoder'].inverse_transform(
                    predicted.cpu().numpy())[0]
            else:
                label = ATTACK_CLASSES[predicted.item()]
            results['BC'] = label

        # Transformer prediction
        with torch.no_grad():
            transformer_model_info = self.models[file_type]['Transformer']

            # Reshape input for transformer (add window dimension)
            window_size = TRANSFORMER_CONFIG['WINDOW_SIZE']
            X_transformer = features_tensor.unsqueeze(0)  # Add batch dimension
            if X_transformer.shape[0] < window_size:
                # Pad if needed
                pad_size = window_size - X_transformer.shape[0]
                X_transformer = torch.nn.functional.pad(
                    X_transformer, (0, 0, 0, pad_size))

            transformer_outputs = transformer_model_info['model'](
                X_transformer)
            _, predicted = torch.max(transformer_outputs, 1)

            if transformer_model_info['label_encoder'] is not None:
                label = transformer_model_info['label_encoder'].inverse_transform(
                    predicted.cpu().numpy())[0]
            else:
                label = ATTACK_CLASSES[predicted.item()]
            results['Transformer'] = label

        # Hybrid model prediction
        with torch.no_grad():
            hybrid_model_info = self.models[file_type]['Hybrid']

            # Prepare CNN input
            X_cnn = features_tensor

            # Prepare transformer input
            sequence_length = 32
            if features.shape[1] % sequence_length != 0:
                pad_size = sequence_length - \
                    (features.shape[1] % sequence_length)
                features_padded = np.pad(
                    features, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
                feature_size = features_padded.shape[1]
            else:
                features_padded = features
                feature_size = features.shape[1]

            X_transformer = torch.FloatTensor(features_padded).reshape(
                -1, sequence_length, feature_size // sequence_length).to(self.device)

            # Get prediction
            hybrid_outputs = hybrid_model_info['model'](X_cnn, X_transformer)
            _, predicted = torch.max(hybrid_outputs, 1)

            if hybrid_model_info['label_encoder'] is not None:
                label = hybrid_model_info['label_encoder'].inverse_transform(
                    predicted.cpu().numpy())[0]
            else:
                label = ATTACK_CLASSES[predicted.item()]
            results['Hybrid'] = label

        return results


def main():
    # Initialize model inference
    inference = ModelInference()

    # Elasticsearch endpoint
    es_endpoint = "http://13.65.147.232:9200/zeek-traffic-index/_search"

    while True:
        try:
            # Get data from Elasticsearch
            response = requests.get(es_endpoint)
            data = response.json()

            for hit in data['hits']['hits']:
                # Determine tenant
                tenant = hit['_source']['cnrt']['tenant_id_ext']
                file_type = 'FileA' if 'sanjose' in tenant.lower() else 'FileB'

                # Get predictions
                predictions = inference.predict(hit, file_type)

                # Print predictions from all models
                print(
                    f"BC: {predictions['BC']} | Transformer: {predictions['Transformer']} | Hybrid: {predictions['Hybrid']}")

                sys.stdout.flush()

        except Exception as e:
            print(f"Error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
