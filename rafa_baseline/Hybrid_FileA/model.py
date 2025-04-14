import math
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, cnn_input_size, transformer_feature_size, num_classes, 
                 d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        # CNN part
        self.cnn = nn.Sequential(
            nn.Linear(cnn_input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature embedding for transformer
        self.feature_embedding = nn.Linear(transformer_feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention for feature importance
        self.attention = nn.Sequential(
            nn.Linear(cnn_input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x_cnn, x_transformer, return_importance=False):
        # CNN path
        cnn_features = self.cnn(x_cnn)
        
        # Calculate feature importance
        importance = self.attention(x_cnn)
        importance = torch.softmax(importance, dim=1)
        
        # Transformer path
        # x_transformer shape: (batch_size, seq_len, feature_size)
        transformer_features = self.feature_embedding(x_transformer)
        transformer_features = self.pos_encoder(transformer_features)
        transformer_features = self.transformer_encoder(transformer_features)
        transformer_features = torch.mean(transformer_features, dim=1)  # Global average pooling
        
        # Concatenate features
        combined_features = torch.cat([cnn_features, transformer_features], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        
        if return_importance:
            return output, importance
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 