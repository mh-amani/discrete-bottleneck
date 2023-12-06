import numpy as np
import torch
from vqvae import VQVAEDiscreteLayer 

X = torch.randn(10, 5, 32) # batch_size, seq_len, embedding_dim
print("X:", X) 
print("X.shape:", X.shape) # (10, 5, 32)

# Create a discrete layer
params = {
    'temperature': 1.0,
    'label_smoothing_scale': 0.0,
    "dist_ord": 2,
    'vocab_size': 3,
    'dictionary_dim': 32,
    'hard': True,
    'projection_method': "layer norm", # "unit-sphere" "scale" "layer norm" or "None"
    'beta': 0.25
}

discrete_layer = VQVAEDiscreteLayer(**params)
print("discrete_layer:", discrete_layer)

# Discretize the input
indices, probs, quantized, vq_loss = discrete_layer.discretize(X)
print("indices:", indices)
print("probs:", probs)
print("quantized:", quantized)
print("vq_loss:", vq_loss)

