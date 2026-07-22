# src/models/spatio_temporal.py
"""
Spatiotemporal deep learning models for NDVI prediction
using the Torch Spatiotemporal Library (TSL).

Models:
    1. STIDModel       — Spatial-Temporal Identity (lightweight MLP baseline)
    2. DCRNNModel      — Diffusion Convolutional RNN (multi-hop spatial propagation)
    3. GRUGCNModel     — GRU encoder + GCN decoder (time-then-space)
    4. GraphWaveNetModel — Learned adjacency + dilated temporal convolutions

All models:
    - Input  : [batch, time, nodes, features]
    - Output : [batch, nodes, 1] — predicted log_ndvi at t+1
    - Target : log_ndvi (continuous, regression task)
"""

import torch
import torch.nn as nn
from tsl.nn.models import (
    STIDModel,
    DCRNNModel,
    GRUGCNModel,
    GraphWaveNetModel,
)


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

def build_stid(n_nodes, n_dynamic, n_static, window_size):
    """
    STIDModel — Spatial-Temporal Identity.

    Lightweight MLP-based model that uses spatial and temporal
    identity embeddings. Fast to train, good comparison floor.
    
    Note: Safely patches a TSL library bug where STIDModel.reset_parameters()
    crashes looking for 'exog_embs' before it is initialized.
    """
    # Grab the original reset_parameters function
    original_reset = STIDModel.reset_parameters
    
    # Intercept the reset call and set exog_embs after PyTorch setup is complete
    def patched_reset(self, *args, **kwargs):
        if not hasattr(self, 'exog_embs'):
            self.exog_embs = nn.ModuleList()
        return original_reset(self, *args, **kwargs)

    # Apply the runtime patch
    STIDModel.reset_parameters = patched_reset
    
    try:
        model = STIDModel(
            input_size   = n_dynamic,
            n_nodes      = n_nodes,
            window       = window_size,
            horizon      = 1,
            output_size  = 1,
            hidden_size  = 64,
            n_layers     = 3,
            dropout      = 0.15,
        )
    finally:
        # Restore original class method to keep clean behavior
        STIDModel.reset_parameters = original_reset

    # Double check attribute assignment on final instance
    if not hasattr(model, 'exog_embs'):
        model.exog_embs = nn.ModuleList()
        
    return model


def build_dcrnn(n_nodes, n_dynamic, n_static, window_size):
    """
    DCRNNModel — Diffusion Convolutional Recurrent Neural Network.
    """
    return DCRNNModel(
        input_size  = n_dynamic,
        output_size = 1,
        horizon     = 1,
        exog_size   = n_static,
        hidden_size = 128,        # Upscaled from 64 for dense hidden tracking
        kernel_size = 2,
        ff_size     = 512,        # Upscaled from 256 to expand layer capacity
        n_layers    = 2,
        dropout     = 0.1,
    )


def build_grugcn(n_nodes, n_dynamic, n_static, window_size):
    """
    GRUGCNModel — GRU Encoder + GCN Decoder (time-then-space).
    """
    return GRUGCNModel(
        input_size  = n_dynamic,
        hidden_size = 128,        # Upscaled from 64 to enrich temporal representations
        output_size = 1,
        horizon     = 1,
        exog_size   = n_static,
        enc_layers  = 2,
        gcn_layers  = 2,
        norm        = 'mean',
    )


def build_graphwavenet(n_nodes, n_dynamic, n_static, window_size):
    """
    GraphWaveNetModel — Learned Adjacency + Dilated Temporal Convolutions.
    """
    return GraphWaveNetModel(
        input_size           = n_dynamic,
        output_size          = 1,
        horizon              = 1,
        exog_size            = n_static,
        hidden_size          = 64,         # Upscaled from 32 to expand causal filter depth
        ff_size              = 256,        # Upscaled from 128 for intermediate dense layers
        n_layers             = 4,
        temporal_kernel_size = 2,
        spatial_kernel_size  = 2,
        learned_adjacency    = True,
        n_nodes              = n_nodes,
        emb_size             = 10,
        dilation             = 2,
        dilation_mod         = 2,
        norm                 = 'batch',
        dropout              = 0.2,
    )


# ------------------------------------------------------------------
# Registry — maps model name to builder function
# ------------------------------------------------------------------

MODEL_REGISTRY = {
    'STID'          : build_stid,
    'DCRNN'         : build_dcrnn,
    'GRUGCNModel'   : build_grugcn,
    'GraphWaveNet'  : build_graphwavenet,
}


def get_model(name: str, n_nodes: int, n_dynamic: int,
              n_static: int, window_size: int) -> nn.Module:
    """
    Instantiate a model by name from the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[name](n_nodes, n_dynamic, n_static, window_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built {name}: {n_params:,} trainable parameters")

    return model