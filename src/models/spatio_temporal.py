# src/models/spatio_temporal.py
"""
Spatiotemporal deep learning models for NDVI prediction
using the Torch Spatiotemporal Library (TSL).

Models:
    1. STIDModel        — Spatial-Temporal Identity (lightweight MLP baseline)
    2. DCRNNModel       — Diffusion Convolutional RNN (multi-hop spatial propagation)
    3. GRUGCNModel      — GRU encoder + GCN decoder (time-then-space)
    4. GraphWaveNetModel — Learned adjacency + dilated temporal convolutions
"""

import torch
import torch.nn as nn
from tsl.nn.models import (
    STIDModel,
    DCRNNModel,
    GRUGCNModel,
    GraphWaveNetModel,
)


def build_stid(n_nodes: int, n_dynamic: int, n_static: int, window_size: int, **kwargs):
    """
    STIDModel — Spatial-Temporal Identity.
    Patches TSL's missing 'exog_embs' initialization bug safely.
    """
    original_reset = STIDModel.reset_parameters

    def patched_reset(self, *args, **kwargs_inner):
        if not hasattr(self, 'exog_embs'):
            self.exog_embs = nn.ModuleList()
        return original_reset(self, *args, **kwargs_inner)

    STIDModel.reset_parameters = patched_reset

    try:
        model = STIDModel(
            input_size   = n_dynamic,
            n_nodes      = n_nodes,
            window       = window_size,
            horizon      = 1,
            output_size  = 1,
            hidden_size  = 32,
            n_layers     = 2,
            dropout      = kwargs.get('dropout', 0.25),
        )
    finally:
        STIDModel.reset_parameters = original_reset

    if not hasattr(model, 'exog_embs'):
        model.exog_embs = nn.ModuleList()

    return model


def build_dcrnn(n_nodes: int, n_dynamic: int, n_static: int, window_size: int, **kwargs):
    """DCRNNModel — Diffusion Convolutional Recurrent Neural Network."""
    return DCRNNModel(
        input_size  = n_dynamic,
        output_size = 1,
        horizon     = 1,
        exog_size   = n_static,
        hidden_size = 64,
        kernel_size = 2,
        ff_size     = 128,
        n_layers    = 2,
        dropout     = kwargs.get('dropout', 0.30),
    )


def build_grugcn(n_nodes: int, n_dynamic: int, n_static: int, window_size: int, **kwargs):
    """
    GRUGCNModel — GRU Encoder + GCN Decoder.
    Explicitly ignores unsupported arguments like 'dropout'.
    """
    return GRUGCNModel(
        input_size  = n_dynamic,
        hidden_size = 96,
        output_size = 1,
        horizon     = 1,
        exog_size   = n_static,
        enc_layers  = 2,
        gcn_layers  = 2,
        norm        = 'mean',
    )


def build_graphwavenet(n_nodes: int, n_dynamic: int, n_static: int, window_size: int, **kwargs):
    """GraphWaveNetModel — Learned Adjacency + Dilated Temporal Convolutions."""
    return GraphWaveNetModel(
        input_size           = n_dynamic,
        output_size          = 1,
        horizon              = 1,
        exog_size            = n_static,
        hidden_size          = 32,
        ff_size              = 128,
        n_layers             = 3,
        temporal_kernel_size = 2,
        spatial_kernel_size  = 2,
        learned_adjacency    = True,
        n_nodes              = n_nodes,
        emb_size             = 4,
        dilation             = 2,
        dilation_mod         = 2,
        norm                 = 'batch',
        dropout              = kwargs.get('dropout', 0.30),
    )


MODEL_REGISTRY = {
    'STID'        : build_stid,
    'DCRNN'       : build_dcrnn,
    'GRUGCNModel' : build_grugcn,
    'GraphWaveNet': build_graphwavenet,
}


def get_model(name: str, n_nodes: int, n_dynamic: int,
              n_static: int, window_size: int, **kwargs) -> nn.Module:
    """Instantiate a model by name from the registry safely."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[name](
        n_nodes=n_nodes,
        n_dynamic=n_dynamic,
        n_static=n_static,
        window_size=window_size,
        **kwargs
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built {name}: {n_params:,} trainable parameters")

    return model