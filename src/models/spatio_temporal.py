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
    identity embeddings. Fastest to train, good comparison floor.

    Args:
        n_nodes     : Number of graph nodes (28,779)
        n_dynamic   : Number of dynamic input features (10)
        n_static    : Number of static/exog features
        window_size : Input sequence length (12)

    Returns:
        model (STIDModel)
    """
    return STIDModel(
        input_size   = n_dynamic,
        n_nodes      = n_nodes,
        window       = window_size,
        horizon      = 1,
        output_size  = 1,
        hidden_size  = 64,
        n_layers     = 3,
        dropout      = 0.15,
    )


def build_dcrnn(n_nodes, n_dynamic, n_static, window_size):
    """
    DCRNNModel — Diffusion Convolutional Recurrent Neural Network.

    Uses bidirectional graph diffusion convolutions inside RNN cells.
    Captures multi-hop spatial propagation — well suited for
    hydrology-driven vegetation dynamics where upstream conditions
    propagate downstream across the basin.

    Args:
        n_nodes     : Number of graph nodes (28,779)
        n_dynamic   : Number of dynamic input features (10)
        n_static    : Number of static/exog features
        window_size : Input sequence length (12)

    Returns:
        model (DCRNNModel)
    """
    return DCRNNModel(
        input_size  = n_dynamic,
        output_size = 1,
        horizon     = 1,
        exog_size   = n_static,
        hidden_size = 64,
        kernel_size = 2,
        ff_size     = 256,
        n_layers    = 2,
        dropout     = 0.1,
    )


def build_grugcn(n_nodes, n_dynamic, n_static, window_size):
    """
    GRUGCNModel — GRU Encoder + GCN Decoder (time-then-space).

    GRU processes the 12-month temporal sequence per node,
    producing a hidden state that summarises temporal dynamics.
    GCN then decodes spatial structure across the graph.

    This architecture directly mirrors the thesis framing:
    temporal memory (GRU) feeding into spatial inference (GCN).

    Args:
        n_nodes     : Number of graph nodes (28,779)
        n_dynamic   : Number of dynamic input features (10)
        n_static    : Number of static/exog features
        window_size : Input sequence length (12)

    Returns:
        model (GRUGCNModel)
    """
    return GRUGCNModel(
        input_size  = n_dynamic,
        hidden_size = 64,
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

    Learns spatial adjacency directly from data rather than assuming
    fixed 8-neighbour structure. Discovers non-obvious spatial
    relationships (e.g. distant pixels with correlated vegetation
    driven by shared hydrological pathways).

    Most powerful model — highest thesis contribution.

    Args:
        n_nodes     : Number of graph nodes (28,779)
        n_dynamic   : Number of dynamic input features (10)
        n_static    : Number of static/exog features
        window_size : Input sequence length (12)

    Returns:
        model (GraphWaveNetModel)
    """
    return GraphWaveNetModel(
        input_size           = n_dynamic,
        output_size          = 1,
        horizon              = 1,
        exog_size            = n_static,
        hidden_size          = 32,
        ff_size              = 128,
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

    Args:
        name        : One of ['STID', 'DCRNN', 'GRUGCNModel', 'GraphWaveNet']
        n_nodes     : Number of spatial nodes
        n_dynamic   : Number of dynamic input channels
        n_static    : Number of static/exog channels
        window_size : Input sequence length

    Returns:
        model (nn.Module)

    Raises:
        ValueError if name not in registry.
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