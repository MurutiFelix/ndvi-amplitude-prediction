# src/models/train.py
"""
Deep Learning training loop for spatiotemporal NDVI prediction.

Trains 4 TSL graph models sequentially:
    1. STIDModel
    2. DCRNNModel
    3. GRUGCNModel
    4. GraphWaveNetModel

Each model is trained on the same train/test split,
evaluated on the same test set, and results saved to
data/processed/dl_metrics.csv for comparison with baselines.

Usage:
    python -m src.models.train

Output:
    data/processed/checkpoint_{model}.pt
    data/processed/history_{model}.csv
    data/processed/dl_metrics.csv
"""

import os
import yaml
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

from src.data.dataset import NDVIGraphDataset, build_datasets
from src.models.spatio_temporal import get_model, MODEL_REGISTRY
from src.utils.spatial import get_edge_index

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

WINDOW_SIZE  = 12       # months lookback
BATCH_SIZE   = 1        # one spatial snapshot per batch (full graph)
N_EPOCHS     = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10      # early stopping patience
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R² and RMSE while ignoring missing/masked values safely."""
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if valid.sum() == 0:
        return {'R2_Score': 0.0, 'RMSE': 1e6}
    rmse = np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
    r2   = r2_score(y_true[valid], y_pred[valid])
    return {'R2_Score': r2, 'RMSE': rmse}


def prepare_tensors(x, u, y):
    """
    Ensures input tensors match TSL format: [Batch, Time, Nodes, Features]
    Swaps axes if necessary, and projects 3D static features to 4D.
    """
    # 1. Align Dynamic Features: x
    # Nodes (N ~ 28779) will be much larger than lookback Window (T = 12).
    # Swap axes if index 1 (Nodes) is larger than index 2 (Time).
    if x.dim() == 4 and x.shape[1] > x.shape[2]:
        x = x.transpose(1, 2)

    # 2. Align Static Exogenous Features: u
    if u is not None:
        if u.dim() == 3:
            # Shape is [Batch, Nodes, Features] (static, no time dimension).
            # Expand u over the Time dimension of x so shape becomes [Batch, Time, Nodes, Features]
            u = u.unsqueeze(1)  # -> [Batch, 1, Nodes, Features]
            u = u.expand(-1, x.shape[1], -1, -1)  # -> [Batch, Time, Nodes, Features]
        elif u.dim() == 4 and u.shape[1] > u.shape[2]:
            u = u.transpose(1, 2)

    # 3. Align Targets: y to [Batch, Nodes, 1]
    if y.dim() == 4:
        if y.shape[1] > y.shape[2]:
            y = y.transpose(1, 2)
        y = y[:, -1, :, :]
    elif y.dim() == 3:
        if y.shape[1] > y.shape[2]:
            y = y.transpose(1, 2)
        y = y[:, -1, :].unsqueeze(-1)

    return x, u, y


def forward_pass(model, x, u, ei, model_name):
    """Execute forward pass conforming exactly to TSL signatures."""
    if model_name == 'STID':
        # STID has no dynamic/static exog parameters in this TSL version
        out = model(x)
    elif model_name in ('DCRNN', 'GRUGCNModel', 'GraphWaveNet'):
        # Pass exogenous features using the explicit parameter name 'u'
        out = model(x, edge_index=ei, u=u)
    else:
        out = model(x, edge_index=ei)
    
    # Standardize output to match target shape: [Batch, Nodes, 1]
    if out.dim() == 4:
        out = out[:, 0, :, :]  # Isolate horizon step 0 -> [B, N, F]
    return out


def train_one_epoch(model, loader, optimizer, criterion,
                    edge_index, model_name):
    """Run one training epoch, return mean loss."""
    model.train()
    total_loss = 0.0

    for x, u, y in loader:
        # Align dimensions to: [Batch, Time, Nodes, Features]
        x, u, y = prepare_tensors(x, u, y)

        x = x.to(DEVICE)
        u = u.to(DEVICE)
        y = y.to(DEVICE)
        ei = edge_index.to(DEVICE)

        optimizer.zero_grad()
        out = forward_pass(model, x, u, ei, model_name)

        loss = criterion(out, y)
        loss.backward()

        # Gradient clipping — important for RNN stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, edge_index, model_name):
    """Evaluate model on loader, return loss, R², RMSE."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_true   = []

    for x, u, y in loader:
        # Align dimensions to: [Batch, Time, Nodes, Features]
        x, u, y = prepare_tensors(x, u, y)

        x  = x.to(DEVICE)
        u  = u.to(DEVICE)
        y  = y.to(DEVICE)
        ei = edge_index.to(DEVICE)

        out = forward_pass(model, x, u, ei, model_name)
        loss = criterion(out, y)
        total_loss += loss.item()

        all_preds.append(out.cpu().numpy().flatten())
        all_true.append(y.cpu().numpy().flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(loader)

    return metrics


# ------------------------------------------------------------------
# Main training orchestrator
# ------------------------------------------------------------------

def train_model(model_name, train_dataset, test_dataset,
                edge_index, config):
    """Full training loop for a single model."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    n_nodes   = train_dataset.n_nodes
    n_dynamic = train_dataset.n_dynamic_features
    n_static  = train_dataset.n_static_features

    # --- Build model ---
    model = get_model(
        name        = model_name,
        n_nodes     = n_nodes,
        n_dynamic   = n_dynamic,
        n_static    = n_static,
        window_size = WINDOW_SIZE,
    ).to(DEVICE)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,   # must preserve temporal order
        num_workers = 4,
        pin_memory  = (DEVICE.type == 'cuda'),  # only pin memory when GPU is present
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = (DEVICE.type == 'cuda'),
    )

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    # --- Training loop with early stopping ---
    best_r2       = -np.inf
    best_metrics  = {}
    patience_ctr  = 0
    history       = []

    checkpoint_path = os.path.join(
        config['paths']['processed_dir'],
        f"checkpoint_{model_name}.pt"
    )

    for epoch in range(1, N_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, edge_index, model_name
        )
        test_metrics = evaluate(
            model, test_loader, criterion,
            edge_index, model_name
        )
        elapsed = time.time() - t0

        scheduler.step(test_metrics['loss'])

        history.append({
            'epoch'     : epoch,
            'train_loss': train_loss,
            'test_loss' : test_metrics['loss'],
            'test_r2'   : test_metrics['R2_Score'],
            'test_rmse' : test_metrics['RMSE'],
        })

        print(
            f"  Epoch {epoch:03d}/{N_EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Test Loss: {test_metrics['loss']:.5f} | "
            f"R²: {test_metrics['R2_Score']:.4f} | "
            f"RMSE: {test_metrics['RMSE']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- Early stopping ---
        if test_metrics['R2_Score'] > best_r2:
            best_r2      = test_metrics['R2_Score']
            best_metrics = {
                'R2_Score': test_metrics['R2_Score'],
                'RMSE'    : test_metrics['RMSE'],
            }
            patience_ctr = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    ✓ New best R²={best_r2:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # Save training history
    history_path = os.path.join(
        config['paths']['processed_dir'],
        f"history_{model_name}.csv"
    )
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"  Training history saved to {history_path}")
    print(f"  Best Test R²={best_metrics['R2_Score']:.4f} | "
          f"RMSE={best_metrics['RMSE']:.4f}")

    return best_metrics


def main():
    # --- Load config ---
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['paths']['processed_dir'], exist_ok=True)

    print(f"Device: {DEVICE}")

    # --- Build edge index ---
    cache_path = os.path.join(
        config['paths']['processed_dir'], "edge_index.pt"
    )
    edge_index = get_edge_index(
        height     = config['spatial']['height'],
        width      = config['spatial']['width'],
        cache_path = cache_path,
    )
    print(f"Edge index: {edge_index.shape}")

    # --- Build datasets ---
    print("\nBuilding train and test datasets...")
    train_dataset, test_dataset = build_datasets(config, window_size=WINDOW_SIZE)

    # --- Train all models sequentially ---
    all_results = {}

    for model_name in MODEL_REGISTRY.keys():
        try:
            metrics = train_model(
                model_name    = model_name,
                train_dataset = train_dataset,
                test_dataset  = test_dataset,
                edge_index    = edge_index,
                config        = config,
            )
            all_results[model_name] = metrics
        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            all_results[model_name] = {'R2_Score': None, 'RMSE': None}
            import traceback
            traceback.print_exc()
            continue

    # --- Final comparison table ---
    print(f"\n{'='*60}")
    print("  DEEP LEARNING MODEL COMPARISON")
    print(f"{'='*60}")
    results_df = pd.DataFrame(all_results).T
    print(results_df.to_string())

    results_path = os.path.join(
        config['paths']['processed_dir'], "dl_metrics.csv"
    )
    results_df.to_csv(results_path)


if __name__ == "__main__":
    main()