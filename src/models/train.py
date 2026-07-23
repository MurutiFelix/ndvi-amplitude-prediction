# src/models/train.py
"""
Deep Learning training loop for spatiotemporal NDVI prediction.

Trains TSL graph models sequentially with AMP acceleration using a strict
three-way (Train, Val, Test) split to eliminate test-set data leakage.

Usage:
    python -m src.models.train
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

from src.data.dataset import build_datasets
from src.models.spatio_temporal import get_model, MODEL_REGISTRY
from src.utils.spatial import get_edge_index

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    """Ensures input tensors match TSL format: [Batch, Time, Nodes, Features]"""
    if x.dim() == 4 and x.shape[1] > x.shape[2]:
        x = x.transpose(1, 2)

    if u is not None:
        if u.dim() == 3:
            u = u.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        elif u.dim() == 4 and u.shape[1] > u.shape[2]:
            u = u.transpose(1, 2)

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
        out = model(x)
    elif model_name in ('DCRNN', 'GRUGCNModel', 'GraphWaveNet'):
        out = model(x, edge_index=ei, u=u)
    else:
        out = model(x, edge_index=ei)

    if out.dim() == 4:
        out = out[:, 0, :, :]
    return out


def train_one_epoch(model, loader, optimizer, criterion,
                    edge_index, model_name, scaler):
    """Run one training epoch using Mixed Precision, return mean loss."""
    model.train()
    total_loss = 0.0

    for x, u, y in loader:
        x, u, y = prepare_tensors(x, u, y)

        x  = x.to(DEVICE, non_blocking=True)
        u  = u.to(DEVICE, non_blocking=True)
        y  = y.to(DEVICE, non_blocking=True)
        ei = edge_index.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            out = forward_pass(model, x, u, ei, model_name)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, edge_index, model_name):
    """Evaluate model on loader, return metrics and raw predictions."""
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    for x, u, y in loader:
        x, u, y = prepare_tensors(x, u, y)

        x  = x.to(DEVICE, non_blocking=True)
        u  = u.to(DEVICE, non_blocking=True)
        y  = y.to(DEVICE, non_blocking=True)
        ei = edge_index.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            out = forward_pass(model, x, u, ei, model_name)
            loss = criterion(out, y)

        total_loss += loss.item()
        all_preds.append(out.cpu().numpy().flatten())
        all_true.append(y.cpu().numpy().flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = total_loss / len(loader)
    metrics['preds'] = y_pred

    return metrics


# ------------------------------------------------------------------
# Main training orchestrator
# ------------------------------------------------------------------

def train_model(model_name, train_dataset, val_dataset, test_dataset, edge_index, config):
    """Full training loop optimizing on Val Loss and evaluating finally on Test."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    window_size   = config['features']['window_size']
    n_epochs      = config['features']['n_epochs']
    learning_rate = config['features']['learning_rate']
    patience      = config['features']['patience']
    weight_decay  = config['features'].get('weight_decay', 1e-4)

    model = get_model(
        name        = model_name,
        n_nodes     = train_dataset.n_nodes,
        n_dynamic   = train_dataset.n_dynamic_features,
        n_static    = train_dataset.n_static_features,
        window_size = window_size,
    ).to(DEVICE)

    # Pin memory config helper dict
    loader_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': (DEVICE.type == 'cuda')
    }

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader   = DataLoader(val_dataset, **loader_kwargs)
    test_loader  = DataLoader(test_dataset, **loader_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler    = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    
    # Track Plateau steps using Validation Loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    patience_ctr  = 0
    min_delta     = 1e-5
    history       = []

    checkpoint_path = os.path.join(
        config['paths']['processed_dir'],
        f"checkpoint_{model_name}.pt"
    )

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        
        # 1. Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, edge_index, model_name, scaler
        )
        
        # 2. Validation Step (Used exclusively for optimization steering)
        val_metrics = evaluate(
            model, val_loader, criterion,
            edge_index, model_name
        )
        elapsed = time.time() - t0

        scheduler.step(val_metrics['loss'])

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_r2': val_metrics['R2_Score'],
            'val_rmse': val_metrics['RMSE'],
        })

        print(
            f"  Epoch {epoch:03d}/{n_epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Val R²: {val_metrics['R2_Score']:.4f} | "
            f"Val RMSE: {val_metrics['RMSE']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early Stopping check strictly using Validation metrics
        if val_metrics['loss'] < (best_val_loss - min_delta):
            best_val_loss = val_metrics['loss']
            patience_ctr = 0

            torch.save({
                'state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_r2': val_metrics['R2_Score'],
                'val_rmse': val_metrics['RMSE']
            }, checkpoint_path)

            print(f"    ✓ New best Val Loss={best_val_loss:.5f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} (no structural val improvement for {patience} epochs)")
                break

    # Save tracking history meta-logs
    history_path = os.path.join(config['paths']['processed_dir'], f"history_{model_name}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"  Training history saved to {history_path}")

    # ==================================================================
    # 3. Final Evaluation Step (Completely Out-of-Sample)
    # ==================================================================
    print(f"  Running final evaluation pass on clean Test Set...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, edge_index, model_name)
    
    # Save test metrics and predictions back into the checkpoint archive cleanly
    checkpoint['test_preds'] = test_metrics['preds']
    checkpoint['test_loss']  = test_metrics['loss']
    checkpoint['test_r2']    = test_metrics['R2_Score']
    checkpoint['test_rmse']  = test_metrics['RMSE']
    torch.save(checkpoint, checkpoint_path)

    print(f"  Best Val R²={checkpoint['val_r2']:.4f} | RMSE={checkpoint['val_rmse']:.4f}")
    print(f"  --> Final Unbiased Test R²={test_metrics['R2_Score']:.4f} | RMSE={test_metrics['RMSE']:.4f}")

    return {
        'Val_R2': checkpoint['val_r2'],
        'Val_RMSE': checkpoint['val_rmse'],
        'Test_R2': test_metrics['R2_Score'],
        'Test_RMSE': test_metrics['RMSE']
    }


def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config['paths']['processed_dir'], exist_ok=True)
    print(f"Device: {DEVICE}")

    cache_path = os.path.join(config['paths']['processed_dir'], "edge_index.pt")
    edge_index = get_edge_index(
        height     = config['spatial']['height'],
        width      = config['spatial']['width'],
        cache_path = cache_path,
    )
    print(f"Edge index: {edge_index.shape}")

    print("\nBuilding train, validation, and test datasets...")
    # build_datasets now handles three targets using the config
    train_dataset, val_dataset, test_dataset = build_datasets(
        config, 
        window_size=config['features']['window_size']
    )

    all_results = {}

    for model_name in MODEL_REGISTRY.keys():
        try:
            metrics = train_model(
                model_name    = model_name,
                train_dataset = train_dataset,
                val_dataset   = val_dataset,
                test_dataset  = test_dataset,
                edge_index    = edge_index,
                config        = config,
            )
            all_results[model_name] = metrics
        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            all_results[model_name] = {'Val_R2': None, 'Val_RMSE': None, 'Test_R2': None, 'Test_RMSE': None}
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("  DEEP LEARNING MODEL COMPARISON")
    print(f"{'='*60}")
    results_df = pd.DataFrame(all_results).T
    print(results_df.to_string())

    results_path = os.path.join(config['paths']['processed_dir'], "dl_metrics.csv")
    results_df.to_csv(results_path)


if __name__ == "__main__":
    main()