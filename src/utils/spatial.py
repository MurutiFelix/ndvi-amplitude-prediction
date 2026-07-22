# src/utils/spatial.py
import numpy as np
import torch


def build_grid_edge_index(height: int, width: int) -> torch.Tensor:
    """
    Builds a sparse edge index for an (height x width) pixel grid
    using 8-connectivity (queen contiguity) — each pixel is connected
    to all valid immediate neighbours including diagonals.

    This defines the spatial graph structure for TSL models:
        - Nodes  : height * width pixels (28,779 for 159x181)
        - Edges  : ~8 edges per interior node, fewer at borders
        - Format : [2, num_edges] torch.long tensor (COO format)

    Args:
        height (int): Number of rows in the raster grid.
        width  (int): Number of columns in the raster grid.

    Returns:
        edge_index (torch.Tensor): Shape [2, E] where E is total edges.
    """
    row_offsets = [-1, -1, -1,  0,  0,  1,  1,  1]
    col_offsets = [-1,  0,  1, -1,  1, -1,  0,  1]

    src_list = []
    dst_list = []

    for r in range(height):
        for c in range(width):
            node_id = r * width + c
            for dr, dc in zip(row_offsets, col_offsets):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbour_id = nr * width + nc
                    src_list.append(node_id)
                    dst_list.append(neighbour_id)

    edge_index = torch.tensor(
        [src_list, dst_list],
        dtype=torch.long
    )

    return edge_index


def build_grid_edge_index_fast(height: int, width: int) -> torch.Tensor:
    """
    Vectorized version of build_grid_edge_index — much faster for large grids.
    Produces identical output to build_grid_edge_index but without Python loops.

    Args:
        height (int): Number of rows in the raster grid.
        width  (int): Number of columns in the raster grid.

    Returns:
        edge_index (torch.Tensor): Shape [2, E] where E is total edges.
    """
    # Node indices for all pixels
    rows = np.arange(height)
    cols = np.arange(width)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')  # [H, W]
    node_ids = rr * width + cc                        # [H, W]

    # 8 directional offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, -1),          ( 0, 1),
               ( 1, -1), ( 1, 0), ( 1, 1)]

    src_list = []
    dst_list = []

    for dr, dc in offsets:
        # Shift grid by offset
        nr = rr + dr
        nc = cc + dc

        # Valid mask — stays within grid bounds
        valid = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)

        src = node_ids[valid]
        dst = (nr[valid]) * width + (nc[valid])

        src_list.append(src.flatten())
        dst_list.append(dst.flatten())

    src_all = np.concatenate(src_list)
    dst_all = np.concatenate(dst_list)

    edge_index = torch.tensor(
        np.stack([src_all, dst_all], axis=0),
        dtype=torch.long
    )

    return edge_index


def get_edge_index(height: int, width: int, cache_path: str = None) -> torch.Tensor:
    """
    Returns the edge index for the grid, loading from cache if available.
    Saves to cache after first computation to avoid rebuilding on every run.

    Args:
        height     (int)          : Grid height.
        width      (int)          : Grid width.
        cache_path (str, optional): Path to save/load cached edge index tensor.

    Returns:
        edge_index (torch.Tensor): Shape [2, E].
    """
    import os

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached edge index from {cache_path}...")
        return torch.load(cache_path, weights_only=True)

    print(f"Building {height}×{width} grid edge index ({height * width:,} nodes)...")
    edge_index = build_grid_edge_index_fast(height, width)
    print(f"Edge index built: {edge_index.shape[1]:,} edges")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(edge_index, cache_path)
        print(f"Edge index cached to {cache_path}")

    return edge_index