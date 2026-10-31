import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import json
from typing import Dict, Tuple, Set, List, Optional

def compute_hotspots(
    df: pd.DataFrame,
    grid_size: int = 100,
    hot_quantile: float = 0.8,
    model_path: str = "dataset_preprocessed/hotspot_model.json",
    heatmap_path: str = "dataset_preprocessed/visit_count_heatmap.png",
    save_heatmap: bool = True,
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, Dict]:

    if "lat" not in df.columns or "lon" not in df.columns:
        raise KeyError("compute_hotspots requires columns: 'lat' and 'lon'")

    d = df.copy()

    # ---- grid definition (from TRAIN only) ----
    lat_min = float(d["lat"].min())
    lat_max = float(d["lat"].max()) + eps
    lon_min = float(d["lon"].min())
    lon_max = float(d["lon"].max()) + eps

    lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)

    d["lat_bin"] = (np.digitize(d["lat"].to_numpy(), lat_bins) - 1).clip(0, grid_size - 1)
    d["lon_bin"] = (np.digitize(d["lon"].to_numpy(), lon_bins) - 1).clip(0, grid_size - 1)

    # ---- visit counts (only visited cells) ----
    counts = d.groupby(["lat_bin", "lon_bin"]).size()

    # ---- hotspot intensity (Choice A: log-normalized) ----
    hot_score_grid = np.zeros((grid_size, grid_size), dtype=float)
    if len(counts) > 0:
        c_max = float(counts.max())
        denom = np.log1p(c_max) + eps
        for (r, c), v in counts.items():
            hot_score_grid[int(r), int(c)] = np.log1p(float(v)) / denom

    # gán hot_score cho từng step (vectorized)
    rb = d["lat_bin"].to_numpy(dtype=int)
    cb = d["lon_bin"].to_numpy(dtype=int)
    d["hot_score"] = hot_score_grid[rb, cb]

    # ---- determine hotspots (quantile over visited cells) ----
    if len(counts) == 0:
        threshold = float("inf")
        hot_cells: Set[Tuple[int, int]] = set()
    else:
        threshold = float(counts.quantile(hot_quantile))
        hot_cells = set(
            (int(r), int(c)) for (r, c), v in counts.items() if float(v) >= threshold
        )

    d["is_hotspot"] = [
        int((int(r), int(c)) in hot_cells)
        for r, c in zip(d["lat_bin"], d["lon_bin"])
    ]

    # ---- save hotspot model ----
    model = {
        "grid_size": int(grid_size),
        "hot_quantile": float(hot_quantile),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_bins": lat_bins.tolist(),
        "lon_bins": lon_bins.tolist(),
        "threshold": threshold,
        "hot_cells": [[int(r), int(c)] for (r, c) in sorted(hot_cells)],
        # NEW
        "hot_score_type": "log1p_max",
        "hot_score_grid": hot_score_grid.tolist(),
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    # ---- save heatmap (TRAIN only) ----
    if save_heatmap:
        os.makedirs(os.path.dirname(heatmap_path) or ".", exist_ok=True)
        save_heatmap_from_counts(
            counts=counts,
            grid_size=grid_size,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            out_path=heatmap_path,
            title="Train Visit Count Heatmap",
        )
    return d, model

def apply_hotspots(
    df: pd.DataFrame,
    model_path: str = "dataset_preprocessed/hotspot_model.json",
    out_of_range: str = "mark_cold",  # "mark_cold" | "clip" | "drop"
) -> pd.DataFrame:

    if "lat" not in df.columns or "lon" not in df.columns:
        raise KeyError("apply_hotspots requires columns: 'lat' and 'lon'")

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    grid_size = int(model["grid_size"])
    lat_bins = np.array(model["lat_bins"], dtype=float)
    lon_bins = np.array(model["lon_bins"], dtype=float)
    hot_cells = set((int(r), int(c)) for r, c in model["hot_cells"])

    lat_min, lat_max = float(model["lat_min"]), float(model["lat_max"])
    lon_min, lon_max = float(model["lon_min"]), float(model["lon_max"])

    # NEW: load hot_score_grid
    if "hot_score_grid" in model:
        hot_score_grid = np.array(model["hot_score_grid"], dtype=float)
        if hot_score_grid.shape != (grid_size, grid_size):
            hot_score_grid = np.zeros((grid_size, grid_size), dtype=float)
    else:
        hot_score_grid = np.zeros((grid_size, grid_size), dtype=float)

    d = df.copy()
    lat = d["lat"].to_numpy()
    lon = d["lon"].to_numpy()

    in_range = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

    if out_of_range == "drop":
        d = d.loc[in_range].copy()
        lat = d["lat"].to_numpy()
        lon = d["lon"].to_numpy()
        in_range = np.ones(len(d), dtype=bool)

    lat_bin = (np.digitize(lat, lat_bins) - 1)
    lon_bin = (np.digitize(lon, lon_bins) - 1)

    if out_of_range == "clip":
        lat_bin = lat_bin.clip(0, grid_size - 1).astype(int)
        lon_bin = lon_bin.clip(0, grid_size - 1).astype(int)

        d["lat_bin"] = lat_bin
        d["lon_bin"] = lon_bin

        d["is_hotspot"] = [
            int((int(r), int(c)) in hot_cells) for r, c in zip(d["lat_bin"], d["lon_bin"])
        ]
        d["hot_score"] = hot_score_grid[lat_bin, lon_bin]

    elif out_of_range == "mark_cold":
        lat_bin_clip = lat_bin.clip(0, grid_size - 1).astype(int)
        lon_bin_clip = lon_bin.clip(0, grid_size - 1).astype(int)

        d["lat_bin"] = np.where(in_range, lat_bin_clip, -1)
        d["lon_bin"] = np.where(in_range, lon_bin_clip, -1)

        d["is_hotspot"] = [
            int((int(r), int(c)) in hot_cells) if (int(r) >= 0 and int(c) >= 0) else 0
            for r, c in zip(d["lat_bin"], d["lon_bin"])
        ]

        hs = hot_score_grid[lat_bin_clip, lon_bin_clip]
        d["hot_score"] = np.where(in_range, hs, 0.0)

    else:
        raise ValueError("out_of_range must be one of: mark_cold | clip | drop")

    return d

def save_heatmap_from_counts(
    counts: pd.Series,
    grid_size: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    out_path: str,
    title: str = "Visit Count Heatmap (Train)",
):
    """
    counts: Series with MultiIndex (lat_bin, lon_bin) -> visit_count
    """

    grid = (
        counts.unstack(fill_value=0)
        .reindex(index=range(grid_size), columns=range(grid_size), fill_value=0)
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(
        grid.values,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],  # x=lon, y=lat
        aspect="equal",
    )
    plt.colorbar(label="Visit count")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()