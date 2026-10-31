"""
Utils for LOEO Grid Search - Reward Function Tuning
"""
from __future__ import annotations
import json, os, shutil, tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from astral import LocationInfo
from astral.sun import sun

from preprocess_hotspots import compute_hotspots, apply_hotspots
from reward import compute_reward_vectorized


# =============================================================================
# TIỆN ÍCH
# =============================================================================


def _zscore(series: pd.Series) -> pd.Series:
    """Chuẩn hóa z-score"""
    return (series - series.mean()) / (series.std(ddof=0) + 1e-9)


def _corr_rank(s1: pd.Series, s2: pd.Series, method="kendall") -> float:
    """Kendall correlation, trả về 0 nếu không đủ dữ liệu"""
    return float(s1.corr(s2, method=method)) if len(s1) >= 2 else 0.0


# =============================================================================
# TÍNH NGÀY/ĐÊM THEO THIÊN VĂN
# =============================================================================

def compute_is_night_astral(df: pd.DataFrame) -> pd.Series:
    """
    Tính is_night dựa trên sunrise/sunset thực tế (astral).
    Returns: 1 nếu đêm, 0 nếu ngày
    """
    timestamps = pd.to_datetime(df["timestamp"])
    
    # Location từ trung tâm dữ liệu
    location = LocationInfo(
        "Kruger", "South Africa", "Africa/Johannesburg",
        df["lat"].median(), df["lon"].median()
    )
    
    # Cache sunrise/sunset theo ngày
    sun_cache = {}
    
    def is_night_for_ts(ts):
        date_key = ts.date()
        if date_key not in sun_cache:
            try:
                s = sun(location.observer, date=date_key)
                sun_cache[date_key] = (
                    s["sunrise"].replace(tzinfo=None),
                    s["sunset"].replace(tzinfo=None)
                )
            except Exception:
                sun_cache[date_key] = None
        
        cached = sun_cache[date_key]
        if cached is None:
            return 1 if (ts.hour >= 18 or ts.hour < 6) else 0
        
        sunrise, sunset = cached
        ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
        return 1 if (ts_naive < sunrise or ts_naive > sunset) else 0
    
    return timestamps.apply(is_night_for_ts)


# =============================================================================
# TIỀN XỬ LÝ FEATURES
# =============================================================================

def precompute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các features thô (chạy 1 lần trước LOEO loop).
    - is_night: Từ astral
    - velocity_raw, turn_score, turning_entropy
    """
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["hour"] = d["timestamp"].dt.hour
    
    # 1. Ngày/Đêm theo thiên văn
    print("Computing astronomical day/night...")
    d["is_night"] = compute_is_night_astral(d)
    
    # 2. Velocity
    time_diff = (d.groupby("episode")["timestamp"].shift(-1) - d["timestamp"]).dt.total_seconds() / 3600
    d["velocity_raw"] = (d["step"] / time_diff).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Turn score = (angle/180)^2
    dx, dy = d["x_next"] - d["x"], d["y_next"] - d["y"]
    dx_prev = dx.groupby(d["episode"]).shift(1).fillna(0)
    dy_prev = dy.groupby(d["episode"]).shift(1).fillna(0)
    
    dot = dx * dx_prev + dy * dy_prev
    denom = np.sqrt(dx**2 + dy**2) * np.sqrt(dx_prev**2 + dy_prev**2)
    cos_sim = np.clip(np.where(denom > 1e-6, dot / denom, 1.0), -1, 1)
    d["turn_score"] = (np.degrees(np.arccos(cos_sim)) / 180) ** 2
    d["turn_score"] = d["turn_score"].fillna(0)
    
    # 4. Turning entropy (rolling window)
    d["bearing"] = np.degrees(np.arctan2(dy, dx)) % 360
    d["turning_angle"] = d.groupby("episode")["bearing"].diff().abs().fillna(0)
    
    def _entropy_18bins(x):
        counts, _ = np.histogram(x, bins=18, range=(0, 360))
        return entropy(counts) if counts.sum() > 0 else 0.0
    
    max_entropy = np.log(18)  # ≈ 2.89, entropy tối đa của 18 bins
    d["turning_entropy"] = d.groupby("episode")["turning_angle"].transform(
        lambda x: x.rolling(10, min_periods=5).apply(_entropy_18bins)
    ).fillna(0) / max_entropy  # Chuẩn hóa về [0, 1]
    
    return d


def compute_training_scales(train_df: pd.DataFrame) -> Tuple[float, float, float]:
    """Tính scales từ tập TRAIN: (vel_scale, temp_mu, temp_sigma)"""
    return (
        train_df["velocity_raw"].quantile(0.95) + 1e-9,
        train_df["temp"].mean(),
        train_df["temp"].std() + 1e-9
    )


def normalize_features(df: pd.DataFrame, vel_scale: float, temp_mu: float, temp_sigma: float) -> pd.DataFrame:
    """Chuẩn hóa vel và temp theo scales từ TRAIN"""
    d = df.copy()
    d["vel_norm"] = (d["velocity_raw"] / vel_scale).clip(0, 1)
    d["temp_std"] = (d["temp"] - temp_mu) / temp_sigma
    return d


# =============================================================================
# TÍNH REWARD (import từ reward.py)
# =============================================================================

def _compute_reward(df, R_hot, W_vel, W_temp, W_smooth, W_entropy, p=1.5, beta=1.0, tau=0.0):
    """Wrapper: Gọi compute_reward_vectorized từ reward.py"""
    return compute_reward_vectorized(df, R_hot, W_vel, W_temp, W_smooth, W_entropy, p, beta, tau)


def _episode_metrics(df, R_hot, W_vel, W_temp, W_smooth, W_entropy, p=1.5, beta=1.0, tau=0.0) -> pd.DataFrame:
    """Tính metrics trung bình theo episode"""
    reward = _compute_reward(df, R_hot, W_vel, W_temp, W_smooth, W_entropy, p, beta, tau)
    
    aggs = {
        "n_steps": ("reward", "size"),
        "avg_reward_per_step": ("reward", "mean"),
        "hot_rate": ("is_hotspot", "mean"),
        "mean_vel": ("vel_norm", "mean"),
        "mean_temp": ("temp_std", "mean"),
    }
    if "hot_score" in df.columns:
        aggs["hot_intensity"] = ("hot_score", "mean")
    if "turn_score" in df.columns:
        aggs["mean_turn"] = ("turn_score", "mean")
    if "turning_entropy" in df.columns:
        aggs["mean_entropy"] = ("turning_entropy", "mean")
    
    return df.assign(reward=reward).groupby("episode").agg(**aggs).reset_index()


# =============================================================================
# ECO SCORE & PAIRWISE ACCURACY
# =============================================================================

def _eco_score(m: pd.DataFrame, bad_mode: bool = False) -> pd.Series:
    """
    Tính Eco Score.
    - bad_mode = False: Giả định CHUẨN
    - bad_mode = True:  Giả định SAI
    """
    hot_col = "hot_intensity" if "hot_intensity" in m.columns else "hot_rate"
    
    # Lấy Z-score
    z_hot = 1.5*_zscore(m[hot_col])
    z_vel = _zscore(m["mean_vel"])
    z_temp = _zscore(m["mean_temp"])
    
    if bad_mode:
        eco = z_hot + z_vel + z_temp 
    else:
        eco = z_hot - z_vel - z_temp

    # Các chỉ số phụ (Turn, Entropy) luôn trừ vì tốn năng lượng
    if "mean_turn" in m.columns:
        eco -= _zscore(m["mean_turn"])
    if "mean_entropy" in m.columns:
        eco -= _zscore(m["mean_entropy"])
        
    return eco

def _held_pairwise_acc(m_train: pd.DataFrame, m_held: pd.DataFrame, bad_mode: bool = False) -> Tuple[float, int]:
    if m_held.shape[0] != 1 or m_train.shape[0] == 0:
        return float("nan"), 0
    
    hot_col = "hot_intensity" if "hot_intensity" in m_train.columns else "hot_rate"
    eps = 1e-9
    
    # Tính Eco Train đúng mode
    eco_train = _eco_score(m_train, bad_mode=bad_mode)
    
    # Tính Eco Held đúng mode
    def z_held_val(col):
        return (float(m_held[col].iloc[0]) - m_train[col].mean()) / (m_train[col].std(ddof=0) + eps)
    
    zh_hot = 1.5 * z_held_val(hot_col)
    zh_vel = z_held_val("mean_vel")
    zh_temp = z_held_val("mean_temp")
    
    if bad_mode:
        eco_held_val = zh_hot + zh_vel + zh_temp
    else:
        eco_held_val = zh_hot - zh_vel - zh_temp

    if "mean_turn" in m_train.columns and "mean_turn" in m_held.columns:
        eco_held_val -= z_held_val("mean_turn")
    if "mean_entropy" in m_train.columns and "mean_entropy" in m_held.columns:
        eco_held_val -= z_held_val("mean_entropy")
    
    # So sánh
    r_train = m_train["avg_reward_per_step"].values
    r_held = float(m_held["avg_reward_per_step"].iloc[0])
    
    y = np.sign(eco_held_val - eco_train)
    p = np.sign(r_held - r_train)
    
    mask = (y != 0)
    n_pairs = int(mask.sum())
    if n_pairs == 0:
        return float("nan"), 0
    
    return float(np.mean(p[mask] == y[mask])), n_pairs


# =============================================================================
# LOEO GRID SEARCH
# =============================================================================

def loeo_grid_search(
    df_raw: pd.DataFrame,
    W_vel_grid: List[float],
    W_temp_grid: List[float],
    W_smooth_grid: List[float] = [0.0],
    W_entropy_grid: List[float] = [0.0],
    R_HOT: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
    p: float = 1.5,
    beta: float = 5.0,
    tau: float = 1.0,
    grid_size: int = 100,
    hot_quantile: float = 0.8,
    use_bad_eco: bool = False, # <--- THÊM THAM SỐ NÀY
) -> pd.DataFrame:
    
    np.random.seed(seed)
    records = []
    episodes = list(df_raw["episode"].unique())
    tmp_dir = tempfile.mkdtemp(prefix="loeo_")
    
    # Pre-compute features (Chỉ chạy 1 lần, không phụ thuộc mode)
    if verbose:
        print("Pre-computing features...")
    df_pre = precompute_raw_features(df_raw)
    
    try:
        for held_ep in episodes:
            if verbose:
                print(f"--- LOEO fold: held-out = {held_ep} ---")
            train_raw = df_pre[df_pre["episode"] != held_ep].copy()
            held_raw = df_pre[df_pre["episode"] == held_ep].copy()
            
            model_path = os.path.join(tmp_dir, f"hotspot_{held_ep}.json")
            train_hot, _ = compute_hotspots(train_raw, grid_size, hot_quantile, model_path, save_heatmap=False)
            held_hot = apply_hotspots(held_raw, model_path)
            
            vel_scale, temp_mu, temp_sigma = compute_training_scales(train_hot)
            train_df = normalize_features(train_hot, vel_scale, temp_mu, temp_sigma)
            held_df = normalize_features(held_hot, vel_scale, temp_mu, temp_sigma)

            # Grid search loop
            for W_vel in W_vel_grid:
                for W_temp in W_temp_grid:
                    for W_smooth in W_smooth_grid:
                        for W_entropy in W_entropy_grid:
                            m_train = _episode_metrics(train_df, R_HOT, W_vel, W_temp, W_smooth, W_entropy, p, beta, tau)
                            m_held = _episode_metrics(held_df, R_HOT, W_vel, W_temp, W_smooth, W_entropy, p, beta, tau)
                            
                            # --- SỬ DỤNG use_bad_eco ---
                            eco_train = _eco_score(m_train, bad_mode=use_bad_eco)
                            
                            tau_hot = _corr_rank(m_train["avg_reward_per_step"], m_train["hot_rate"])
                            tau_eco = _corr_rank(m_train["avg_reward_per_step"], eco_train)
                            held_acc, n_pairs = _held_pairwise_acc(m_train, m_held, bad_mode=use_bad_eco)
                            
                            records.append({
                                "held": str(held_ep),
                                "W_vel": W_vel, "W_temp": W_temp,
                                "W_smooth": W_smooth, "W_entropy": W_entropy,
                                "held_acc": held_acc, "tau_hot_train": tau_hot,
                                "tau_eco_train": tau_eco, "n_pairs": n_pairs
                            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return pd.DataFrame(records)


# =============================================================================
# CHỌN PARAMS TỐI ƯU
# =============================================================================

def select_final_params_loeo(
    res_df: pd.DataFrame,
    verbose: bool = True,
    alpha_hot: float = 0.4,
    eps_score: float = 1e-4,
) -> Dict:
    """Chọn params tốt nhất: score cao → L1 thấp → std thấp"""
    
    if res_df.empty:
        raise ValueError("res_df rỗng!")
    
    # Group cols
    group_cols = ["W_vel", "W_temp"]
    if "W_smooth" in res_df.columns:
        group_cols.append("W_smooth")
    if "W_entropy" in res_df.columns:
        group_cols.append("W_entropy")
    
    g = res_df.groupby(group_cols, as_index=False).agg(
        median_held_acc=("held_acc", "median"),
        std_held_acc=("held_acc", "std"),
        median_tau_hot=("tau_hot_train", "median"),
        median_tau_eco=("tau_eco_train", "median"),
        n_folds=("held", "nunique"),
    ).fillna({"std_held_acc": 0})
    
    # L1 norm & Score
    g["l1_norm"] = sum(g[c].abs() for c in group_cols)
    g["score"] = g["median_held_acc"] + alpha_hot * g["median_tau_hot"]
    
    if verbose:
        show = g.sort_values(["score", "l1_norm"], ascending=[False, True])
        print("=" * 90)
        print("LOEO GRID SEARCH RESULTS (Top 10)")
        print("=" * 90)
        print(show[group_cols + ["median_held_acc", "std_held_acc", "median_tau_hot", "l1_norm", "score"]].head(10).to_string(index=False))
        print(f"... (Total: {len(show)} parameter combinations)")
        print()
    
    # Chọn best từ plateau
    best_score = g["score"].max()
    plateau = g[g["score"] >= best_score - eps_score].sort_values(
        ["l1_norm", "std_held_acc", "median_tau_hot"],
        ascending=[True, True, False]
    )
    best = plateau.iloc[0]
    
    if verbose:
        print(f"[Best] W_vel={best['W_vel']:.3f}, W_temp={best['W_temp']:.3f}, "
              f"W_smooth={best.get('W_smooth', 0):.3f}, W_entropy={best.get('W_entropy', 0):.3f}, "
              f"score={best['score']:.4f}, held_acc={best['median_held_acc']:.3f}, L1={best['l1_norm']:.2f}")
    
    result = {
        "W_vel": float(best["W_vel"]),
        "W_temp": float(best["W_temp"]),
        "score": float(best["score"]),
        "median_held_acc": float(best["median_held_acc"]),
        "median_tau_hot": float(best["median_tau_hot"]),
        "l1_norm": float(best["l1_norm"]),
        "n_folds": int(best["n_folds"]),
    }
    if "W_smooth" in best:
        result["W_smooth"] = float(best["W_smooth"])
    if "W_entropy" in best:
        result["W_entropy"] = float(best["W_entropy"])
    
    return result


def save_final_params(best_params: Dict, output_path: str):
    """Lưu params ra JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved: {output_path}")