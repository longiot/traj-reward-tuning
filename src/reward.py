import numpy as np

def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# ==================== NONLINEAR REWARD ====================
def make_reward_fn_nonlinear(R_hot, W_vel, W_temp, W_smooth, W_entropy, p=1.5, beta=1.0, tau=0.0):
    def reward_fn(row):
        # Lấy dữ liệu
        is_night = float(row.get("is_night", 0))
        hot = float(row.get("hot_score", 0)) if "hot_score" in row else float(row.get("is_hotspot", 0))
        vel_norm = float(row.get("vel_norm", 0))
        
        # Hình phạt vận tốc
        w_vel_adaptive = W_vel * (1.0 + 0.5 * is_night)
        movement_penalty = w_vel_adaptive * (vel_norm ** p)
        
        # Hình phạt nhiệt độ
        temp_std = float(row.get("temp_std", 0))
        temp_penalty = softplus(beta * (temp_std - tau)) / beta
        
        # Hình phạt góc quay (tức thời)
        turn_val = float(row.get("turn_score", 0))
        
        # Hình phạt hỗn loạn (quá trình)
        entropy_val = float(row.get("turning_entropy", 0))
        
        return (
            R_hot * hot
            - movement_penalty 
            - W_temp * temp_penalty
            - W_smooth * turn_val
            - W_entropy * entropy_val
        )
    return reward_fn


# ==================== VECTORIZED REWARD ====================
def compute_reward_vectorized(df, R_hot, W_vel, W_temp, W_smooth, W_entropy, p=1.5, beta=5.0, tau=1.0):
    """Vectorized reward: R_hot*hot - vel_penalty - temp_penalty - smooth - entropy"""
    hot = df["hot_score"].values if "hot_score" in df.columns else df["is_hotspot"].values.astype(float)
    is_night = df["is_night"].values.astype(float)
    
    # Đêm → phạt vận tốc nặng hơn 50%
    w_vel_adaptive = W_vel * (1 + 0.5 * is_night)
    
    return (
        R_hot * hot
        - w_vel_adaptive * (df["vel_norm"].values ** p)
        - W_temp * softplus(beta * (df["temp_std"].values - tau)) / beta
        - W_smooth * df["turn_score"].values
        - W_entropy * df["turning_entropy"].values
    )


# ==================== EVALUATION ====================
def eval_reward(df, reward_fn):
    rewards = df.apply(reward_fn, axis=1)
    return rewards.groupby(df["episode"]).mean()
