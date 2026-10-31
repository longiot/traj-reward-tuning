import pandas as pd
from utils import loeo_grid_search, select_final_params_loeo, save_final_params

CSV_PATH = "dataset/elephants_kruger_2007_transition.csv"
R_HOT = 1.5
SEED = 42

ALPHA_HOT = 0.2   # Hệ số tính Score
EPS_SCORE = 1e-4  # Điểm gần nhau thì chọn W nhỏ hơn

USE_BAD_ECO = True


def main():
    print("=" * 60)
    print("LOAD DATA")
    print("=" * 60)
    # =======================================
    # LOAD DATA
    # =======================================
    df_raw = pd.read_csv(CSV_PATH)
    n_episodes = df_raw["episode"].nunique()
    print(f"CSV path   : {CSV_PATH}")
    print(f"#episodes  : {n_episodes}")
    print(f"#rows      : {len(df_raw)}")
    # =======================================

    print("\n" + "=" * 60)
    print("LOEO GRID SEARCH (held-pairwise + hotspot-priority)")
    print("=" * 60)
    print(f"Config: R_HOT={R_HOT}, seed={SEED}")
    print(f"Select score = median_held_acc + {ALPHA_HOT} * median_tau_hot (eps={EPS_SCORE})")

    W_vel_grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    W_temp_grid = [0, 0.25, 0.5, 0.75, 1.0]
    W_smooth_grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    W_entropy_grid = [0, 0.25, 0.5, 0.75, 1.0]

    print(f"Grid: W_vel   = {W_vel_grid}")
    print(f"Grid: W_temp  = {W_temp_grid}")
    print(f"Grid: W_smooth = {W_smooth_grid}")  
    print(f"Grid: W_entropy = {W_entropy_grid}")
    total_evals = len(W_vel_grid) * len(W_temp_grid) * len(W_smooth_grid) * len(W_entropy_grid) * n_episodes
    print(f"Total evals: {total_evals}\n")

    # ===========================
    # LOEO GRID SEARCH
    # ===========================
    if USE_BAD_ECO:
        print("\n[!!!] ĐANG CHẠY CHẾ ĐỘ BAD ECO (Kiểm chứng giả định sai) [!!!]")
    else:
        print("\n[INFO] Đang chạy chế độ NORMAL ECO (Giả định chuẩn)")

    res = loeo_grid_search(
        df_raw=df_raw,
        W_vel_grid=W_vel_grid,
        W_temp_grid=W_temp_grid,
        W_smooth_grid=W_smooth_grid,
        W_entropy_grid=W_entropy_grid,
        R_HOT=R_HOT,
        seed=SEED,
        verbose=True,
        p=1.5,
        beta=5.0,
        tau=1.0,
        grid_size=50,
        hot_quantile=0.8,
        use_bad_eco=USE_BAD_ECO,
    )
    # ===========================

    print("\n" + "=" * 60)
    print("SELECT BEST PARAMS")
    print("=" * 60)

    # ================================
    # SELECT BEST PARAMS
    # ================================
    best = select_final_params_loeo(
        res,
        verbose=True,
        alpha_hot=ALPHA_HOT,
        eps_score=EPS_SCORE,
    )
    # ================================

    print("\n" + "=" * 60)
    print("SAVE FINAL PARAMS")
    print("=" * 60)
    save_final_params(best, output_path="dataset_preprocessed/final_params_loeo.json")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Best W_vel      = {best['W_vel']:.2f}")
    print(f"Best W_temp     = {best['W_temp']:.2f}")
    if 'W_smooth' in best:
        print(f"Best W_smooth   = {best['W_smooth']:.2f}")
    if 'W_entropy' in best:
        print(f"Best W_entropy  = {best['W_entropy']:.2f}")

    print(f"score           = {best['score']:.6f}")
    print(f"median_held_acc = {best['median_held_acc']:.3f}")
    print(f"median_tau_hot  = {best['median_tau_hot']:.3f}")
    print(f"L1 norm         = {best['l1_norm']:.3f}")
    if 'n_folds' in best:
        print(f"#LOEO folds     = {best['n_folds']}")
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()