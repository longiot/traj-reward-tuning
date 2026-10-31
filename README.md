# Wildlife Trajectory Reward Optimization

A lightweight experimental project on designing and tuning a **nonlinear reward function** from GPS movement trajectories of African elephants in Kruger National Park.

---

## 🎯 What is this?

This is a fun mini-research experiment exploring whether a handcrafted, ecologically-inspired reward function can correctly rank animal movement trajectories across different individuals without data leakage.

It tunes weights for spatial preference, energy conservation, thermal stress, and path smoothness using **Leave-One-Episode-Out (LOEO)** cross-validation.

---

## 📐 Reward Function

At each trajectory step, the reward $R(s, a)$ is computed as:

$$R = R_{hot} \cdot \text{hot} - \mathcal{P}_{vel} - \mathcal{P}_{temp} - \mathcal{P}_{turn} - \mathcal{P}_{entropy}$$

- **Hotspot Reward ($R_{hot}$)**: Incentive for visiting frequently used resource zones (2D grid density).
- **Velocity Penalty ($\mathcal{P}_{vel}$)**: $W_{vel} \cdot (1 + 0.5 \cdot \text{is\_night}) \cdot v^{1.5}$. Nocturnal movement is penalized 50% more (day/night computed via `astral` sun ephemeris).
- **Temperature Penalty ($\mathcal{P}_{temp}$)**: Softplus barrier $W_{temp} \cdot \frac{\text{softplus}(\beta(T - \tau))}{\beta}$ to penalize extreme heat.
- **Turn Penalty ($\mathcal{P}_{turn}$)**: Instantaneous turn angle penalty $W_{smooth} \cdot (\Delta \theta / 180^\circ)^2$.
- **Entropy Penalty ($\mathcal{P}_{entropy}$)**: Rolling-window directional entropy $W_{entropy} \cdot \frac{\mathcal{H}(\Delta \theta)}{\ln(18)}$ to discourage aimless wandering.

---

## 🔬 How it Works

1. **LOEO Cross-Validation**: Leaves one elephant episode out for testing. Hotspot grids and feature scalers are fitted **only** on training folds to prevent data leakage.
2. **Pairwise Ranking Accuracy**: Ranks held-out trajectories against an ecological composite metric (Eco-Score).
3. **Plateau Selection**: Picks top-scoring parameters and applies L1-regularization (favoring simpler, smaller weights).
4. **Sanity Check (Bad Eco Mode)**: An ablation mode with inverted ecological assumptions to verify the model doesn't fit on noise.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install numpy pandas scipy matplotlib astral
```

### 2. Run Grid Search
```bash
python src/main.py
```
Outputs the optimal parameters to `dataset_preprocessed/final_params_loeo.json`.

### 3. Explore Experiments
Open `src/optimize.ipynb` to view top-10 parameter rankings and the Normal vs. Bad Eco comparison.

---

## 📁 Project Structure

```text
├── dataset/                     # Trajectory CSV data
├── dataset_preprocessed/        # Heatmaps & saved params (JSON)
├── src/
│   ├── main.py                  # Grid search runner
│   ├── reward.py                # Scalar & vectorized reward functions
│   ├── preprocess_hotspots.py   # 2D spatial binning & hotspot extraction
│   ├── utils.py                 # Astronomy day/night, feature engineering, LOEO CV
│   └── optimize.ipynb           # Interactive analysis & comparison
└── README.md
```
