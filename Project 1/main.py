# Supervised Learning Mini-Project (Undergrad): Time-Series Regression on Lorenz & Ikeda
# Models: Linear (MLE/OLS), Linear MAP (Ridge), MLP, RBF via Random Fourier Features + Ridge
# Tasks: Clear + Noisy (10 dB, 3 dB), Horizons L ∈ {1, 10}, window size D
#
# Outputs:
#   - results_timeseries_regression.csv  (metrics)
#   - figs/*.png                         (a few example prediction plots)
#
# Dependencies: numpy, pandas, scikit-learn, matplotlib (no seaborn)

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------
# Config (tweak if needed)
# -------------------------
RANDOM_SEED = 42
N_POINTS     = 2000   # base series length (kept modest so it runs quickly)
D_WINDOW     = 10     # input window size
HORIZONS     = [1, 10]
TRAIN_RATIO  = 0.7
SNR_LIST_DB  = [None, 10.0, 3.0]  # None = clear, or add noise at these SNRs
SAVE_DIR     = "figs"
MAX_PLOTS    = 4      # cap the number of saved example plots
# MLP sizing to keep runtime short
MLP_HIDDEN   = (64,)
MLP_MAX_ITER = 150
# Random Fourier features sizing
RFF_GAMMA    = 1.0
RFF_NCOMP    = 200
# -------------------------

# Reproducibility
def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)

set_seed(RANDOM_SEED)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(SAVE_DIR)

# ---------- Helpers ----------
def add_awgn_snr(x: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to achieve target SNR (in dB)."""
    x = np.asarray(x).reshape(-1)
    p_signal = np.mean(x**2)
    snr_linear = 10.0**(snr_db / 10.0)
    p_noise = p_signal / snr_linear
    noise_std = math.sqrt(p_noise)
    n = np.random.normal(0.0, noise_std, size=x.shape)
    return x + n

def make_supervised(series: np.ndarray, D: int, L: int):
    """
    Build supervised pairs:
        X_n = [s(n), s(n+1), ..., s(n+D-1)]
        y_n = s(n + D - 1 + L)
    """
    s = np.asarray(series).reshape(-1)
    N = len(s)
    M = N - (D - 1) - L
    if M <= 0:
        raise ValueError("Series too short for given D and L.")
    X = np.zeros((M, D))
    y = np.zeros(M)
    for i in range(M):
        X[i] = s[i:i+D]
        y[i] = s[i + D - 1 + L]
    return X, y

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    rmse = math.sqrt(mse)
    stdy = np.std(y_true)
    nrmse = rmse / stdy if stdy > 0 else np.nan
    return {"MSE": mse, "MAE": mae, "R2": r2, "NRMSE": nrmse}

# ---------- Time-series generators (no SciPy) ----------
def lorenz_1d(N=2000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0,
    transient=400, x0=1.0, y0=1.0, z0=1.0):
    """Integrate Lorenz and return x(t) only (1-D)."""
    def f(state):
        x, y, z = state
        dx = sigma*(y - x)
        dy = x*(rho - z) - y
        dz = x*y - beta*z
        return np.array([dx, dy, dz], float)

    state = np.array([x0, y0, z0], float)
    steps = N + transient
    xs = np.zeros(steps)

    for i in range(steps):
        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        xs[i] = state[0]
    return xs[transient:]

def ikeda_1d(N=2000, u=0.918, x0=0.1, y0=0.1, transient=300):
    """Iterate Ikeda map and return x(n) only (1-D)."""
    x, y = x0, y0
    xs = []
    steps = N + transient
    for _ in range(steps):
        t = 0.4 - 6.0 / (1 + x*x + y*y)
        x_next = 1 + u*(x*np.cos(t) - y*np.sin(t))
        y_next =     u*(x*np.sin(t) + y*np.cos(t))
        x, y = x_next, y_next
        xs.append(x)
    return np.array(xs, float)[transient:]

# ---------- Models ----------
def make_models(random_state=RANDOM_SEED):
    """Return a dict of 4 pipelines, each with input scaling and target scaling."""
    def TTR(est):
        # Scale y for better conditioning
        return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())

    models = {}

    # Linear MLE (OLS)
    models["Linear_MLE"] = Pipeline([
        ("xscale", StandardScaler()),
        ("reg", TTR(LinearRegression()))
    ])

    # Linear MAP (Ridge)
    models["Linear_MAP_Ridge"] = Pipeline([
        ("xscale", StandardScaler()),
        ("reg", TTR(Ridge(alpha=1.0, random_state=random_state)))
    ])

    # MLP (small, tanh) — sized to fit quickly
    models["MLP"] = Pipeline([
        ("xscale", StandardScaler()),
        ("reg", TTR(MLPRegressor(hidden_layer_sizes=MLP_HIDDEN,
                                    activation="tanh",
                                    solver="adam",
                                    max_iter=MLP_MAX_ITER,
                                    random_state=random_state)))
    ])

    # "RBF network" via Random Fourier Features + Ridge
    models["RBF_RandomFeatures"] = Pipeline([
        ("xscale", StandardScaler()),
        ("rbf", RBFSampler(gamma=RFF_GAMMA, n_components=RFF_NCOMP, random_state=random_state)),
        ("reg", TTR(Ridge(alpha=1.0, random_state=random_state)))
    ])

    return models

# ---------- Experiment ----------
def run():
    set_seed(RANDOM_SEED)

    # Base series
    lorenz = lorenz_1d(N=N_POINTS)
    ikeda  = ikeda_1d(N=N_POINTS)

    series_bank = [
        ("Lorenz", None, lorenz),
        ("Lorenz", 10.0, add_awgn_snr(lorenz, 10.0)),
        ("Lorenz",  3.0, add_awgn_snr(lorenz,  3.0)),
        ("Ikeda",  None, ikeda),
        ("Ikeda", 10.0, add_awgn_snr(ikeda, 10.0)),
        ("Ikeda",  3.0, add_awgn_snr(ikeda,  3.0)),
    ]

    models = make_models()
    results = []
    plots_made = 0

    for (name, snr, s) in series_bank:
        label = f"{name}_{'Clear' if snr is None else f'SNR_{int(snr)}dB'}"
        for L in HORIZONS:
            X, y = make_supervised(s, D=D_WINDOW, L=L)

            n = len(y)
            ntr = int(TRAIN_RATIO * n)
            Xtr, ytr = X[:ntr], y[:ntr]
            Xte, yte = X[ntr:], y[ntr:]

            for mname, pipe in models.items():
                pipe.fit(Xtr, ytr)
                yhat = pipe.predict(Xte)
                m = metrics(yte, yhat)
                results.append({
                    "Series": label,
                    "Horizon_L": L,
                    "Model": mname,
                    **m
                })

                # Save a couple of illustrative plots
                if plots_made < MAX_PLOTS:
                    w = min(300, len(yte))
                    plt.figure()
                    plt.plot(yte[:w], label="True")
                    plt.plot(yhat[:w], label="Pred")
                    plt.title(f"{label} | L={L} | {mname}")
                    plt.xlabel("Test sample index")
                    plt.ylabel("s(k)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(SAVE_DIR, f"{label}_L{L}_{mname}.png"), dpi=150)
                    plt.close()
                    plots_made += 1

    # Report + save
    df = pd.DataFrame(results).sort_values(["Series", "Horizon_L", "MSE"])
    print("\n=== Top rows (sorted by MSE) ===")
    print(df.head(12).to_string(index=False))

    out_csv = "results_timeseries_regression.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to: {out_csv}")
    print(f"Saved example plots to: {SAVE_DIR}/")

if __name__ == "__main__":
    run()
