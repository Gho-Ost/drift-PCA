import os
import pandas as pd
import numpy as np
import scipy.stats
import time
from river import datasets

def load_csv_data(file_path, target_col):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y

def load_river_elec():
    elec = datasets.Elec2()
    examples = list(elec.take(45332))
    num_features = len(examples[0][0])
    feature_columns = list(examples[0][0].keys())
    X = np.zeros((len(examples), num_features))
    y = np.zeros(len(examples))
    for idx, (x, target) in enumerate(examples):
        X[idx] = [x[col] for col in feature_columns]
        y[idx] = int(target) if isinstance(target, (int, float, bool)) else (1 if target else 0)
    return X, y

def compute_ks_score(X_pre, X_post):
    n_features = X_pre.shape[1]
    ks_stats = []
    for j in range(n_features):
        stat, p_val = scipy.stats.ks_2samp(X_pre[:, j], X_post[:, j], method='asymp')
        ks_stats.append(stat)
    return np.mean(ks_stats)

def search_ks_consecutive_windows(X, y, dataset_name, W, step_t=50):
    n = len(X)
    best_score = -1
    best_config = None
    
    t0 = time.time()
    for T in range(0, n - 3*W + 1, step_t):
        score1 = compute_ks_score(X[T : T+W], X[T+W : T+2*W])
        score2 = compute_ks_score(X[T+W : T+2*W], X[T+2*W : T+3*W])
        
        # Harmonic mean to ensure both boundaries show strong drift
        if score1 + score2 > 0:
            h_mean = 2 * score1 * score2 / (score1 + score2)
        else:
            h_mean = 0.0
            
        if h_mean > best_score:
            best_score = h_mean
            best_config = (W, T, score1, score2, h_mean)
            
    dur = time.time() - t0
    if best_config:
        W, T, s1, s2, hm = best_config
        print(f"Best Config for {dataset_name} (W={W}, step_t={step_t}, search_time={dur:.2f}s):")
        print(f"  Start Index T: {T}")
        print(f"  Window 1: [{T}, {T+W}]")
        print(f"  Window 2: [{T+W}, {T+2*W}]")
        print(f"  Window 3: [{T+2*W}, {T+3*W}]")
        print(f"  Drift 1 KS-Score: {s1:.4f}")
        print(f"  Drift 2 KS-Score: {s2:.4f}")
        print(f"  Harmonic Mean: {hm:.4f}")
    else:
        print(f"No valid configuration found for {dataset_name}")

def main():
    datasets_to_test = [
        ("elec", load_river_elec, 1000, 200),
        ("keystroke", lambda: load_csv_data("data-masters/real/Keystroke.csv", "Target"), 400, 20),
        ("insects", lambda: load_csv_data("data-masters/real/INSECTS-abrupt_balanced_norm.csv", "Class"), 1000, 200),
        ("gassensor", lambda: load_csv_data("data-masters/real/GasSensorArray.csv", "Target"), 1000, 100),
        ("noaa", lambda: load_csv_data("data-masters/real/NOAA.csv", "Target"), 1000, 100)
    ]
    
    for name, loader, W, step_t in datasets_to_test:
        X, y = loader()
        search_ks_consecutive_windows(X, y, name, W, step_t)

if __name__ == "__main__":
    main()
