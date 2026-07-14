import os
import pandas as pd
import numpy as np
import scipy.stats
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
        stat, p_val = scipy.stats.ks_2samp(X_pre[:, j], X_post[:, j])
        ks_stats.append(stat)
    return np.mean(ks_stats)

def search_ks_consecutive_windows(X, y, dataset_name, min_w=500, max_w=2000, step_w=100, step_t=50):
    n = len(X)
    best_score = -1
    best_config = None
    
    actual_max_w = min(max_w, n // 3)
    actual_min_w = min(min_w, actual_max_w)
    
    for W in range(actual_min_w, actual_max_w + 1, step_w):
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
                
    if best_config:
        W, T, s1, s2, hm = best_config
        print(f"Best Config for {dataset_name}:")
        print(f"  Window Size W: {W}")
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
        ("elec", load_river_elec, 500, 2000, 100, 100),
        ("keystroke", lambda: load_csv_data("data-masters/real/Keystroke.csv", "Target"), 300, 500, 50, 20),
        ("insects", lambda: load_csv_data("data-masters/real/INSECTS-abrupt_balanced_norm.csv", "Class"), 1000, 2000, 100, 200),
        ("gassensor", lambda: load_csv_data("data-masters/real/GasSensorArray.csv", "Target"), 500, 2000, 100, 100),
        ("noaa", lambda: load_csv_data("data-masters/real/NOAA.csv", "Target"), 500, 2000, 100, 100)
    ]
    
    for name, loader, min_w, max_w, step_w, step_t in datasets_to_test:
        X, y = loader()
        search_ks_consecutive_windows(X, y, name, min_w, max_w, step_w, step_t)

if __name__ == "__main__":
    main()
