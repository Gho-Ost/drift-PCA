import os
import logging
import pandas as pd
import numpy as np
import scipy.stats
from river import datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = "data/real"
TARGET_DIR = "data/real/gen"

def load_csv_data(file_path, target_col):
    logger.info(f"Loading CSV data from {file_path}")
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = list(df.drop(columns=[target_col]).columns)
    return X, y, feature_names

def load_river_elec():
    logger.info("Loading Elec2 dataset from River")
    elec = datasets.Elec2()
    examples = list(elec.take(45332))
    num_features = len(examples[0][0])
    feature_columns = list(examples[0][0].keys())
    
    X = np.zeros((len(examples), num_features))
    y = np.zeros(len(examples))
    for idx, (x, target) in enumerate(examples):
        X[idx] = [x[col] for col in feature_columns]
        y[idx] = int(target) if isinstance(target, (int, float, bool)) else (1 if target else 0)
    return X, y, feature_columns

def compute_ks_score(X_pre, X_post):
    n_features = X_pre.shape[1]
    ks_stats = []
    p_vals = []
    sig_count = 0
    for j in range(n_features):
        stat, p_val = scipy.stats.ks_2samp(X_pre[:, j], X_post[:, j], method='asymp')
        ks_stats.append(stat)
        p_vals.append(p_val)
        if p_val < 0.01:
            sig_count += 1
    return np.mean(ks_stats), np.mean(p_vals), sig_count

def find_optimal_consecutive_windows_ks(X, y, dataset_name, W, step_t=50):
    n = len(X)
    best_score = -1
    best_config = None
    
    logger.info(f"Searching for 3 consecutive windows for {dataset_name} using Kolmogorov-Smirnov test (N={n}, W={W}, step_t={step_t})")
    
    for T in range(0, n - 3*W + 1, step_t):
        stat1, p1, sig1 = compute_ks_score(X[T : T+W], X[T+W : T+2*W])
        stat2, p2, sig2 = compute_ks_score(X[T+W : T+2*W], X[T+2*W : T+3*W])
        
        # Harmonic mean to ensure both boundaries show strong drift (based on KS statistic)
        if stat1 + stat2 > 0:
            h_mean = 2 * stat1 * stat2 / (stat1 + stat2)
        else:
            h_mean = 0.0
            
        if h_mean > best_score:
            best_score = h_mean
            best_config = (W, T, (stat1, p1, sig1), (stat2, p2, sig2), h_mean)
            
    return best_config

def save_window_df(X_w, y_w, feature_names, filename):
    df = pd.DataFrame(X_w, columns=feature_names)
    df["target"] = y_w
    filepath = os.path.join(TARGET_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {filepath} (shape: {df.shape})")

def main():
    logger.info("Starting real-world dataset processing script")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    datasets_info = [
        {
            "name": "elec",
            "loader": load_river_elec,
            "W": 1000, "step_t": 100
        },
        {
            "name": "keystroke",
            "loader": lambda: load_csv_data(os.path.join(DATA_DIR, "Keystroke.csv"), "Target"),
            "W": 400, "step_t": 20
        },
        {
            "name": "insects",
            "loader": lambda: load_csv_data(os.path.join(DATA_DIR, "INSECTS-abrupt_balanced_norm.csv"), "Class"),
            "W": 1000, "step_t": 100
        },
        {
            "name": "gassensor",
            "loader": lambda: load_csv_data(os.path.join(DATA_DIR, "GasSensorArray.csv"), "Target"),
            "W": 1000, "step_t": 100
        },
        {
            "name": "noaa",
            "loader": lambda: load_csv_data(os.path.join(DATA_DIR, "NOAA.csv"), "Target"),
            "W": 1000, "step_t": 100
        }
    ]
    
    results = {}
    
    for d in datasets_info:
        name = d["name"]
        X, y, feature_names = d["loader"]()
        
        config = find_optimal_consecutive_windows_ks(
            X, y, name, W=d["W"], step_t=d["step_t"]
        )
        
        if not config:
            logger.error(f"Failed to find optimal windows for {name}")
            continue
            
        W, T, drift1_info, drift2_info, h_mean = config
        stat1, p1, sig1 = drift1_info
        stat2, p2, sig2 = drift2_info
        
        results[name] = {
            "W": W, "T": T, 
            "stat1": stat1, "p1": p1, "sig1": sig1,
            "stat2": stat2, "p2": p2, "sig2": sig2,
            "h_mean": h_mean,
            "n_features": X.shape[1]
        }
        
        logger.info(f"Optimal window configuration for {name}: W={W}, T={T}")
        
        # Slice the 3 consecutive windows
        w1_X, w1_y = X[T : T+W], y[T : T+W]
        w2_X, w2_y = X[T+W : T+2*W], y[T+W : T+2*W]
        w3_X, w3_y = X[T+2*W : T+3*W], y[T+2*W : T+3*W]
        
        # Save 3 consecutive windows
        save_window_df(w1_X, w1_y, feature_names, f"{name}_w1.csv")
        save_window_df(w2_X, w2_y, feature_names, f"{name}_w2.csv")
        save_window_df(w3_X, w3_y, feature_names, f"{name}_w3.csv")
        
        # Save traditional drift 1 pre and post
        save_window_df(w1_X, w1_y, feature_names, f"{name}_drift1_pre.csv")
        save_window_df(w2_X, w2_y, feature_names, f"{name}_drift1_post.csv")
        
        # Save traditional drift 2 pre and post
        save_window_df(w2_X, w2_y, feature_names, f"{name}_drift2_pre.csv")
        save_window_df(w3_X, w3_y, feature_names, f"{name}_drift2_post.csv")
        
    print("\n" + "="*80)
    print("DRIFT DETECTION PROCESS EXPLANATION")
    print("="*80)
    print("Because real-world datasets lack ground-truth drift timestamps, a standard, independent")
    print("concept drift detection method based on the Kolmogorov-Smirnov (KS) test was used.")
    print("\nMethodology Details:")
    print("1. The detector is completely independent of the Drift-PCA method. It operates directly")
    print("   on the raw feature distributions rather than PCA scores or SVD projection metrics.")
    print("2. For any candidate split point T and window size W, we compute the two-sample KS test")
    print("   statistic and p-value for each feature comparing the reference window [T, T+W] and current window [T+W, T+2W].")
    print("3. The KS statistic measures the maximum distance between the cumulative distribution functions (CDFs)")
    print("   of the two samples. The feature-wise KS statistics are averaged to compute a global drift score in [0, 1].")
    print("4. To capture multiple consecutive drifts (bonus points), we search for three consecutive windows")
    print("   W1=[T, T+W], W2=[T+W, T+2W], and W3=[T+2W, T+3W] separated by two drift boundaries (T+W and T+2W).")
    print("5. We maximize the harmonic mean of the drift scores of the two boundaries to ensure both drifts")
    print("   are strong and balanced, avoiding scenarios where one boundary has high drift and the other has none.")
    print("\nDetected Window Configurations:")
    for name, res in results.items():
        print(f"\n- Dataset: {name.upper()}")
        print(f"  Window size (W): {res['W']}")
        print(f"  Start index (T): {res['T']}")
        print(f"  Window 1 range:  [{res['T']}, {res['T'] + res['W']}]")
        print(f"  Window 2 range:  [{res['T'] + res['W']}, {res['T'] + 2*res['W']}]")
        print(f"  Window 3 range:  [{res['T'] + 2*res['W']}, {res['T'] + 3*res['W']}]")
        print(f"  Drift 1 Boundary (T+W):   Index {res['T'] + res['W']} (Avg KS: {res['stat1']:.4f}, Avg P-value: {res['p1']:.2e}, Sig Features (p<0.01): {res['sig1']}/{res['n_features']})")
        print(f"  Drift 2 Boundary (T+2W):  Index {res['T'] + 2*res['W']} (Avg KS: {res['stat2']:.4f}, Avg P-value: {res['p2']:.2e}, Sig Features (p<0.01): {res['sig2']}/{res['n_features']})")
        print(f"  Harmonic Mean KS Score:   {res['h_mean']:.4f}")
        
    print("\nSaved Output Files:")
    print("For each dataset, the script has written the following CSVs to:")
    print(f"  {TARGET_DIR}/")
    print("  - {{dataset}}_w1.csv, {{dataset}}_w2.csv, {{dataset}}_w3.csv (consecutive windows)")
    print("  - {{dataset}}_drift1_pre.csv / post.csv (pre/post pairs for the first drift boundary)")
    print("  - {{dataset}}_drift2_pre.csv / post.csv (pre/post pairs for the second drift boundary)")
    
    # Save LaTeX table summary
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Concept Drift Window Configurations for Real-World Datasets}",
        "\\label{tab:real_windows_drift}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccccccc}",
        "\\toprule",
        "Dataset & $W$ & $T$ & Drift 1 Range & Avg $D_1$ & Avg $p_1$ & Sig. Feat. 1 & Drift 2 Range & Avg $D_2$ & Avg $p_2$ & Sig. Feat. 2 \\\\",
        "\\midrule"
    ]
    
    for name, res in results.items():
        name_esc = name.upper().replace("_", "\\_")
        w1_range = f"[{res['T']}, {res['T'] + res['W']}]"
        w2_range = f"[{res['T'] + res['W']}, {res['T'] + 2*res['W']}]"
        w3_range = f"[{res['T'] + 2*res['W']}, {res['T'] + 3*res['W']}]"
        
        def fmt_p(p):
            if p == 0:
                return "0"
            s = f"{p:.2e}"
            base, exp = s.split("e")
            return f"{base} \\times 10^{{{int(exp)}}}"

        p1_tex = fmt_p(res['p1'])
        p2_tex = fmt_p(res['p2'])
        
        latex_lines.append(
            f"{name_esc} & {res['W']} & {res['T']} & {w1_range} vs {w2_range} & {res['stat1']:.4f} & ${p1_tex}$ & {res['sig1']}/{res['n_features']} & {w2_range} vs {w3_range} & {res['stat2']:.4f} & ${p2_tex}$ & {res['sig2']}/{res['n_features']} \\\\"
        )
        
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])
    
    latex_path = os.path.join(TARGET_DIR, "real_windows_summary.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  - real_windows_summary.tex (LaTeX table summary of drift configurations)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
