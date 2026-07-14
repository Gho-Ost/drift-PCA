import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for speed testing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_methods.drift_component_analysis import DriftComponentAnalysis
from analysis_methods.dca_utils import (
    plot_dca_scatter,
    plot_loadings_compass,
    plot_drift_compass
)

RESULTS_DIR = "results/speed_experiments"

def generate_synthetic_drift_data(n_samples, n_features, n_classes, drift_magnitude=1.5):
    """
    Generate pre-drift and post-drift synthetic data.
    Each class has a baseline center (pre-drift) and a shifted center (post-drift).
    """
    # Ensure each class has enough samples
    samples_per_class = max(2, n_samples // n_classes)
    
    X_pre_list = []
    y_pre_list = []
    X_post_list = []
    y_post_list = []
    
    # Class centers
    pre_centers = np.random.uniform(-3.0, 3.0, size=(n_classes, n_features))
    drift_vectors = np.random.uniform(-drift_magnitude, drift_magnitude, size=(n_classes, n_features))
    post_centers = pre_centers + drift_vectors
    
    for c in range(n_classes):
        X_c_pre = np.random.normal(loc=pre_centers[c], scale=1.0, size=(samples_per_class, n_features))
        X_pre_list.append(X_c_pre)
        y_pre_list.append(np.full(samples_per_class, c))
        
        X_c_post = np.random.normal(loc=post_centers[c], scale=1.0, size=(samples_per_class, n_features))
        X_post_list.append(X_c_post)
        y_post_list.append(np.full(samples_per_class, c))
        
    X_pre = np.vstack(X_pre_list)
    y_pre = np.concatenate(y_pre_list)
    X_post = np.vstack(X_post_list)
    y_post = np.concatenate(y_post_list)
    
    feature_names = [f"F_{i}" for i in range(n_features)]
    classes = np.array(range(n_classes))
    
    return X_pre, y_pre, X_post, y_post, feature_names, classes

def benchmark_single_config(n_samples, n_features, n_classes, temp_path):
    """
    Benchmark scaling, fitting, plotting, and saving for a single configuration.
    """
    # Generate data
    X_pre, y_pre, X_post, y_post, feature_names, classes = generate_synthetic_drift_data(
        n_samples, n_features, n_classes
    )
    
    start_time = time.time()
    
    # 1. Scaling
    scaler = StandardScaler()
    X_pre_scaled = scaler.fit_transform(X_pre)
    X_post_scaled = scaler.transform(X_post)
    
    # 2. DCA fitting
    dca = DriftComponentAnalysis(n_components=2, by_class=True)
    dca.fit(X_pre_scaled, X_post_scaled, y_pre, y_post)
    
    # 3. Figure creation & plotting
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
    
    ax_scatter = fig.add_subplot(gs[0, :])     
    ax_loadings = fig.add_subplot(gs[1, 0])     
    ax_drift = fig.add_subplot(gs[1, 1])        
    
    # Plot components (conforming to sudden, per-class, no boundary, class colors)
    plot_dca_scatter(
        X_pre_scaled, y_pre, X_post_scaled, y_post, dca, ax=ax_scatter, 
        pre_drift_model=None, color_scheme="class",
        discrete_boundary=False,
        draw_boundary=False,
        highlight_misclassifications=False,
        hide_pre_drift_points=False,
        grid_points=50,
        drift_type="sudden"
    )
    
    plot_loadings_compass(dca, ax=ax_loadings, feature_names=feature_names, scale_loadings=True, feature_importances=None)
    plot_drift_compass(dca, ax=ax_drift, classes=classes, color_scheme="class")
    
    plt.tight_layout()
    fig.savefig(temp_path, bbox_inches='tight')
    plt.close(fig)
    
    end_time = time.time()
    return end_time - start_time

def benchmark_feature_config(n_samples, n_features, n_classes, temp_path, 
                             use_importance=False, draw_boundary=False, 
                             grid_points=50, highlight_misclassifications=False):
    """
    Benchmark speed when model, importance, or boundary drawing are added.
    """
    # Generate data
    X_pre, y_pre, X_post, y_post, feature_names, classes = generate_synthetic_drift_data(
        n_samples, n_features, n_classes
    )
    
    start_time = time.time()
    
    # 1. Scaling
    scaler = StandardScaler()
    X_pre_scaled = scaler.fit_transform(X_pre)
    X_post_scaled = scaler.transform(X_post)
    
    # 2. Train SVC model if needed
    needs_model = draw_boundary or highlight_misclassifications or use_importance
    pre_drift_model = None
    if needs_model:
        pre_drift_model = SVC(kernel='rbf', probability=True, random_state=42)
        pre_drift_model.fit(X_pre_scaled, y_pre)
        
    # 3. Calculate feature importance if needed
    feature_importances = None
    if use_importance and pre_drift_model is not None:
        result = permutation_importance(pre_drift_model, X_pre_scaled, y_pre, n_repeats=5, random_state=42, n_jobs=-1)
        feature_importances = np.maximum(result.importances_mean, 0)
        
    # 4. DCA fitting
    dca = DriftComponentAnalysis(n_components=2, by_class=True)
    dca.fit(X_pre_scaled, X_post_scaled, y_pre, y_post)
    
    # 5. Figure creation & plotting
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])
    
    ax_scatter = fig.add_subplot(gs[0, :])     
    ax_loadings = fig.add_subplot(gs[1, 0])     
    ax_drift = fig.add_subplot(gs[1, 1])        
    
    contour_info = plot_dca_scatter(
        X_pre_scaled, y_pre, X_post_scaled, y_post, dca, ax=ax_scatter, 
        pre_drift_model=pre_drift_model, color_scheme="class",
        discrete_boundary=False,
        draw_boundary=draw_boundary,
        highlight_misclassifications=highlight_misclassifications,
        hide_pre_drift_points=False,
        grid_points=grid_points,
        drift_type="sudden"
    )
    
    # Handle colorbar for probability boundaries if drawn (C=2 only)
    contour, is_discrete = contour_info if isinstance(contour_info, tuple) else (contour_info, False)
    if contour is not None and not is_discrete and len(classes) == 2:
        cbar_ax = fig.add_axes([0.91, 0.55, 0.02, 0.3])
        fig.colorbar(contour, cax=cbar_ax)
        plt.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)
        
    plot_loadings_compass(dca, ax=ax_loadings, feature_names=feature_names, scale_loadings=True, feature_importances=feature_importances)
    plot_drift_compass(dca, ax=ax_drift, classes=classes, color_scheme="class")
    
    if contour is None:
        plt.tight_layout()
        
    fig.savefig(temp_path, bbox_inches='tight')
    plt.close(fig)
    
    end_time = time.time()
    return end_time - start_time

def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    temp_path = os.path.join(RESULTS_DIR, "temp_speed_dca.png")
    
    points_list = [500, 1000, 5000]
    features_list = [5, 20, 100]
    classes_list = [2, 5, 10, 20]
    
    n_trials = 5
    
    # ----------------- PART 1: Size Benchmarking -----------------
    print("Starting Size Benchmarking Experiment...")
    print(f"Running each configuration {n_trials} times for average stability.")
    print("-" * 60)
    
    results = []
    for points in points_list:
        for features in features_list:
            for classes in classes_list:
                trial_times = []
                for trial in range(n_trials):
                    elapsed = benchmark_single_config(points, features, classes, temp_path)
                    trial_times.append(elapsed)
                avg_time = np.mean(trial_times)
                std_time = np.std(trial_times)
                print(f"Points: {points:5d} | Features: {features:3d} | Classes: {classes:2d} | Time: {avg_time:.4f}s (std: {std_time:.4f}s)")
                results.append({
                    "points": points,
                    "features": features,
                    "classes": classes,
                    "avg_time": avg_time,
                    "std_time": std_time
                })
                
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "speed_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw size results to {csv_path}")
    
    # Construct Size LaTeX Table
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Execution Speed (seconds) of Drift PCA (DCA) Pipeline by Dataset Size}")
    latex_lines.append(r"\label{tab:dca_execution_speed_size}")
    latex_lines.append(r"\begin{tabular}{cc|cccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\textbf{Points ($N$)} & \textbf{Features ($F$)} & \textbf{C = 2} & \textbf{C = 5} & \textbf{C = 10} & \textbf{C = 20} \\")
    latex_lines.append(r"\hline")
    
    for points in points_list:
        points_str = f"{points:,}"
        for f_idx, features in enumerate(features_list):
            row_times = []
            for classes in classes_list:
                row_data = df[(df["points"] == points) & (df["features"] == features) & (df["classes"] == classes)].iloc[0]
                avg = row_data["avg_time"]
                std = row_data["std_time"]
                row_times.append(f"${avg:.4f} \\pm {std:.4f}$")
            
            p_val = points_str if f_idx == 0 else ""
            latex_lines.append(f"{p_val} & {features} & " + " & ".join(row_times) + r" \\")
        latex_lines.append(r"\hline")
        
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    size_latex_code = "\n".join(latex_lines)
    
    # Common configurations to test for both C=2 and C=10
    overhead_configs = [
        {"name": "Baseline (DCA only)", "importance": False, "boundary": False, "grid": 50, "highlight": False},
        {"name": "Feature Importance", "importance": True, "boundary": False, "grid": 50, "highlight": False},
        {"name": "Misclassification Highlight", "importance": False, "boundary": False, "grid": 50, "highlight": True},
        {"name": "Decision Boundary (Grid 50)", "importance": False, "boundary": True, "grid": 50, "highlight": False},
        {"name": "Decision Boundary (Grid 100)", "importance": False, "boundary": True, "grid": 100, "highlight": False},
        {"name": "Decision Boundary (Grid 200)", "importance": False, "boundary": True, "grid": 200, "highlight": False},
        {"name": "Combined features (Grid 100)", "importance": True, "boundary": True, "grid": 100, "highlight": True},
    ]

    # ----------------- PART 2: Feature Overhead Benchmarking (C=2) -----------------
    print("\n" + "="*60)
    print("Starting Feature Overhead Benchmarking Experiment (C=2)...")
    print("Baseline configuration: N = 2000, F = 100, C = 2")
    print(f"Running each configuration {n_trials} times.")
    print("-" * 60)
    
    overhead_results = []
    for cfg in overhead_configs:
        trial_times = []
        for trial in range(n_trials):
            elapsed = benchmark_feature_config(
                n_samples=2000, n_features=100, n_classes=2, temp_path=temp_path,
                use_importance=cfg["importance"], draw_boundary=cfg["boundary"],
                grid_points=cfg["grid"], highlight_misclassifications=cfg["highlight"]
            )
            trial_times.append(elapsed)
        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        print(f"{cfg['name']:30s} | Time: {avg_time:.4f}s (std: {std_time:.4f}s)")
        overhead_results.append({
            "name": cfg["name"],
            "avg_time": avg_time,
            "std_time": std_time
        })
        
    df_oh = pd.DataFrame(overhead_results)
    csv_oh_path = os.path.join(RESULTS_DIR, "speed_test_overhead_results.csv")
    df_oh.to_csv(csv_oh_path, index=False)
    print(f"\nSaved raw overhead results to {csv_oh_path}")
    
    # Construct C=2 Overhead LaTeX Table (Narrow 2-Column format)
    latex_oh_lines = []
    latex_oh_lines.append(r"\begin{table}[htbp]")
    latex_oh_lines.append(r"\centering")
    latex_oh_lines.append(r"\caption{Execution Overhead of Additional Features ($N=2\,000, F=100, C=2$)}")
    latex_oh_lines.append(r"\label{tab:dca_execution_overhead_c2}")
    latex_oh_lines.append(r"\begin{tabular}{l|c}")
    latex_oh_lines.append(r"\hline")
    latex_oh_lines.append(r"\textbf{Feature Configuration} & \textbf{Execution Time (s)} \\")
    latex_oh_lines.append(r"\hline")
    for row in overhead_results:
        time_str = f"${row['avg_time']:.4f} \\pm {row['std_time']:.4f}$"
        latex_oh_lines.append(f"{row['name']} & {time_str} \\\\")
    latex_oh_lines.append(r"\hline")
    latex_oh_lines.append(r"\end{tabular}")
    latex_oh_lines.append(r"\end{table}")
    oh_latex_code = "\n".join(latex_oh_lines)

    # ----------------- PART 3: Feature Overhead Benchmarking (C=10) -----------------
    print("\n" + "="*60)
    print("Starting Feature Overhead Benchmarking Experiment (C=10)...")
    print("Baseline configuration: N = 2000, F = 100, C = 10")
    print(f"Running each configuration {n_trials} times.")
    print("-" * 60)
    
    overhead_c10_results = []
    for cfg in overhead_configs:
        trial_times = []
        for trial in range(n_trials):
            elapsed = benchmark_feature_config(
                n_samples=2000, n_features=100, n_classes=10, temp_path=temp_path,
                use_importance=cfg["importance"], draw_boundary=cfg["boundary"],
                grid_points=cfg["grid"], highlight_misclassifications=cfg["highlight"]
            )
            trial_times.append(elapsed)
        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        print(f"{cfg['name']:30s} | Time: {avg_time:.4f}s (std: {std_time:.4f}s)")
        overhead_c10_results.append({
            "name": cfg["name"],
            "avg_time": avg_time,
            "std_time": std_time
        })
        
    df_oh_c10 = pd.DataFrame(overhead_c10_results)
    csv_oh_c10_path = os.path.join(RESULTS_DIR, "speed_test_overhead_c10_results.csv")
    df_oh_c10.to_csv(csv_oh_c10_path, index=False)
    print(f"\nSaved raw C=10 overhead results to {csv_oh_c10_path}")
    
    # Construct C=10 Overhead LaTeX Table (Narrow 2-Column format)
    latex_oh_c10_lines = []
    latex_oh_c10_lines.append(r"\begin{table}[htbp]")
    latex_oh_c10_lines.append(r"\centering")
    latex_oh_c10_lines.append(r"\caption{Execution Overhead of Additional Features ($N=2\,000, F=100, C=10$)}")
    latex_oh_c10_lines.append(r"\label{tab:dca_execution_overhead_c10}")
    latex_oh_c10_lines.append(r"\begin{tabular}{l|c}")
    latex_oh_c10_lines.append(r"\hline")
    latex_oh_c10_lines.append(r"\textbf{Feature Configuration} & \textbf{Execution Time (s)} \\")
    latex_oh_c10_lines.append(r"\hline")
    for row in overhead_c10_results:
        time_str = f"${row['avg_time']:.4f} \\pm {row['std_time']:.4f}$"
        latex_oh_c10_lines.append(f"{row['name']} & {time_str} \\\\")
    latex_oh_c10_lines.append(r"\hline")
    latex_oh_c10_lines.append(r"\end{tabular}")
    latex_oh_c10_lines.append(r"\end{table}")
    oh_c10_latex_code = "\n".join(latex_oh_c10_lines)
    
    # Save LaTeX files
    with open(os.path.join(RESULTS_DIR, "speed_test_table.tex"), "w") as f:
        f.write(size_latex_code)
    with open(os.path.join(RESULTS_DIR, "speed_test_overhead_table.tex"), "w") as f:
        f.write(oh_latex_code)
    with open(os.path.join(RESULTS_DIR, "speed_test_overhead_c10_table.tex"), "w") as f:
        f.write(oh_c10_latex_code)
        
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    # Print the tables out to terminal
    print("\nGenerated LaTeX Size Table:")
    print("=" * 80)
    print(size_latex_code)
    print("=" * 80)
    
    print("\nGenerated LaTeX C=2 Overhead Table:")
    print("=" * 80)
    print(oh_latex_code)
    print("=" * 80)

    print("\nGenerated LaTeX C=10 Overhead Table:")
    print("=" * 80)
    print(oh_c10_latex_code)
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
