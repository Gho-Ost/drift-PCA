import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_real_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "real", "gen")
    script_path = os.path.join(base_dir, "scripts", "run_dca.py")
    
    simple_comp_dir = os.path.join(base_dir, "results", "real", "simple_comparisons")
    window_comb_dir = os.path.join(base_dir, "results", "real", "combination_plots")
    
    os.makedirs(simple_comp_dir, exist_ok=True)
    os.makedirs(window_comb_dir, exist_ok=True)
    
    datasets = ["elec", "keystroke", "insects", "gassensor", "noaa"]
    
    logger.info("Starting Real-World Dataset DCA experiments with SVC model...")
    
    # ---------------------------------------------------------
    # Batch 1: Drift Pre/Post (Simple Comparisons)
    # ---------------------------------------------------------
    logger.info("=== Running Batch 1: Drift Pre/Post (Simple Comparisons) ===")
    for ds in datasets:
        for boundary in ["drift1", "drift2"]:
            pre_file = os.path.join(data_dir, f"{ds}_{boundary}_pre.csv")
            post_file = os.path.join(data_dir, f"{ds}_{boundary}_post.csv")
            
            if not os.path.exists(pre_file) or not os.path.exists(post_file):
                logger.warning(f"Files not found for {ds} ({boundary}), skipping.")
                continue
                
            display_name = f"{ds}_{boundary}"
            logger.info(f"Running DCA on {display_name}...")
            
            cmd = [
                sys.executable, script_path,
                "--data_dir", data_dir,
                "--results_dir", simple_comp_dir,
                "--dataset", display_name,
                "--pre_file", pre_file,
                "--post_file", post_file,
                "--drift_mode", "per-class",
                "--color_scheme", "class",
                "--drift_type", "sudden",
                "--feature_importance",
                "--highlight_misclassifications",
                "--grid_points", "200",
                "--model", "svc"
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to process {display_name}: {e}")

    # ---------------------------------------------------------
    # Batch 2: Consecutive Window Combinations
    # ---------------------------------------------------------
    logger.info("=== Running Batch 2: Consecutive Window Combinations ===")
    combinations = [
        ("w1", "w2", "w1_vs_w2"),
        ("w2", "w3", "w2_vs_w3"),
        ("w1", "w3", "w1_vs_w3")
    ]
    
    for ds in datasets:
        for pre_w, post_w, comb_label in combinations:
            pre_file = os.path.join(data_dir, f"{ds}_{pre_w}.csv")
            post_file = os.path.join(data_dir, f"{ds}_{post_w}.csv")
            
            if not os.path.exists(pre_file) or not os.path.exists(post_file):
                logger.warning(f"Files not found for {ds} ({comb_label}), skipping.")
                continue
                
            display_name = f"{ds}_{comb_label}"
            logger.info(f"Running DCA on {display_name}...")
            
            cmd = [
                sys.executable, script_path,
                "--data_dir", data_dir,
                "--results_dir", window_comb_dir,
                "--dataset", display_name,
                "--pre_file", pre_file,
                "--post_file", post_file,
                "--drift_mode", "per-class",
                "--color_scheme", "class",
                "--drift_type", "sudden",
                "--feature_importance",
                "--highlight_misclassifications",
                "--grid_points", "200",
                "--model", "svc"
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to process {display_name}: {e}")
                
    logger.info("All real-world experiments finished successfully.")

if __name__ == "__main__":
    run_real_experiments()
