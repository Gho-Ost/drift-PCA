import os
import sys
import glob
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_gradrec_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "synthetic", "gradrec")
    script_path = os.path.join(base_dir, "scripts", "run_dca.py")
    
    baseline_highest_dir = os.path.join(base_dir, "results", "synthetic", "gradrec_baseline_vs_highest")
    combinations_dir = os.path.join(base_dir, "results", "synthetic", "gradrec_combinations")
    
    os.makedirs(baseline_highest_dir, exist_ok=True)
    os.makedirs(combinations_dir, exist_ok=True)
    
    generators = ["linear", "rollingtorus", "chocolaterotation"]
    drift_types = ["gradual", "recurrent"]
    features = [5]
    
    logger.info("Starting THU Gradual/Recurrent DCA experiments with SVC model...")
    
    # ---------------------------------------------------------
    # Batch 1: Baseline vs Highest Drift State
    # ---------------------------------------------------------
    logger.info("=== Running Batch 1: Baseline vs Highest Drift ===")
    for gen in generators:
        for drift in drift_types:
            for f in features:
                dataset_name = f"{gen}_{drift}_f{f}"
                pre_file = os.path.join(data_dir, f"{dataset_name}_w1.csv")
                
                # Gradual highest drift is w3, Recurrent highest drift is w2
                if drift == "gradual":
                    post_file = os.path.join(data_dir, f"{dataset_name}_w3.csv")
                    label_comb = "w1_vs_w3"
                else:
                    post_file = os.path.join(data_dir, f"{dataset_name}_w2.csv")
                    label_comb = "w1_vs_w2"
                    
                if not os.path.exists(pre_file) or not os.path.exists(post_file):
                    logger.warning(f"Files not found for {dataset_name}, skipping.")
                    continue
                    
                display_name = f"{dataset_name}_{label_comb}"
                logger.info(f"Running DCA on {display_name}...")
                
                cmd = [
                    sys.executable, script_path,
                    "--data_dir", data_dir,
                    "--results_dir", baseline_highest_dir,
                    "--dataset", display_name,
                    "--pre_file", pre_file,
                    "--post_file", post_file,
                    "--drift_mode", "per-class",
                    "--color_scheme", "class",
                    "--drift_type", "gradual",
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
    # Batch 2: Linear Dataset Combinations (w1-w2, w2-w3, w1-w3)
    # ---------------------------------------------------------
    logger.info("=== Running Batch 2: Linear Dataset Combinations ===")
    combinations = [
        ("w1", "w2", "w1_vs_w2"),
        ("w2", "w3", "w2_vs_w3"),
        ("w1", "w3", "w1_vs_w3")
    ]
    
    for drift in drift_types:
        for f in features:
            for pre_w, post_w, comb_label in combinations:
                dataset_prefix = f"linear_{drift}_f{f}"
                pre_file = os.path.join(data_dir, f"{dataset_prefix}_{pre_w}.csv")
                post_file = os.path.join(data_dir, f"{dataset_prefix}_{post_w}.csv")
                
                if not os.path.exists(pre_file) or not os.path.exists(post_file):
                    logger.warning(f"Files not found for {dataset_prefix} ({comb_label}), skipping.")
                    continue
                    
                display_name = f"{dataset_prefix}_{comb_label}"
                logger.info(f"Running DCA on {display_name}...")
                
                cmd = [
                    sys.executable, script_path,
                    "--data_dir", data_dir,
                    "--results_dir", combinations_dir,
                    "--dataset", display_name,
                    "--pre_file", pre_file,
                    "--post_file", post_file,
                    "--drift_mode", "per-class",
                    "--color_scheme", "class",
                    "--drift_type", "gradual",
                    "--feature_importance",
                    "--highlight_misclassifications",
                    "--grid_points", "200",
                    "--model", "svc"
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to process {display_name}: {e}")
                    
    logger.info("All experiments finished successfully.")

if __name__ == "__main__":
    run_gradrec_experiments()
