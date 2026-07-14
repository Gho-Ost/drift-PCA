import os
import sys
import glob
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "synthetic", "gen")
    script_path = os.path.join(base_dir, "scripts", "run_dca.py")
    results_dir = os.path.join(base_dir, "results", "synthetic", "gen")
    
    os.makedirs(results_dir, exist_ok=True)

    # Look for all *_pre.csv files in data_dir
    pre_files = glob.glob(os.path.join(data_dir, "*_pre.csv"))
    datasets = [os.path.basename(f).replace("_pre.csv", "") for f in pre_files]
    
    if not datasets:
        logger.error(f"No datasets found in {data_dir}. Please run dataset generation scripts first.")
        return

    # Sort datasets to run sequentially
    datasets.sort()

    logger.info(f"Found {len(datasets)} datasets. Starting DCA experiments...")
    
    for idx, ds in enumerate(datasets, 1):
        logger.info(f"--- [{idx}/{len(datasets)}] Running DCA on {ds} ---")
        
        # Build command invoking run_dca.py with specified arguments
        cmd = [
            sys.executable, script_path,
            "--data_dir", data_dir,
            "--results_dir", results_dir,
            "--dataset", ds,
            "--drift_mode", "per-class",
            "--color_scheme", "class",
            "--drift_type", "gradual",
            "--feature_importance",
            "--highlight_misclassifications",
            "--grid_points", "200"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully processed {ds}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed for {ds}: {e}")

if __name__ == "__main__":
    run_experiments()

