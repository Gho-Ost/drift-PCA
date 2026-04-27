import os
import glob
import subprocess
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def run_experiments():
    parser = argparse.ArgumentParser(description="Run Batch Algorithm Comparisons on Synthetic Data")
    parser.add_argument("--strategy", type=str, choices=["pre_fit", "joint_fit", "separate_fit"], default="pre_fit",
                        help="Fallback fitting strategy configuration for batch dataset scripts.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "synth_stream_datasets")
    script_path = os.path.join(base_dir, "scripts", "run_algorithm_comparison.py")
    results_base_dir = os.path.join(base_dir, "results_comparison", "synth")
    
    os.makedirs(results_base_dir, exist_ok=True)
    
    pre_files = glob.glob(os.path.join(data_dir, "*_pre.csv"))
    datasets = [os.path.basename(f).replace("_pre.csv", "") for f in pre_files]
    
    if not datasets:
        logger.error(f"No datasets found in {data_dir}.")
        return

    logger.info(f"Found {len(datasets)} synthetic datasets. Starting multi-algorithm experiments using {args.strategy} strategy...")
    
    for idx, ds in enumerate(datasets, 1):
        logger.info(f"--- [{idx}/{len(datasets)}] Running Algorithm Comparison on {ds} ---")
        
        # Build command invoking run_algorithm_comparison.py
        cmd = [
            "python", script_path,
            "--data_dir", data_dir,
            "--results_dir", results_base_dir,
            "--dataset", ds,
            "--strategy", args.strategy
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully processed {ds}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed for {ds}: {e}")

if __name__ == "__main__":
    run_experiments()
