import os
import sys
import glob
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_scenarios():
    script_path = os.path.join(PROJECT_ROOT, "scripts", "run_dca.py")
    python_exe = sys.executable
    
    # 1. Run DCA on Class-Aggregate Scenarios
    agg_dir = os.path.join(PROJECT_ROOT, "data", "synthetic", "agg")
    agg_results_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", "agg_experiments")
    os.makedirs(agg_results_dir, exist_ok=True)
    
    pre_files_agg = glob.glob(os.path.join(agg_dir, "*_pre.csv"))
    agg_datasets = sorted([os.path.basename(f).replace("_pre.csv", "") for f in pre_files_agg])
    
    logger.info("=============================================================")
    logger.info("Running DCA Experiments on Class-Aggregate Scenarios...")
    logger.info("=============================================================")
    
    if not agg_datasets:
        logger.warning(f"No datasets found in {agg_dir}. Please run dataset generation first.")
    
    for idx, ds in enumerate(agg_datasets, 1):
        logger.info(f"--- [{idx}/{len(agg_datasets)}] DCA on class-aggregate scenario: {ds} ---")
        
        # Run in unsupervised data mode (no boundaries as target classes are random)
        cmd = [
            python_exe, script_path,
            "--data_dir", agg_dir,
            "--results_dir", agg_results_dir,
            "--dataset", ds,
            "--drift_mode", "data",
            "--no_target",
            "--no_boundary",
            "--color_scheme", "drift"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully processed class-aggregate {ds}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed for {ds}: {e}")
            
    # 2. Run DCA on Class-Specific Scenarios
    class_dir = os.path.join(PROJECT_ROOT, "data", "synthetic", "class")
    class_results_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", "class_experiments")
    os.makedirs(class_results_dir, exist_ok=True)
    
    pre_files_class = glob.glob(os.path.join(class_dir, "*_pre.csv"))
    class_datasets = sorted([os.path.basename(f).replace("_pre.csv", "") for f in pre_files_class])
    
    logger.info("=============================================================")
    logger.info("Running DCA Experiments on Class-Specific Scenarios...")
    logger.info("=============================================================")
    
    if not class_datasets:
        logger.warning(f"No datasets found in {class_dir}. Please run dataset generation first.")
        
    for idx, ds in enumerate(class_datasets, 1):
        logger.info(f"--- [{idx}/{len(class_datasets)}] DCA on class-specific scenario: {ds} ---")
        
        # Run in supervised class mode with boundaries, feature importances, and misclassifications
        cmd = [
            python_exe, script_path,
            "--data_dir", class_dir,
            "--results_dir", class_results_dir,
            "--dataset", ds,
            "--drift_mode", "per-class",
            "--model", "svc",
            "--color_scheme", "class",
            "--feature_importance",
            "--highlight_misclassifications",
            "--grid_points", "200"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully processed class-specific {ds}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed for {ds}: {e}")


if __name__ == "__main__":
    run_scenarios()
