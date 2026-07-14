import os
import sys
import subprocess
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_cmd(args_list):
    cmd = [sys.executable, "scripts/run_dca.py"] + args_list
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        logger.error(f"Stdout:\n{result.stdout}")
        logger.error(f"Stderr:\n{result.stderr}")
        raise RuntimeError(f"Failed to run command: {' '.join(cmd)}")
    else:
        logger.info("Command completed successfully.")
        if result.stdout:
            logger.debug(f"Stdout:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"Stderr:\n{result.stderr}")

def main():
    # Base directory for showcase
    showcase_dir = os.path.join("results", "rbf_flexibility_showcase")
    os.makedirs(showcase_dir, exist_ok=True)
    
    # 5 Scenarios configuration
    scenarios = [
        {
            "name": "scenario1_max_features",
            "args": [
                "--data_dir", "data/synthetic/gen",
                "--results_dir", os.path.join(showcase_dir, "scenario1_max_features"),
                "--dataset", "rbf",
                "--drift_mode", "per-class",
                "--color_scheme", "class",
                "--highlight_misclassifications",
                "--feature_importance",
                "--drift_type", "gradual"
            ]
        },
        {
            "name": "scenario2_discrete_boundary",
            "args": [
                "--data_dir", "data/synthetic/gen",
                "--results_dir", os.path.join(showcase_dir, "scenario2_discrete_boundary"),
                "--dataset", "rbf",
                "--drift_mode", "per-class",
                "--color_scheme", "class",
                "--discrete_boundary",
                "--drift_type", "sudden"
            ]
        },
        {
            "name": "scenario3_drift_coloring",
            "args": [
                "--data_dir", "data/synthetic/gen",
                "--results_dir", os.path.join(showcase_dir, "scenario3_drift_coloring"),
                "--dataset", "rbf",
                "--drift_mode", "global",
                "--color_scheme", "drift",
                "--drift_type", "sudden"
            ]
        },
        {
            "name": "scenario4_unsupervised_drift",
            "args": [
                "--data_dir", "data/synthetic/gen",
                "--results_dir", os.path.join(showcase_dir, "scenario4_unsupervised_drift"),
                "--dataset", "rbf",
                "--drift_mode", "data",
                "--no_target"
            ]
        },
        {
            "name": "scenario5_global_no_boundary",
            "args": [
                "--data_dir", "data/synthetic/gen",
                "--results_dir", os.path.join(showcase_dir, "scenario5_global_no_boundary"),
                "--dataset", "rbf",
                "--drift_mode", "global",
                "--color_scheme", "class",
                "--no_boundary",
                "--drift_type", "gradual"
            ]
        }
    ]

    for idx, sc in enumerate(scenarios, 1):
        logger.info(f"\n--- Running Scenario {idx}: {sc['name']} ---")
        try:
            run_cmd(sc["args"])
            
            # Copy output to the main showcase directory with descriptive name
            src_img = os.path.join(showcase_dir, sc["name"], "rbf_dca.png")
            dst_img = os.path.join(showcase_dir, f"{sc['name']}.png")
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                logger.info(f"Successfully copied output to {dst_img}")
            else:
                logger.error(f"Expected output file not found: {src_img}")
        except Exception as e:
            logger.error(f"Error executing scenario {sc['name']}: {e}")
            sys.exit(1)

    logger.info("\nAll scenarios executed successfully! Visualizations saved in results/rbf_flexibility_showcase/")

if __name__ == "__main__":
    main()
