import subprocess
import sys
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_generation():
    python_exe = sys.executable
    script_path = os.path.join(SCRIPT_DIR, "generate_synthetic_drift.py")
    
    # 1. Generate Class-Aggregate Datasets
    agg_json = os.path.join(PROJECT_ROOT, "data", "synthetic_data.json")
    agg_data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic", "agg")
    agg_vis_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", "vis")
    
    logger.info("=============================================================")
    logger.info("Generating Class-Aggregate Synthetic Datasets...")
    logger.info("=============================================================")
    
    cmd_agg = [
        python_exe, script_path,
        "--json", agg_json,
        "--data_dir", agg_data_dir,
        "--vis_dir", agg_vis_dir
    ]
    
    try:
        subprocess.run(cmd_agg, check=True)
        logger.info("Successfully completed class-aggregate dataset generation.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate class-aggregate datasets: {e}")
        
    # 2. Generate Class-Specific Datasets
    class_json = os.path.join(PROJECT_ROOT, "data", "synthetic_data_classes.json")
    class_data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic", "class")
    class_vis_dir = os.path.join(PROJECT_ROOT, "results", "synthetic", "vis")
    
    logger.info("=============================================================")
    logger.info("Generating Class-Specific Synthetic Datasets...")
    logger.info("=============================================================")
    
    cmd_class = [
        python_exe, script_path,
        "--json", class_json,
        "--data_dir", class_data_dir,
        "--vis_dir", class_vis_dir
    ]
    
    try:
        subprocess.run(cmd_class, check=True)
        logger.info("Successfully completed class-specific dataset generation.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate class-specific datasets: {e}")


if __name__ == "__main__":
    run_generation()
