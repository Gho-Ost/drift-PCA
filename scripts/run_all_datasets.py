import os
import subprocess
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_datasets(data_dir):
    """
    Looks at all files in the given directory and extracts unique 
    dataset names (assuming files are named {name}_pre.csv and {name}_post.csv)
    """
    if not os.path.isdir(data_dir):
        logging.error(f"Directory {data_dir} not found.")
        return []
        
    datasets = set()
    for filename in os.listdir(data_dir):
        if filename.endswith("_pre.csv"):
            dataset_name = filename.replace("_pre.csv", "")
            datasets.add(dataset_name)
        elif filename.endswith("_post.csv"):
            dataset_name = filename.replace("_post.csv", "")
            datasets.add(dataset_name)
            
    return sorted(list(datasets))

def run_all_datasets(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    python_script = os.path.join(script_dir, "run_pca_comparison.py")
    data_dir = os.path.join(project_root, "data", "synth_stream_datasets")
    
    datasets = get_datasets(data_dir)
    
    if not datasets:
        logging.warning(f"No datasets found in {data_dir}.")
        return

    logging.info(f"Found {len(datasets)} datasets to process: {', '.join(datasets)}")
    
    success_count = 0
    
    for dataset in datasets:
        logging.info(f"\n[{success_count+1}/{len(datasets)}] Running analysis for dataset: {dataset}")
        
        cmd = [
            sys.executable, python_script,
            "--data_dir", data_dir,
            "--dataset", dataset,
            "--results_dir", args.results_dir,
            "--add_drift_vectors",
            "--add_anchor_point"
        ]
        
        try:
            # Run the command
            result = subprocess.run(cmd, check=True)
            logging.info(f"Completed {dataset}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred processing {dataset}")
            
    logging.info(f"\n========================================")
    logging.info(f"Finished processing datasets. {success_count}/{len(datasets)} successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA Comparison on all datasets")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results")
    args = parser.parse_args()
    
    run_all_datasets(args)
