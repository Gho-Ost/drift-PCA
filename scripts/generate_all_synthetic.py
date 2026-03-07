import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
scenarios_file = os.path.join(project_root, "synthetic_data.json")

# Load the drift scenarios from scenarios.json
with open(scenarios_file, "r") as f:
    scenarios = json.load(f)

def run_generation():
    python_exe = sys.executable
    script_path = os.path.join(script_dir, "generate_synthetic_drift.py")
    
    success_count = 0
    
    for i, s in enumerate(scenarios):
        logging.info(f"--- Generating scenario {i+1}/{len(scenarios)}: {s['name']} ---")
        logging.info(f"Description: {s['desc']}")
        
        cmd = [
            python_exe, script_path,
            "--name", s["name"], 
            "--data_dir", "data\\synth_stream_datasets", 
            "--vis_dir", "data\\synth_datasets_vis",
            "--means_pre"
        ] + [str(m) for m in s["means_pre"]] + [
            "--stds_pre"
        ] + [str(std) for std in s["stds_pre"]] + [
            "--means_post"
        ] + [str(m) for m in s["means_post"]] + [
            "--stds_post"
        ] + [str(std) for std in s["stds_post"]]
        
        # Add sample size if needed, using default 4000
        
        try:
            # Run the command and wait for completion
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(f"Successfully generated {s['name']}")
            # logging.debug(result.stdout)
            success_count += 1
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate {s['name']}")
            logging.error(f"Error output:\n{e.stderr}")
            
    logging.info(f"Finished generating datasets. {success_count}/{len(scenarios)} successful.")


if __name__ == "__main__":
    run_generation()
