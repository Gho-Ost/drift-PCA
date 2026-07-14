import os
import sys
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
thu_dir = os.path.join(base_dir, "THU-Concept-Drift-Datasets-v1.0")
sys.path.append(thu_dir)

from DataStreamGenerator import DataStreamGenerator

def generate_splits():
    out_dir = os.path.join(base_dir, "data", "synthetic", "gradrec")
    os.makedirs(out_dir, exist_ok=True)
    
    feature_targets = [3, 5, 7, 11]
    
    dataset_methods = [
        # (dataset_prefix, generator_method_name, drift_type)
        ("linear_gradual", "Linear_Gradual_Rotation", "gradual"),
        ("linear_recurrent", "Linear_Recurrent_Rotation", "recurrent"),
        ("rollingtorus_gradual", "Nonlinear_Gradual_RollingTorus", "gradual"),
        ("rollingtorus_recurrent", "Nonlinear_Recurrent_RollingTorus", "recurrent"),
        ("chocolaterotation_gradual", "Nonlinear_Gradual_ChocolateRotation", "gradual"),
        ("chocolaterotation_recurrent", "Nonlinear_Recurrent_ChocolateRotation", "recurrent")
    ]
    
    for f in feature_targets:
        logger.info(f"--- Generating THU gradual/recurrent datasets with {f} features ---")
        generator = DataStreamGenerator(class_count=2, attribute_count=f, sample_count=10000, noise=False, redunce_variable=False)
        
        for ds_name, method_name, drift_type in dataset_methods:
            logger.info(f"Generating {ds_name}...")
            
            method = getattr(generator, method_name)
            
            if "Linear" in method_name:
                # Linear methods require spin axis parameters
                data, label = method(plot=False, save=False, x_spinaxis=0.0, y_spinaxis=0.0)
            else:
                data, label = method(plot=False, save=False)
                
            label = label.flatten()
            
            # Define window boundaries based on drift type
            if drift_type == "recurrent":
                # For recurrent, we want to capture a full cycle: start -> rotation -> back to start
                if "linear" in ds_name:
                    # Linear recurrent has abrupt concept changes every 1000 samples
                    w1_start, w1_end = 0, 1000
                    w2_start, w2_end = 1000, 2000
                    w3_start, w3_end = 2000, 3000
                else:
                    # Nonlinear recurrent methods shift gradually over 1000 sample cycles:
                    # Cycle 0 (0-1000): shifts from 0 to max
                    # Cycle 1 (1000-2000): shifts from max to 0
                    # Cycle 2 (2000-3000): shifts from 0 to max
                    # Cycle 3 (3000-4000): shifts from max to 0
                    # To capture the pure unrotated concept at the start, max rotated, and return to start:
                    # We take windows centered at T=2000 (concept 0), T=3000 (concept max), T=4000 (concept 0)
                    w1_start, w1_end = 1500, 2500
                    w2_start, w2_end = 2500, 3500
                    w3_start, w3_end = 3500, 4500
            else:
                # For gradual, we want to capture progression of drift from the beginning:
                # Window size of 1500 allows a strong and observable distribution shift.
                w1_start, w1_end = 0, 1500
                w2_start, w2_end = 1500, 3000
                w3_start, w3_end = 3000, 4500
                
            cols = [f"x{i+1}" for i in range(f)] + ["label"]
            
            # Slice windows
            w1_data = data[w1_start:w1_end]
            w1_label = label[w1_start:w1_end]
            df_w1 = pd.DataFrame(np.column_stack((w1_data, w1_label)), columns=cols)
            df_w1['label'] = df_w1['label'].astype(int)
            
            w2_data = data[w2_start:w2_end]
            w2_label = label[w2_start:w2_end]
            df_w2 = pd.DataFrame(np.column_stack((w2_data, w2_label)), columns=cols)
            df_w2['label'] = df_w2['label'].astype(int)
            
            w3_data = data[w3_start:w3_end]
            w3_label = label[w3_start:w3_end]
            df_w3 = pd.DataFrame(np.column_stack((w3_data, w3_label)), columns=cols)
            df_w3['label'] = df_w3['label'].astype(int)
            
            # Save files
            df_w1.to_csv(os.path.join(out_dir, f"{ds_name}_f{f}_w1.csv"), index=False)
            df_w2.to_csv(os.path.join(out_dir, f"{ds_name}_f{f}_w2.csv"), index=False)
            df_w3.to_csv(os.path.join(out_dir, f"{ds_name}_f{f}_w3.csv"), index=False)
            
            logger.info(f"Saved {ds_name}_f{f} windows (w1: [{w1_start}:{w1_end}], w2: [{w2_start}:{w2_end}], w3: [{w3_start}:{w3_end}])")
            
    logger.info("Successfully generated all splits.")

if __name__ == "__main__":
    generate_splits()
