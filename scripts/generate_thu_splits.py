import os
import sys
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Add repo base to sys.path to allow importing DataStreamGenerator natively
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
thu_dir = os.path.join(base_dir, "THU-Concept-Drift-Datasets-v1.0")
sys.path.append(thu_dir)

from DataStreamGenerator import DataStreamGenerator

def generate_splits():
    out_dir = os.path.join(base_dir, "data", "thu_stream_datasets")
    os.makedirs(out_dir, exist_ok=True)
    
    feature_targets = [3, 5, 7, 11]
    
    # We will generate a specific set of representative datasets for sudden and gradual
    dataset_methods = [
        ("linear_sudden", "Linear_Sudden_Rotation"),
        ("linear_gradual", "Linear_Gradual_Rotation"),
        ("nonlinear_sudden_cakerotation", "Nonlinear_Sudden_CakeRotation"),
        ("nonlinear_gradual_cakerotation", "Nonlinear_Gradual_CakeRotation"),
        ("nonlinear_sudden_chocolaterotation", "Nonlinear_Sudden_ChocolateRotation"),
        ("nonlinear_gradual_chocolaterotation", "Nonlinear_Gradual_ChocolateRotation"),
        ("nonlinear_sudden_rollingtorus", "Nonlinear_Sudden_RollingTorus"),
        ("nonlinear_gradual_rollingtorus", "Nonlinear_Gradual_RollingTorus")
    ]
    
    for f in feature_targets:
        logger.info(f"--- Generating THU datasets with {f} features ---")
        generator = DataStreamGenerator(class_count=2, attribute_count=f, sample_count=10000, noise=False, redunce_variable=False)
        
        for ds_name, method_name in dataset_methods:
            logger.info(f"Generating {ds_name}...")
            
            method = getattr(generator, method_name)
            
            if "Linear" in method_name:
                if "Rotation" in method_name:
                    data, label = method(plot=False, save=False, x_spinaxis=0.0, y_spinaxis=0.0)
                else:
                    data, label = method(plot=False, save=False, x_pass=0.0, y_pass=0.0) # Abrupt
            else:
                data, label = method(plot=False, save=False)
                
            # Extract fully pure drift boundaries
            # For sudden: 20% interval of 10000 = 2000 points.
            # Pre = [0:2000], Post = [2000:4000]
            # For gradual: Concept drifts continuously from 0 to 10000. 
            # To get fully distinct concepts, we take start [0:2000] and end [8000:10000].
            
            data_pre = data[0:2000]
            label_pre = label[0:2000]
            
            if "gradual" in ds_name:
                data_post = data[8000:10000]
                label_post = label[8000:10000]
            else:
                data_post = data[2000:4000]
                label_post = label[2000:4000]
            
            out_pre_name = os.path.join(out_dir, f"thu_{ds_name}_f{f}_pre.csv")
            out_post_name = os.path.join(out_dir, f"thu_{ds_name}_f{f}_post.csv")
            
            cols = [f"x{i+1}" for i in range(f)] + ["label"]
            df_pre = pd.DataFrame(np.column_stack((data_pre, label_pre)), columns=cols)
            df_post = pd.DataFrame(np.column_stack((data_post, label_post)), columns=cols)
            
            df_pre['label'] = df_pre['label'].astype(int)
            df_post['label'] = df_post['label'].astype(int)
            
            df_pre.to_csv(out_pre_name, index=False)
            df_post.to_csv(out_post_name, index=False)
            
    logger.info("Successfully generated pure pre/post splits with exact boundaries using THU Generator natively.")

if __name__ == "__main__":
    generate_splits()
